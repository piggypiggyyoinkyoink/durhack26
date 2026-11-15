import struct
import zlib
import threading
import time
import os
from datetime import datetime
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image

# ============================================================================
# MODEM CONFIGURATION & CONSTANTS
# ============================================================================
SAMPLE_RATE = 44100
SYMBOL_DURATION = 0.100  # 100 ms total per symbol
TONE_DURATION = 0.080    # 80 ms tone + 20 ms guard silence
SYMBOL_SAMPLES = int(SAMPLE_RATE * SYMBOL_DURATION)
WINDOW_SAMPLES = int(SAMPLE_RATE * 0.060)
STEP_SAMPLES = int(SAMPLE_RATE * 0.010)
CHUNK_SIZE = 32

# Payload Type Identifiers
TYPE_TEXT = 0x01
TYPE_IMAGE = 0x02
TYPE_FILE = 0x03
TYPE_ACK = 0x04
TYPE_NAK = 0x05
TYPE_HS_REQ = 0x10
TYPE_HS_RESP = 0x11
TYPE_TARGET_REQ = 0x12
TYPE_TARGET_RESP = 0x13
TYPE_FORWARD_REQ = 0x20   # Router -> Target: Paging data delivery
TYPE_FORWARD_RESP = 0x21  # Target -> Router: Ready ACK
TYPE_FORWARD_FAIL = 0x22  # Router -> Sender: Target unreachable notification

DATA_FREQS = np.array([8000 + i * 250 for i in range(16)])
SYNC_FREQ_A = 12250
SYNC_FREQ_B = 12750
ALL_FREQS = np.append(DATA_FREQS, [SYNC_FREQ_A, SYNC_FREQ_B])

FFT_SIZE = 8192
HANN = np.hanning(WINDOW_SAMPLES)
FFT_FREQS = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
FREQ_BINS = [np.argmin(np.abs(FFT_FREQS - f)) for f in ALL_FREQS]
SNR_THRESHOLD = 6.0

PALETTE = [
    "#000000", "#1D2B53", "#7E2553", "#008751",
    "#AB5236", "#5F574F", "#C2C3C7", "#FFF1E8",
    "#FF004D", "#FFA300", "#FFEC27", "#00E436",
    "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA"
]

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

# ============================================================================
# ENCODER & ACOUSTIC FUNCTIONS
# ============================================================================
def generate_tone(freq, duration, total_slot_duration, sample_rate=SAMPLE_RATE, amplitude=0.8):
    tone_samples = int(sample_rate * duration)
    total_samples = int(sample_rate * total_slot_duration)
    t = np.linspace(0, duration, tone_samples, endpoint=False)
    fade_len = int(sample_rate * 0.005)
    
    envelope = np.ones(tone_samples)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    
    wave = amplitude * np.sin(2 * np.pi * freq * t) * envelope
    if (total_samples - tone_samples) > 0:
        wave = np.pad(wave, (0, total_samples - tone_samples), mode='constant')
    return wave

def build_packet(data_payload, payload_type, seq_num=0, total_packets=1):
    if payload_type in (TYPE_ACK, TYPE_NAK, TYPE_HS_RESP, TYPE_TARGET_RESP, TYPE_FORWARD_RESP, TYPE_FORWARD_FAIL):
        full_payload = bytes([payload_type]) + data_payload
    else:
        header = struct.pack('>BHH', payload_type, seq_num, total_packets)
        full_payload = header + data_payload
    crc = zlib.crc32(full_payload) & 0xFFFFFFFF
    return struct.pack('>H', len(full_payload)) + full_payload + struct.pack('>I', crc)

def play_audio_waveform(data_payload, payload_type, seq_num=0, total_packets=1):
    packet = build_packet(data_payload, payload_type, seq_num, total_packets)
    bitstring = "".join(f"{byte:08b}" for byte in packet)
    
    waves = []
    for freq in [SYNC_FREQ_A, SYNC_FREQ_B] * 3:
        waves.append(generate_tone(freq, TONE_DURATION, SYMBOL_DURATION))
    for i in range(0, len(bitstring), 4):
        waves.append(generate_tone(DATA_FREQS[int(bitstring[i:i+4], 2)], TONE_DURATION, SYMBOL_DURATION))
    waves.append(generate_tone(SYNC_FREQ_A, TONE_DURATION, SYMBOL_DURATION))
    waves.append(np.zeros(int(SAMPLE_RATE * 0.150), dtype=np.float32))
    
    sd.play(np.concatenate(waves).astype(np.float32), samplerate=SAMPLE_RATE)
    sd.wait()

def analyze_window(samples):
    samples = samples - np.mean(samples)
    spectrum = np.abs(np.fft.rfft(samples * HANN, n=FFT_SIZE))
    noise_floor = np.median(spectrum[(FFT_FREQS >= 7500) & (FFT_FREQS <= 13500)]) + 1e-9
    strengths = spectrum[FREQ_BINS]
    best_idx = int(np.argmax(strengths))
    snr = strengths[best_idx] / noise_floor
    if snr < SNR_THRESHOLD:
        return None, snr, int(np.argmax(strengths[:16]))
    return ALL_FREQS[best_idx], snr, int(np.argmax(strengths[:16]))

def decode_nibbles_to_bytes(nibbles):
    bitstring = "".join(format(n, "04b") for n in nibbles)
    return bytes([int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)])

# ============================================================================
# ROUTER THREAD WITH STORE-AND-FORWARD PAGING
# ============================================================================
class TkinterRouterThread(threading.Thread):
    def __init__(self, router_addr, log_callback, data_callback):
        super().__init__()
        self.daemon = True
        self.router_addr = router_addr
        self.log_callback = log_callback
        self.data_callback = data_callback
        self.running = False
        self.active_sender = None
        self.active_target = None

    def _send_response(self, stream, ring_buffer, payload, p_type):
        time.sleep(0.3)
        play_audio_waveform(payload, p_type)
        time.sleep(0.15)
        while stream.read_available > 0:
            stream.read(min(stream.read_available, STEP_SAMPLES))
        ring_buffer.fill(0)

    def _listen_for_specific_response(self, expected_type, timeout, stream, ring_buffer):
        start_time = time.time()
        state = "SEARCH_SYNC"
        samples_until_next_read = 0
        nibbles = []
        expected_nibbles = None
        expected_sync = [SYNC_FREQ_A, SYNC_FREQ_B] * 3
        sync_armed = False
        best_sync_snr = 0.0

        while time.time() - start_time < timeout and self.running:
            audio, _ = stream.read(STEP_SAMPLES)
            ring_buffer = np.roll(ring_buffer, -STEP_SAMPLES)
            ring_buffer[-STEP_SAMPLES:] = audio[:, 0]
            
            if state == "SEARCH_SYNC":
                detected_sync = []
                last_snr = 0.0
                for k in range(6):
                    end_offset = -((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.010))
                    freq, snr, _ = analyze_window(ring_buffer[end_offset - WINDOW_SAMPLES:end_offset])
                    detected_sync.append(freq)
                    if k == 5:
                        last_snr = snr

                if detected_sync == expected_sync:
                    if last_snr >= best_sync_snr:
                        best_sync_snr = last_snr
                        sync_armed = True
                    elif sync_armed:
                        state = "RECEIVE_DATA"
                        nibbles = []
                        expected_nibbles = None
                        sync_armed = False
                        best_sync_snr = 0.0
                        samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES
                elif sync_armed:
                    state = "RECEIVE_DATA"
                    nibbles = []
                    expected_nibbles = None
                    sync_armed = False
                    best_sync_snr = 0.0
                    samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES         
            elif state == "RECEIVE_DATA":
                samples_until_next_read -= STEP_SAMPLES
                if samples_until_next_read <= 0:
                    _, _, best_data_idx = analyze_window(ring_buffer[-int(SAMPLE_RATE * 0.010) - WINDOW_SAMPLES:-int(SAMPLE_RATE * 0.010)])
                    nibbles.append(best_data_idx)
                    current_nibble_count = len(nibbles)

                    if current_nibble_count == 4:
                        payload_len = struct.unpack('>H', decode_nibbles_to_bytes(nibbles[:4]))[0]
                        expected_nibbles = (2 + payload_len + 4) * 2

                    if expected_nibbles is not None and current_nibble_count >= expected_nibbles:
                        raw_bytes = decode_nibbles_to_bytes(nibbles)
                        full_payload = raw_bytes[2:-4]
                        received_crc = struct.unpack('>I', raw_bytes[-4:])[0]
                        
                        if received_crc == (zlib.crc32(full_payload) & 0xFFFFFFFF) and len(full_payload) >= 1:
                            p_type = full_payload[0]
                            if p_type == expected_type:
                                return True
                        return False
                    samples_until_next_read += SYMBOL_SAMPLES
        return False

    def _listen_for_ack_nak_internal(self, timeout, stream, ring_buffer):
        start_time = time.time()
        state = "SEARCH_SYNC"
        samples_until_next_read = 0
        nibbles = []
        expected_nibbles = None
        expected_sync = [SYNC_FREQ_A, SYNC_FREQ_B] * 3
        sync_armed = False
        best_sync_snr = 0.0

        while time.time() - start_time < timeout and self.running:
            audio, _ = stream.read(STEP_SAMPLES)
            ring_buffer = np.roll(ring_buffer, -STEP_SAMPLES)
            ring_buffer[-STEP_SAMPLES:] = audio[:, 0]
            
            if state == "SEARCH_SYNC":
                detected_sync = []
                last_snr = 0.0
                for k in range(6):
                    end_offset = -((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.010))
                    freq, snr, _ = analyze_window(ring_buffer[end_offset - WINDOW_SAMPLES:end_offset])
                    detected_sync.append(freq)
                    if k == 5:
                        last_snr = snr

                if detected_sync == expected_sync:
                    if last_snr >= best_sync_snr:
                        best_sync_snr = last_snr
                        sync_armed = True
                    elif sync_armed:
                        state = "RECEIVE_DATA"
                        nibbles = []
                        expected_nibbles = None
                        sync_armed = False
                        best_sync_snr = 0.0
                        samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES
                elif sync_armed:
                    state = "RECEIVE_DATA"
                    nibbles = []
                    expected_nibbles = None
                    sync_armed = False
                    best_sync_snr = 0.0
                    samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES         
            elif state == "RECEIVE_DATA":
                samples_until_next_read -= STEP_SAMPLES
                if samples_until_next_read <= 0:
                    _, _, best_data_idx = analyze_window(ring_buffer[-int(SAMPLE_RATE * 0.010) - WINDOW_SAMPLES:-int(SAMPLE_RATE * 0.010)])
                    nibbles.append(best_data_idx)
                    current_nibble_count = len(nibbles)

                    if current_nibble_count == 4:
                        payload_len = struct.unpack('>H', decode_nibbles_to_bytes(nibbles[:4]))[0]
                        expected_nibbles = (2 + payload_len + 4) * 2

                    if expected_nibbles is not None and current_nibble_count >= expected_nibbles:
                        raw_bytes = decode_nibbles_to_bytes(nibbles)
                        full_payload = raw_bytes[2:-4]
                        received_crc = struct.unpack('>I', raw_bytes[-4:])[0]
                        if received_crc == (zlib.crc32(full_payload) & 0xFFFFFFFF) and len(full_payload) >= 3:
                            p_type = full_payload[0]
                            target_seq = struct.unpack('>H', full_payload[1:3])[0]
                            if p_type == TYPE_ACK:
                                return "ACK", target_seq
                            elif p_type == TYPE_NAK:
                                return "NAK", target_seq
                        return None, None
                    samples_until_next_read += SYMBOL_SAMPLES
        return None, None

    def _page_and_forward_to_target(self, target_addr, sender_addr, payload_type, data, stream, ring_buffer):
        self.log_callback(f"[ROUTER] Paging Target {target_addr}...\n")
        req_payload = struct.pack('>HH', target_addr, sender_addr)
        self._send_response(stream, ring_buffer, req_payload, TYPE_FORWARD_REQ)
        
        target_ready = self._listen_for_specific_response(TYPE_FORWARD_RESP, timeout=5.0, stream=stream, ring_buffer=ring_buffer)
        if not target_ready:
            return False
            
        self.log_callback(f"[ROUTER] Target {target_addr} ready! Forwarding data stream...\n")
        chunks = [data[i:i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)] if len(data) > 0 else [b""]
        total_packets = len(chunks)
        seq_num = 0

        while seq_num < total_packets and self.running:
            self._send_response(stream, ring_buffer, chunks[seq_num], payload_type) # Wait, forward packet with header
            # Actually, use play_audio_waveform with proper sequence headers:
            packet_bytes = build_packet(chunks[seq_num], payload_type, seq_num=seq_num, total_packets=total_packets)
            # Let's send via play_audio_waveform:
            play_audio_waveform(chunks[seq_num], payload_type, seq_num=seq_num, total_packets=total_packets)
            time.sleep(0.3)
            
            ack_status, resp_seq = self._listen_for_ack_nak_internal(timeout=4.5, stream=stream, ring_buffer=ring_buffer)
            if ack_status == "ACK" and resp_seq == seq_num:
                seq_num += 1
            elif ack_status == "NAK":
                seq_num = resp_seq if (resp_seq is not None and 0 <= resp_seq < total_packets) else seq_num
            else:
                return False

        self.log_callback(f"[ROUTER SUCCESS] Data successfully delivered to Target {target_addr}!\n")
        return True

    def _notify_sender_failure(self, sender_addr, target_addr, stream, ring_buffer):
        self.log_callback(f"[ROUTER FAIL] Target {target_addr} unreachable. Notifying Sender {sender_addr}...\n")
        fail_payload = struct.pack('>HH', sender_addr, target_addr)
        self._send_response(stream, ring_buffer, fail_payload, TYPE_FORWARD_FAIL)

    def run(self):
        self.running = True
        self.log_callback(f"[ROUTER START] Listening on Router Address: {self.router_addr}\n")
        
        ring_buffer = np.zeros(SYMBOL_SAMPLES * 7, dtype=np.float32)
        state = "SEARCH_SYNC"
        samples_until_next_read = 0
        nibbles = []
        expected_nibbles = None
        expected_seq = 0
        reassembly_buffer = {}
        active_transfer_type = None

        expected_sync = [SYNC_FREQ_A, SYNC_FREQ_B] * 3
        sync_armed = False
        best_sync_snr = 0.0

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=STEP_SAMPLES) as stream:
                while self.running:
                    audio, _ = stream.read(STEP_SAMPLES)
                    ring_buffer = np.roll(ring_buffer, -STEP_SAMPLES)
                    ring_buffer[-STEP_SAMPLES:] = audio[:, 0]
                    
                    if state == "SEARCH_SYNC":
                        detected_sync = []
                        last_snr = 0.0
                        for k in range(6):
                            end_offset = -((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.010))
                            freq, snr, _ = analyze_window(ring_buffer[end_offset - WINDOW_SAMPLES:end_offset])
                            detected_sync.append(freq)
                            if k == 5:
                                last_snr = snr

                        if detected_sync == expected_sync:
                            if last_snr >= best_sync_snr:
                                best_sync_snr = last_snr
                                sync_armed = True
                            elif sync_armed:
                                state = "RECEIVE_DATA"
                                nibbles = []
                                expected_nibbles = None
                                sync_armed = False
                                best_sync_snr = 0.0
                                samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES
                        elif sync_armed:
                            state = "RECEIVE_DATA"
                            nibbles = []
                            expected_nibbles = None
                            sync_armed = False
                            best_sync_snr = 0.0
                            samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES         
                    elif state == "RECEIVE_DATA":
                        samples_until_next_read -= STEP_SAMPLES
                        if samples_until_next_read <= 0:
                            _, _, best_data_idx = analyze_window(ring_buffer[-int(SAMPLE_RATE * 0.010) - WINDOW_SAMPLES:-int(SAMPLE_RATE * 0.010)])
                            nibbles.append(best_data_idx)
                            current_nibble_count = len(nibbles)

                            if current_nibble_count == 4:
                                payload_len = struct.unpack('>H', decode_nibbles_to_bytes(nibbles[:4]))[0]
                                expected_nibbles = (2 + payload_len + 4) * 2

                            if expected_nibbles is not None and current_nibble_count >= expected_nibbles:
                                raw_bytes = decode_nibbles_to_bytes(nibbles)
                                full_payload = raw_bytes[2:-4]
                                received_crc = struct.unpack('>I', raw_bytes[-4:])[0]
                                
                                if received_crc == (zlib.crc32(full_payload) & 0xFFFFFFFF) and len(full_payload) >= 1:
                                    p_type = full_payload[0]
                                    
                                    # Handshake
                                    if p_type == TYPE_HS_REQ and len(full_payload) >= 5:
                                        r_addr, s_addr = struct.unpack('>HH', full_payload[1:5])
                                        if r_addr == self.router_addr:
                                            self.active_sender = s_addr
                                            self.log_callback(f"[HANDSHAKE] Verified connection from Sender {s_addr}.\n")
                                            self._send_response(stream, ring_buffer, full_payload[1:5], TYPE_HS_RESP)
                                    
                                    # Target Designation
                                    elif p_type == TYPE_TARGET_REQ and len(full_payload) >= 5:
                                        r_addr, t_addr = struct.unpack('>HH', full_payload[1:5])
                                        if r_addr == self.router_addr:
                                            self.active_target = t_addr
                                            self.log_callback(f"[ROUTING] Target designated: {t_addr}.\n")
                                            self._send_response(stream, ring_buffer, full_payload[1:5], TYPE_TARGET_RESP)

                                    # Data Packets Upload
                                    elif p_type in (TYPE_TEXT, TYPE_IMAGE, TYPE_FILE) and len(full_payload) >= 5:
                                        p_type_data, seq_num, total_pkts = struct.unpack('>BHH', full_payload[:5])
                                        chunk_data = full_payload[5:]
                                        
                                        if seq_num == 0 and expected_seq != 0:
                                            reassembly_buffer.clear()
                                            expected_seq = 0
                                            active_transfer_type = p_type_data

                                        if active_transfer_type is None:
                                            active_transfer_type = p_type_data

                                        if seq_num == expected_seq:
                                            reassembly_buffer[seq_num] = chunk_data
                                            self.log_callback(f"[UPLOAD RX] Packet {seq_num + 1}/{total_pkts} received from Sender {self.active_sender}. Sending ACK.\n")
                                            self._send_response(stream, ring_buffer, struct.pack('>H', seq_num), TYPE_ACK)
                                            expected_seq += 1

                                            if expected_seq == total_pkts:
                                                complete_data = b"".join(reassembly_buffer[i] for i in range(total_pkts))
                                                self.log_callback(f"[UPLOAD COMPLETE] All packets gathered from {self.active_sender}. Initiating Target Paging...\n")
                                                self.data_callback(active_transfer_type, complete_data, self.active_sender, self.active_target)
                                                
                                                # --- TRIGGER TARGET PAGING & FORWARDING ---
                                                success = self._page_and_forward_to_target(
                                                    self.active_target, self.active_sender, 
                                                    active_transfer_type, complete_data, stream, ring_buffer
                                                )
                                                if not success:
                                                    self._notify_sender_failure(self.active_sender, self.active_target, stream, ring_buffer)

                                                reassembly_buffer.clear()
                                                expected_seq = 0
                                                active_transfer_type = None
                                        elif seq_num < expected_seq:
                                            self._send_response(stream, ring_buffer, struct.pack('>H', seq_num), TYPE_ACK)
                                        else:
                                            self._send_response(stream, ring_buffer, struct.pack('>H', expected_seq), TYPE_NAK)
                                
                                state = "SEARCH_SYNC"
                                continue
                            samples_until_next_read += SYMBOL_SAMPLES
        except Exception as e:
            self.log_callback(f"[ROUTER ERROR] {e}\n")

    def stop(self):
        self.running = False

# ============================================================================
# ROUTER GUI APPLICATION
# ============================================================================
class RouterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Acoustic Central Router Node")
        self.geometry("900x750")
        self.configure(bg="#1e1e2e")

        top_frame = tk.LabelFrame(self, text="Router Configuration", font=("Helvetica", 11, "bold"), bg="#1e1e2e", fg="#cdd6f4", padx=15, pady=10)
        top_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(top_frame, text="Router ID:", bg="#1e1e2e", fg="#a6adc8").grid(row=0, column=0, sticky="w")
        self.router_id_entry = tk.Entry(top_frame, width=8, bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4")
        self.router_id_entry.insert(0, "999")
        self.router_id_entry.grid(row=0, column=1, padx=5)

        self.toggle_btn = tk.Button(
            top_frame, text="🚀 Start Router", font=("Helvetica", 11, "bold"),
            bg="#a6e3a1", fg="#11111b", padx=12, pady=4, cursor="hand2", command=self.toggle_router
        )
        self.toggle_btn.grid(row=0, column=2, padx=20)

        self.status_lbl = tk.Label(top_frame, text="Status: Offline", font=("Helvetica", 11, "bold"), bg="#1e1e2e", fg="#f38ba8")
        self.status_lbl.grid(row=0, column=3, padx=10)

        body_frame = tk.Frame(self, bg="#1e1e2e")
        body_frame.pack(fill="both", expand=True, padx=20, pady=5)

        left_sub = tk.Frame(body_frame, bg="#1e1e2e")
        left_sub.pack(side="top", fill="both", expand=True, pady=(0, 10))

        tk.Label(left_sub, text="Live Router Event Log:", font=("Helvetica", 11, "bold"), bg="#1e1e2e", fg="#cdd6f4").pack(anchor="w")
        self.log_box = tk.Text(left_sub, height=12, font=("Courier", 10), bg="#313244", fg="#cdd6f4", relief="flat")
        self.log_box.pack(fill="both", expand=True, pady=(5, 0))

        right_sub = tk.Frame(body_frame, bg="#1e1e2e")
        right_sub.pack(side="bottom", fill="both", expand=True)

        tk.Label(right_sub, text="Stored Data / Reassembled Payload Preview:", font=("Helvetica", 11, "bold"), bg="#1e1e2e", fg="#cdd6f4").pack(anchor="w")
        
        data_container = tk.Frame(right_sub, bg="#313244", bd=1, relief="solid")
        data_container.pack(fill="both", expand=True, pady=(5, 0))

        self.preview_canvas = tk.Canvas(data_container, width=200, height=200, bg="#000000", highlightthickness=0)
        self.preview_canvas.pack(side="left", padx=15, pady=15)
        
        self.preview_rects = [[None for _ in range(16)] for _ in range(16)]
        for r in range(16):
            for c in range(16):
                self.preview_rects[r][c] = self.preview_canvas.create_rectangle(c*12, r*12, (c+1)*12, (r+1)*12, fill="#000000", outline="#1e1e2e")

        self.data_text_box = tk.Text(data_container, font=("Courier", 11), bg="#313244", fg="#a6e3a1", relief="flat", wrap="word")
        self.data_text_box.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.router_thread = None
        self.is_running = False

    def log_msg(self, msg):
        def _update():
            self.log_box.insert("end", msg)
            self.log_box.see("end")
        self.after(0, _update)

    def display_received_data(self, p_type, data, sender, target):
        def _render():
            self.data_text_box.delete("1.0", "end")
            header_str = f"--- Payload Uploaded (From: {sender} -> Target: {target}) ---\n\n"
            self.data_text_box.insert("end", header_str)

            for r in range(16):
                for c in range(16):
                    self.preview_canvas.itemconfig(self.preview_rects[r][c], fill="#000000")

            if p_type == TYPE_TEXT:
                try:
                    text_content = data.decode('utf-8')
                    self.data_text_box.insert("end", f"Text Message:\n{text_content}")
                except Exception as e:
                    self.data_text_box.insert("end", f"Decoded binary text error: {e}")
            elif p_type == TYPE_IMAGE:
                self.data_text_box.insert("end", "Reconstructed 16x16 Pixel Image:\n")
                flat_nibbles = []
                for b in data:
                    flat_nibbles.append((b >> 4) & 0x0F)
                    flat_nibbles.append(b & 0x0F)
                for idx, color_idx in enumerate(flat_nibbles[:256]):
                    r = idx // 16
                    c = idx % 16
                    if 0 <= color_idx < len(PALETTE):
                        self.preview_canvas.itemconfig(self.preview_rects[r][c], fill=PALETTE[color_idx])
            elif p_type == TYPE_FILE:
                os.makedirs("router_stored", exist_ok=True)
                filename = datetime.now().strftime("forward_%Y%m%d_%H%M%S.bin")
                filepath = os.path.join("router_stored", filename)
                with open(filepath, "wb") as f:
                    f.write(data)
                self.data_text_box.insert("end", f"Stored Binary File ({len(data)} bytes)\nSaved locally to:\n{filepath}")
        self.after(0, _render)

    def toggle_router(self):
        self.is_running = not self.is_running
        if self.is_running:
            try:
                r_addr = int(self.router_id_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Router ID must be an integer.")
                self.is_running = False
                return

            self.toggle_btn.config(text="⏹️ Stop Router", bg="#f38ba8", fg="#11111b")
            self.status_lbl.config(text=f"Status: Listening on ID {r_addr}", fg="#a6e3a1")
            
            self.router_thread = TkinterRouterThread(
                router_addr=r_addr,
                log_callback=self.log_msg,
                data_callback=self.display_received_data
            )
            self.router_thread.start()
        else:
            if self.router_thread:
                self.router_thread.stop()
            self.toggle_btn.config(text="🚀 Start Router", bg="#a6e3a1", fg="#11111b")
            self.status_lbl.config(text="Status: Offline", fg="#f38ba8")
            self.log_msg("Router stopped.\n")

if __name__ == "__main__":
    app = RouterApp()
    app.mainloop()