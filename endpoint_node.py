import struct
import zlib
import threading
import time
import os
from datetime import datetime
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
TYPE_FORWARD_REQ = 0x20
TYPE_FORWARD_RESP = 0x21
TYPE_FORWARD_FAIL = 0x22

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

def listen_for_response(expected_type, timeout=4.0):
    start_time = time.time()
    ring_buffer = np.zeros(SYMBOL_SAMPLES * 7, dtype=np.float32)
    state = "SEARCH_SYNC"
    samples_until_next_read = 0
    nibbles = []
    expected_nibbles = None
    expected_sync = [SYNC_FREQ_A, SYNC_FREQ_B] * 3
    sync_armed = False
    best_sync_snr = 0.0

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=STEP_SAMPLES) as stream:
            while time.time() - start_time < timeout:
                audio, _ = stream.read(STEP_SAMPLES)
                ring_buffer = np.roll(ring_buffer, -STEP_SAMPLES)
                ring_buffer[-STEP_SAMPLES:] = audio[:, 0]
                
                if state == "SEARCH_SYNC":
                    detected_sync = []
                    last_snr = 0.0
                    for k in range(6):
                        freq, snr, _ = analyze_window(ring_buffer[-((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.025)) - WINDOW_SAMPLES:-((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.025))])
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
                        _, _, best_data_idx = analyze_window(ring_buffer[-int(SAMPLE_RATE * 0.025) - WINDOW_SAMPLES:-int(SAMPLE_RATE * 0.025)])
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
                                    return True, full_payload[1:]
                            return False, None
                        samples_until_next_read += SYMBOL_SAMPLES
    except Exception:
        pass
    return False, None

def perform_router_handshake(router_addr, my_addr, target_addr, callback_status):
    callback_status("Initiating Handshake with Router...", "#f9e2af")
    play_audio_waveform(struct.pack('>HH', router_addr, my_addr), TYPE_HS_REQ)
    
    success, resp_data = listen_for_response(TYPE_HS_RESP, timeout=5.0)
    if not success or len(resp_data) < 4:
        callback_status("Handshake FAILED: No confirmation from Router.", "#f38ba8")
        return False
        
    callback_status("Handshake verified! Designating target...", "#f9e2af")
    play_audio_waveform(struct.pack('>HH', router_addr, target_addr), TYPE_TARGET_REQ)
    
    success_tgt, _ = listen_for_response(TYPE_TARGET_RESP, timeout=5.0)
    if not success_tgt:
        callback_status("Target Designation FAILED: Router did not acknowledge target.", "#f38ba8")
        return False
        
    callback_status("Router ready! Uploading data...", "#a6e3a1")
    return True

def transmit_with_arq(data_payload, payload_type, callback_status, max_retries=3):
    chunks = [data_payload[i:i + CHUNK_SIZE] for i in range(0, len(data_payload), CHUNK_SIZE)] if len(data_payload) > 0 else [b""]
    total_packets = len(chunks)
    seq_num = 0
    retries = 0

    while seq_num < total_packets:
        pkt_label = f"Packet {seq_num + 1}/{total_packets}"
        callback_status(f"Uploading {pkt_label}...", "#89b4fa")
        
        try:
            play_audio_waveform(chunks[seq_num], payload_type, seq_num=seq_num, total_packets=total_packets)
        except Exception as e:
            callback_status(f"Transmission error: {e}", "#f38ba8")
            return False
        
        time.sleep(0.3)
        response, resp_seq = listen_for_ack_nak_local(timeout=4.5)
        
        if response == "ACK" and resp_seq == seq_num:
            seq_num += 1
            retries = 0
        elif response == "NAK":
            target = resp_seq if (resp_seq is not None and 0 <= resp_seq < total_packets) else seq_num
            retries += 1
            if retries >= max_retries:
                callback_status(f"FAILED: Max retries exceeded on {pkt_label}", "#f38ba8")
                return False
            seq_num = target
        else:
            retries += 1
            if retries >= max_retries:
                callback_status(f"FAILED: Timeout on {pkt_label}", "#f38ba8")
                return False

    callback_status(f"Upload SUCCESS! Waiting for router target delivery confirmation...", "#a6e3a1")
    
    success_deliv, deliv_data = listen_for_response(TYPE_FORWARD_FAIL, timeout=8.0)
    if success_deliv:
        callback_status("DELIVERY FAILED: Target was unreachable.", "#f38ba8")
        return False
        
    callback_status("Delivery CONFIRMED by Router!", "#a6e3a1")
    return True

def listen_for_ack_nak_local(timeout=4.0):
    start_time = time.time()
    ring_buffer = np.zeros(SYMBOL_SAMPLES * 7, dtype=np.float32)
    state = "SEARCH_SYNC"
    samples_until_next_read = 0
    nibbles = []
    expected_nibbles = None
    expected_sync = [SYNC_FREQ_A, SYNC_FREQ_B] * 3
    sync_armed = False
    best_sync_snr = 0.0

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=STEP_SAMPLES) as stream:
            while time.time() - start_time < timeout:
                audio, _ = stream.read(STEP_SAMPLES)
                ring_buffer = np.roll(ring_buffer, -STEP_SAMPLES)
                ring_buffer[-STEP_SAMPLES:] = audio[:, 0]
                
                if state == "SEARCH_SYNC":
                    detected_sync = []
                    last_snr = 0.0
                    for k in range(6):
                        freq, snr, _ = analyze_window(ring_buffer[-((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.025)) - WINDOW_SAMPLES:-((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.025))])
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
                        _, _, best_data_idx = analyze_window(ring_buffer[-int(SAMPLE_RATE * 0.025) - WINDOW_SAMPLES:-int(SAMPLE_RATE * 0.025)])
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
    except Exception:
        pass
    return None, None

# ============================================================================
# ENDPOINT RECEIVER THREAD
# ============================================================================
class EndpointReceiverThread(threading.Thread):
    def __init__(self, my_addr, callback_packet_received, callback_pixel_received, callback_log):
        super().__init__()
        self.daemon = True
        self.my_addr = my_addr
        self.callback_packet_received = callback_packet_received
        self.callback_pixel_received = callback_pixel_received
        self.callback_log = callback_log
        self.running = False

    def _send_control_response(self, stream, ring_buffer, payload, control_type):
        time.sleep(0.3)
        play_audio_waveform(payload, control_type)
        time.sleep(0.15)
        while stream.read_available > 0:
            stream.read(min(stream.read_available, STEP_SAMPLES))
        ring_buffer.fill(0)

    def run(self):
        self.running = True
        self.callback_log(f"[INFO] Endpoint listener active on ID {self.my_addr}...\n")
        
        ring_buffer = np.zeros(SYMBOL_SAMPLES * 7, dtype=np.float32)
        state = "SEARCH_SYNC"
        samples_until_next_read = 0
        nibbles = []
        expected_nibbles = None
        payload_len = 0
        is_image_packet = False
        current_seq_num = 0
        
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
                            freq, snr, _ = analyze_window(ring_buffer[-((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.010)) - WINDOW_SAMPLES:-((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.010))])
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
                                payload_len = 0
                                is_image_packet = False
                                current_seq_num = expected_seq
                                sync_armed = False
                                best_sync_snr = 0.0
                                samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES
                        elif sync_armed:
                            state = "RECEIVE_DATA"
                            nibbles = []
                            expected_nibbles = None
                            payload_len = 0
                            is_image_packet = False
                            current_seq_num = expected_seq
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

                            elif current_nibble_count == 14 and expected_nibbles is not None:
                                try:
                                    header_bytes = decode_nibbles_to_bytes(nibbles[4:14])
                                    payload_type, seq_num, total_pkts = struct.unpack('>BHH', header_bytes)
                                    current_seq_num = seq_num if seq_num < total_pkts else expected_seq
                                    if payload_type == TYPE_IMAGE or active_transfer_type == TYPE_IMAGE:
                                        is_image_packet = True
                                except Exception:
                                    pass

                            elif (expected_nibbles is not None and 14 < current_nibble_count <= expected_nibbles - 8 and is_image_packet):
                                nibble_in_chunk = current_nibble_count - 15
                                pixel_idx = (current_seq_num * CHUNK_SIZE * 2) + nibble_in_chunk
                                if pixel_idx < 256:
                                    self.callback_pixel_received(pixel_idx // 16, pixel_idx % 16, best_data_idx)

                            if expected_nibbles is not None and current_nibble_count >= expected_nibbles:
                                raw_bytes = decode_nibbles_to_bytes(nibbles)
                                full_payload = raw_bytes[2:-4]
                                received_crc = struct.unpack('>I', raw_bytes[-4:])[0]
                                
                                if received_crc == (zlib.crc32(full_payload) & 0xFFFFFFFF) and len(full_payload) >= 1:
                                    p_type = full_payload[0]
                                    
                                    if p_type == TYPE_FORWARD_REQ and len(full_payload) >= 5:
                                        target_addr, sender_addr = struct.unpack('>HH', full_payload[1:5])
                                        if target_addr == self.my_addr:
                                            self.callback_log(f"[PAGED] Router paged us for incoming transfer from {sender_addr}. Replying READY...\n")
                                            self._send_control_response(stream, ring_buffer, b"", TYPE_FORWARD_RESP)
                                    
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
                                            self._send_control_response(stream, ring_buffer, struct.pack('>H', seq_num), TYPE_ACK)
                                            expected_seq += 1

                                            if expected_seq == total_pkts:
                                                complete_data = b"".join(reassembly_buffer[i] for i in range(total_pkts))
                                                self.callback_packet_received(bytes([active_transfer_type]) + complete_data)
                                                reassembly_buffer.clear()
                                                expected_seq = 0
                                                active_transfer_type = None
                                        elif seq_num < expected_seq:
                                            self._send_control_response(stream, ring_buffer, struct.pack('>H', seq_num), TYPE_ACK)
                                        else:
                                            self._send_control_response(stream, ring_buffer, struct.pack('>H', expected_seq), TYPE_NAK)
                                else:
                                    self._send_control_response(stream, ring_buffer, struct.pack('>H', expected_seq), TYPE_NAK)
                                    
                                state = "SEARCH_SYNC"
                                continue
                                
                            samples_until_next_read += SYMBOL_SAMPLES
        except Exception as e:
            self.callback_log(f"[INFO] Receiver stream stopped: {e}\n")

    def stop(self):
        self.running = False

# ============================================================================
# ENDPOINT GUI APPLICATION
# ============================================================================
class EndpointApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Acoustic End Point Node (Sender & Receiver)")
        self.geometry("950x760")
        self.configure(bg="#1e1e2e")

        net_frame = tk.LabelFrame(self, text="Network Configuration", font=("Helvetica", 11, "bold"), bg="#1e1e2e", fg="#cdd6f4", padx=15, pady=10)
        net_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(net_frame, text="My Addr:", bg="#1e1e2e", fg="#a6adc8").grid(row=0, column=0, sticky="w")
        self.my_addr_entry = tk.Entry(net_frame, width=8, bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4")
        self.my_addr_entry.insert(0, "101")
        self.my_addr_entry.grid(row=0, column=1, padx=5)

        tk.Label(net_frame, text="Router Addr:", bg="#1e1e2e", fg="#a6adc8").grid(row=0, column=2, sticky="w", padx=(15, 0))
        self.router_addr_entry = tk.Entry(net_frame, width=8, bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4")
        self.router_addr_entry.insert(0, "999")
        self.router_addr_entry.grid(row=0, column=3, padx=5)

        tk.Label(net_frame, text="Target Addr:", bg="#1e1e2e", fg="#a6adc8").grid(row=0, column=4, sticky="w", padx=(15, 0))
        self.target_addr_entry = tk.Entry(net_frame, width=8, bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4")
        self.target_addr_entry.insert(0, "102")
        self.target_addr_entry.grid(row=0, column=5, padx=5)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Helvetica", 11, "bold"), padding=[15, 8])

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        self.tab_send_text = tk.Frame(self.notebook, bg="#181825")
        self.tab_send_file = tk.Frame(self.notebook, bg="#181825")
        self.tab_editor = tk.Frame(self.notebook, bg="#181825")
        self.tab_receive = tk.Frame(self.notebook, bg="#181825")

        self.notebook.add(self.tab_send_text, text="   1. Text Message   ")
        self.notebook.add(self.tab_send_file, text="   2. Choose File   ")
        self.notebook.add(self.tab_editor, text="   3. 16x16 Pixel Editor   ")
        self.notebook.add(self.tab_receive, text="   4. Listen / Receive Feed   ")

        self._build_text_tab()
        self._build_file_tab()
        self._build_editor_tab()
        self._build_receive_tab()

        bottom = tk.Frame(self, bg="#181825", height=70)
        bottom.pack(fill="x", side="bottom")

        self.status_lbl = tk.Label(bottom, text="Ready.", font=("Helvetica", 11), bg="#181825", fg="#a6adc8")
        self.status_lbl.pack(side="left", padx=20, pady=15)

        self.tx_btn = tk.Button(
            bottom, text="🤝 ROUTED TRANSMIT (ACTIVE TAB)", font=("Helvetica", 12, "bold"),
            bg="#f38ba8", fg="#11111b", padx=15, pady=8, cursor="hand2", command=self.start_routed_transmission
        )
        self.tx_btn.pack(side="right", padx=20, pady=10)

    def _build_text_tab(self):
        tk.Label(self.tab_send_text, text="Enter UTF-8 Text Payload:", font=("Helvetica", 12, "bold"), bg="#181825", fg="#cdd6f4").pack(anchor="w", padx=20, pady=(20, 5))
        self.text_input = tk.Text(self.tab_send_text, height=10, font=("Courier", 12), bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4")
        self.text_input.pack(fill="x", padx=20, pady=10)
        self.text_input.insert("1.0", "Hello through the central store-and-forward acoustic router!")

    def _build_file_tab(self):
        self.selected_file_path = None
        self.raw_file_bytes = None
        frame = tk.Frame(self.tab_send_file, bg="#181825")
        frame.pack(expand=True)

        tk.Button(frame, text="📁 Browse File...", font=("Helvetica", 12, "bold"), bg="#89b4fa", fg="#11111b", command=self.choose_file).pack(pady=10)
        self.file_info_lbl = tk.Label(frame, text="No file selected.", font=("Helvetica", 11), bg="#181825", fg="#a6adc8")
        self.file_info_lbl.pack(pady=10)

    def choose_file(self):
        path = filedialog.askopenfilename()
        if not path: return
        self.selected_file_path = path
        with open(path, "rb") as f:
            self.raw_file_bytes = f.read()
        self.file_info_lbl.config(text=f"Selected: {path.split('/')[-1]} ({len(self.raw_file_bytes)} bytes)")

    def _build_editor_tab(self):
        self.grid_size = 16
        self.cell_size = 22
        self.current_color_idx = 8
        self.pixel_data = [[0 for _ in range(16)] for _ in range(16)]
        self.rect_ids = [[None for _ in range(16)] for _ in range(16)]

        editor_container = tk.Frame(self.tab_editor, bg="#181825")
        editor_container.pack(expand=True, fill="both", padx=20, pady=10)

        canvas_size = self.grid_size * self.cell_size
        self.canvas = tk.Canvas(editor_container, width=canvas_size, height=canvas_size, bg="#000000", highlightthickness=2, highlightbackground="#585b70")
        self.canvas.pack(side="left", padx=(10, 20), pady=5)

        for r in range(16):
            for c in range(16):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=PALETTE[0], outline="#1e1e2e", width=1)
                self.rect_ids[r][c] = rect

        self.canvas.bind("<Button-1>", self._paint_event)
        self.canvas.bind("<B1-Motion>", self._paint_event)
        for btn in ("<Button-2>", "<B2-Motion>", "<Button-3>", "<B3-Motion>"):
            self.canvas.bind(btn, self._erase_event)

        tools_frame = tk.Frame(editor_container, bg="#181825")
        tools_frame.pack(side="left", fill="y", pady=5)

        tk.Label(tools_frame, text="16-Color Palette (4-bit)", font=("Helvetica", 11, "bold"), bg="#181825", fg="#cdd6f4").pack(anchor="w", pady=(0, 5))
        palette_grid = tk.Frame(tools_frame, bg="#181825")
        palette_grid.pack()

        self.swatch_widgets = []
        for idx, hex_color in enumerate(PALETTE):
            is_selected = (idx == self.current_color_idx)
            swatch = tk.Canvas(palette_grid, width=32, height=28, bg=hex_color, highlightthickness=3, highlightbackground="#ffffff" if is_selected else "#181825", cursor="hand2")
            swatch.grid(row=idx // 4, column=idx % 4, padx=3, pady=3)
            swatch.bind("<Button-1>", lambda e, i=idx: self.select_color(i))
            self.swatch_widgets.append(swatch)

        self.active_color_lbl = tk.Label(tools_frame, text=f"Selected: Color #{self.current_color_idx}", font=("Helvetica", 9), bg="#181825", fg="#a6adc8")
        self.active_color_lbl.pack(pady=5)

        tk.Button(tools_frame, text="Fill Canvas with Selected", font=("Helvetica", 9, "bold"), highlightbackground="#181825", width=22, pady=4, cursor="hand2", command=self.fill_canvas).pack(pady=3)
        tk.Button(tools_frame, text="Clear Canvas (Black)", font=("Helvetica", 9, "bold"), highlightbackground="#181825", width=22, pady=4, cursor="hand2", command=lambda: self.fill_canvas(0)).pack(pady=3)
        tk.Button(tools_frame, text="Load Test Pattern", font=("Helvetica", 9, "bold"), highlightbackground="#181825", width=22, pady=4, cursor="hand2", command=self.load_test_pattern).pack(pady=3)
        
        # --- NEW BUTTON: Load Saved Image into 16x16 Editor ---
        tk.Button(
            tools_frame, text="📁 Load Saved Image (.png)", font=("Helvetica", 9, "bold"),
            bg="#89b4fa", fg="#11111b", highlightbackground="#181825", width=22, pady=4, cursor="hand2",
            command=self.load_editor_image
        ).pack(pady=3)

        tk.Button(
            tools_frame, text="💾 Save Drawn Image (.png)", font=("Helvetica", 9, "bold"),
            bg="#89b4fa", fg="#11111b", highlightbackground="#181825", width=22, pady=5, cursor="hand2",
            command=self.save_drawn_image
        ).pack(pady=(10, 3))

    def _build_receive_tab(self):
        self.is_listening = False
        self.receiver_thread = None
        self.rx_pixel_data = [[0 for _ in range(16)] for _ in range(16)]

        main_body = tk.Frame(self.tab_receive, bg="#181825")
        main_body.pack(fill="both", expand=True, padx=15, pady=15)

        left_col = tk.Frame(main_body, bg="#181825")
        left_col.pack(side="left", fill="y", padx=(0, 15))

        tk.Label(left_col, text="Live Received Image", font=("Helvetica", 11, "bold"), bg="#181825", fg="#cdd6f4").pack(pady=(0, 5))
        self.rx_cell_size = 20
        self.rx_canvas = tk.Canvas(left_col, width=320, height=320, bg="#000000", highlightthickness=2, highlightbackground="#585b70")
        self.rx_canvas.pack()

        self.rx_rects = [[None for _ in range(16)] for _ in range(16)]
        for r in range(16):
            for c in range(16):
                self.rx_rects[r][c] = self.rx_canvas.create_rectangle(c*20, r*20, (c+1)*20, (r+1)*20, fill="#000000", outline="#11111b")

        right_col = tk.Frame(main_body, bg="#181825")
        right_col.pack(side="left", fill="both", expand=True)

        self.listen_btn = tk.Button(
            right_col, text="🎙️ START LISTENING FOR DATA", font=("Helvetica", 12, "bold"),
            bg="#a6e3a1", fg="#11111b", padx=15, pady=8, cursor="hand2", command=self.toggle_listening
        )
        self.listen_btn.pack(fill="x", pady=(0, 10))

        tk.Label(right_col, text="Incoming Packet Log & Decoded Text:", font=("Helvetica", 11, "bold"), bg="#181825", fg="#cdd6f4").pack(anchor="w")
        self.rx_text_box = tk.Text(right_col, height=14, font=("Courier", 10), bg="#313244", fg="#cdd6f4", relief="flat")
        self.rx_text_box.pack(fill="both", expand=True, pady=(5, 0))

    def toggle_listening(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            try:
                my_addr = int(self.my_addr_entry.get())
            except ValueError:
                messagebox.showerror("Error", "My Addr must be an integer.")
                self.is_listening = False
                return

            self.listen_btn.config(text="⏹️ STOP LISTENING", bg="#f38ba8")
            self.receiver_thread = EndpointReceiverThread(
                my_addr=my_addr,
                callback_packet_received=self.on_packet_received,
                callback_pixel_received=self.update_rx_pixel_safely,
                callback_log=self.append_log
            )
            self.receiver_thread.start()
        else:
            self.listen_btn.config(text="🎙️ START LISTENING FOR DATA", bg="#a6e3a1")
            if self.receiver_thread:
                self.receiver_thread.stop()
            self.append_log("[INFO] Stopped listening.\n")

    def append_log(self, text):
        def _update():
            self.rx_text_box.insert("end", text)
            self.rx_text_box.see("end")
        self.after(0, _update)

    def update_rx_pixel_safely(self, row, col, color_idx):
        def _paint():
            if 0 <= row < 16 and 0 <= col < 16 and 0 <= color_idx < 16:
                self.rx_pixel_data[row][col] = color_idx
                self.rx_canvas.itemconfig(self.rx_rects[row][col], fill=PALETTE[color_idx])
        self.after(0, _paint)

    def on_packet_received(self, full_payload):
        def _handle():
            payload_type = full_payload[0]
            actual_data = full_payload[1:]

            if payload_type == TYPE_IMAGE:
                self.append_log(f"[SUCCESS] 16x16 Image successfully received and reassembled!\n\n")
            elif payload_type == TYPE_TEXT:
                try:
                    text_msg = actual_data.decode("utf-8")
                    self.append_log(f"Decoded Message:\n{text_msg}\n[SUCCESS] Message verified.\n\n")
                except UnicodeDecodeError:
                    self.append_log(f"[ERROR] Text payload contained invalid UTF-8 bytes.\n")
            elif payload_type == TYPE_FILE:
                self.append_log(f"Received binary file ({len(actual_data)} bytes). Saving to ./received/...\n")
                os.makedirs("received", exist_ok=True)
                filepath = os.path.join("received", datetime.now().strftime("received_%Y%m%d_%H%M%S.bin"))
                with open(filepath, "wb") as f:
                    f.write(actual_data)
                self.append_log(f"Saved to {filepath}\n\n")
        self.after(0, _handle)

    def select_color(self, idx):
        self.current_color_idx = idx
        for i, swatch in enumerate(self.swatch_widgets):
            swatch.config(highlightbackground="#ffffff" if i == idx else "#181825")
        self.active_color_lbl.config(text=f"Selected: Color #{idx}")

    def set_grid_pixel(self, r, c, color_idx):
        if 0 <= r < 16 and 0 <= c < 16:
            self.pixel_data[r][c] = color_idx
            self.canvas.itemconfig(self.rect_ids[r][c], fill=PALETTE[color_idx])

    def _paint_event(self, event):
        self.set_grid_pixel(event.y // self.cell_size, event.x // self.cell_size, self.current_color_idx)

    def _erase_event(self, event):
        self.set_grid_pixel(event.y // self.cell_size, event.x // self.cell_size, 0)

    def fill_canvas(self, color_idx=None):
        idx = self.current_color_idx if color_idx is None else color_idx
        for r in range(16):
            for c in range(16):
                self.set_grid_pixel(r, c, idx)

    def load_test_pattern(self):
        for r in range(16):
            for c in range(16):
                self.set_grid_pixel(r, c, ((r // 4) * 4 + (c // 4)) % 16)

    def load_editor_image(self):
        """Loads a saved upscaled PNG image back into the 16x16 grid."""
        file_path = filedialog.askopenfilename(title="Open Saved 16x16 Image", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize((16, 16), Image.Resampling.NEAREST)
            for r in range(16):
                for c in range(16):
                    rgb = img.getpixel((c, r))
                    closest_idx = min(
                        range(16),
                        key=lambda i: sum((rgb[k] - hex_to_rgb(PALETTE[i])[k]) ** 2 for k in range(3))
                    )
                    self.set_grid_pixel(r, c, closest_idx)
            messagebox.showinfo("Success", "Image successfully loaded into 16x16 editor grid!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{e}")

    def save_drawn_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not file_path: return
        try:
            img = Image.new("RGB", (16, 16))
            for r in range(16):
                for c in range(16):
                    img.putpixel((c, r), hex_to_rgb(PALETTE[self.pixel_data[r][c]]))
            img.resize((256, 256), Image.Resampling.NEAREST).save(file_path)
            messagebox.showinfo("Success", f"Image saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save image:\n{e}")

    def get_packed_canvas_bytes(self) -> bytes:
        flat = [self.pixel_data[r][c] for r in range(16) for c in range(16)]
        return bytes([(flat[i] << 4) | (flat[i+1] & 0x0F) for i in range(0, len(flat), 2)])

    def update_status(self, text, color):
        def _set(): self.status_lbl.config(text=text, fg=color)
        self.after(0, _set)

    def start_routed_transmission(self):
        try:
            my_addr = int(self.my_addr_entry.get())
            router_addr = int(self.router_addr_entry.get())
            target_addr = int(self.target_addr_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Addresses must be integers.")
            return

        active_tab = self.notebook.index(self.notebook.select())
        if active_tab == 0:
            payload = self.text_input.get("1.0", "end-1c").encode("utf-8")
            payload_type = TYPE_TEXT
        elif active_tab == 1:
            if not self.raw_file_bytes:
                messagebox.showwarning("Warning", "Please select a file first.")
                return
            payload = self.raw_file_bytes
            payload_type = TYPE_FILE
        elif active_tab == 2:
            payload = self.get_packed_canvas_bytes()
            payload_type = TYPE_IMAGE
        else:
            messagebox.showinfo("Info", "Switch to a send tab (Text, File, or Editor) to transmit data.")
            return

        def _task():
            success = perform_router_handshake(router_addr, my_addr, target_addr, self.update_status)
            if success:
                transmit_with_arq(payload, payload_type, self.update_status)

        threading.Thread(target=_task, daemon=True).start()

if __name__ == "__main__":
    app = EndpointApp()
    app.mainloop()