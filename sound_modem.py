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
WINDOW_SAMPLES = int(SAMPLE_RATE * 0.060)  # 60ms analysis window
STEP_SAMPLES = int(SAMPLE_RATE * 0.010)    # 10ms step during sync search

CHUNK_SIZE = 32  # Bytes per packet chunk (32 bytes = 6.4s audio per chunk)

# Payload Type Identifiers
TYPE_TEXT = 0x01
TYPE_IMAGE = 0x02
TYPE_FILE = 0x03
TYPE_ACK = 0x04
TYPE_NAK = 0x05

# 16 Data Frequencies: 8000 Hz to 11750 Hz (250 Hz spacing)
DATA_FREQS = np.array([8000 + i * 250 for i in range(16)])
SYNC_FREQ_A = 12250
SYNC_FREQ_B = 12750
ALL_FREQS = np.append(DATA_FREQS, [SYNC_FREQ_A, SYNC_FREQ_B])

FFT_SIZE = 8192
HANN = np.hanning(WINDOW_SAMPLES)
FFT_FREQS = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
FREQ_BINS = [np.argmin(np.abs(FFT_FREQS - f)) for f in ALL_FREQS]
SNR_THRESHOLD = 6.0

# Fixed 16-color palette (PICO-8 inspired)
PALETTE = [
    "#000000", "#1D2B53", "#7E2553", "#008751",
    "#AB5236", "#5F574F", "#C2C3C7", "#FFF1E8",
    "#FF004D", "#FFA300", "#FFEC27", "#00E436",
    "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA"
]

def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

PALETTE_RGB = [hex_to_rgb(c) for c in PALETTE]


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
    guard_samples = total_samples - tone_samples
    if guard_samples > 0:
        wave = np.pad(wave, (0, guard_samples), mode='constant')
    return wave

def build_packet(data_payload, payload_type, seq_num=0, total_packets=1):
    """
    Packet format:
    [2B Length] + [1B Type | 2B SeqNum | 2B TotalPackets | N-B Data] + [4B CRC32]
    For ACK/NAK, data_payload holds the 2-byte target sequence number.
    """
    if payload_type in (TYPE_ACK, TYPE_NAK):
        full_payload = bytes([payload_type]) + data_payload
    else:
        header = struct.pack('>BHH', payload_type, seq_num, total_packets)
        full_payload = header + data_payload

    if len(full_payload) > 65535:
        raise ValueError("Payload too large for 2-byte length header.")
    
    crc = zlib.crc32(full_payload) & 0xFFFFFFFF
    packet = struct.pack('>H', len(full_payload)) + full_payload + struct.pack('>I', crc)
    return packet

def play_audio_waveform(data_payload, payload_type, seq_num=0, total_packets=1):
    packet = build_packet(data_payload, payload_type, seq_num, total_packets)
    bitstring = "".join(f"{byte:08b}" for byte in packet)
    
    waves = []
    preamble_freqs = [SYNC_FREQ_A, SYNC_FREQ_B] * 3
    for freq in preamble_freqs:
        waves.append(generate_tone(freq, TONE_DURATION, SYMBOL_DURATION))
        
    for i in range(0, len(bitstring), 4):
        nibble_val = int(bitstring[i:i+4], 2)
        freq = DATA_FREQS[nibble_val]
        waves.append(generate_tone(freq, TONE_DURATION, SYMBOL_DURATION))

    # Trailing postamble tone to protect the final CRC nibble from tail cutoff/echo
    waves.append(generate_tone(SYNC_FREQ_A, TONE_DURATION, SYMBOL_DURATION))
        
    waves.append(np.zeros(int(SAMPLE_RATE * 0.150), dtype=np.float32))
    full_signal = np.concatenate(waves).astype(np.float32)
    
    sd.play(full_signal, samplerate=SAMPLE_RATE)
    sd.wait()


# ============================================================================
# DECODER & LISTENER FUNCTIONS
# ============================================================================
def analyze_window(samples):
    samples = samples - np.mean(samples)
    windowed = samples * HANN
    spectrum = np.abs(np.fft.rfft(windowed, n=FFT_SIZE))
    
    band_mask = (FFT_FREQS >= 7500) & (FFT_FREQS <= 13500)
    noise_floor = np.median(spectrum[band_mask]) + 1e-9
    
    strengths = spectrum[FREQ_BINS]
    best_idx = int(np.argmax(strengths))
    peak_val = strengths[best_idx]
    
    snr = peak_val / noise_floor
    best_data_idx = int(np.argmax(strengths[:16]))
    
    if snr < SNR_THRESHOLD:
        return None, snr, best_data_idx
    return ALL_FREQS[best_idx], snr, best_data_idx

def decode_nibbles_to_bytes(nibbles):
    bitstring = "".join(format(n, "04b") for n in nibbles)
    byte_list = [int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)]
    return bytes(byte_list)


def listen_for_ack_nak(timeout=4.0):
    """Returns tuple: (status_str, target_seq_num) e.g. ('ACK', 0) or ('NAK', 2)."""
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
                new_samples = audio[:, 0]
                
                ring_buffer = np.roll(ring_buffer, -STEP_SAMPLES)
                ring_buffer[-STEP_SAMPLES:] = new_samples
                
                if state == "SEARCH_SYNC":
                    detected_sync = []
                    last_snr = 0.0
                    for k in range(6):
                        end_offset = -((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.025))
                        start_offset = end_offset - WINDOW_SAMPLES
                        freq, snr, _ = analyze_window(ring_buffer[start_offset:end_offset])
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
                        end_offset = -int(SAMPLE_RATE * 0.025)
                        start_offset = end_offset - WINDOW_SAMPLES
                        window = ring_buffer[start_offset:end_offset]
                        
                        _, _, best_data_idx = analyze_window(window)
                        nibbles.append(best_data_idx)
                        current_nibble_count = len(nibbles)

                        if current_nibble_count == 4:
                            length_bytes = decode_nibbles_to_bytes(nibbles[:4])
                            payload_len = struct.unpack('>H', length_bytes)[0]
                            expected_nibbles = (2 + payload_len + 4) * 2

                        if expected_nibbles is not None and current_nibble_count >= expected_nibbles:
                            raw_bytes = decode_nibbles_to_bytes(nibbles)
                            full_payload = raw_bytes[2:-4]
                            received_crc = struct.unpack('>I', raw_bytes[-4:])[0]
                            calculated_crc = zlib.crc32(full_payload) & 0xFFFFFFFF
                            
                            if received_crc == calculated_crc and len(full_payload) >= 3:
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


def transmit_with_arq(data_payload, payload_type, callback_status, max_retries=3):
    # Split payload into fixed-size chunks (ensure at least 1 chunk even if empty)
    if len(data_payload) == 0:
        chunks = [b""]
    else:
        chunks = [data_payload[i:i + CHUNK_SIZE] for i in range(0, len(data_payload), CHUNK_SIZE)]
    
    total_packets = len(chunks)
    seq_num = 0
    retries_for_current_pkt = 0

    while seq_num < total_packets:
        pkt_label = f"Packet {seq_num + 1}/{total_packets}"
        attempt_label = f"Attempt {retries_for_current_pkt + 1}/{max_retries}"
        
        callback_status(f"Transmitting {pkt_label} ({attempt_label})...", "#89b4fa")
        print(f"Transmitting {pkt_label} ({attempt_label})...")
        
        try:
            play_audio_waveform(chunks[seq_num], payload_type, seq_num=seq_num, total_packets=total_packets)
        except Exception as e:
            callback_status(f"Transmission error: {e}", "#f38ba8")
            return False
        
        callback_status(f"Waiting for ACK/NAK on {pkt_label}...", "#f9e2af")
        print(f"Listening for response on {pkt_label}...")
        time.sleep(0.3)
        
        response, resp_seq = listen_for_ack_nak(timeout=4.5)
        
        if response == "ACK" and resp_seq == seq_num:
            print(f"ACK received for {pkt_label}.")
            seq_num += 1
            retries_for_current_pkt = 0
        elif response == "NAK":
            # Receiver explicitly specifies which packet sequence number to resend
            target = resp_seq if (resp_seq is not None and 0 <= resp_seq < total_packets) else seq_num
            retries_for_current_pkt += 1
            if retries_for_current_pkt >= max_retries:
                callback_status(f"FAILED: Packet {target + 1}/{total_packets} exceeded max retries.", "#f38ba8")
                print(f"Packet {target + 1} failed after max retries.")
                return False
            seq_num = target
            callback_status(f"NAK received! Resending Packet {seq_num + 1}/{total_packets}...", "#fab387")
            print(f"Receiver requested resend of Packet {seq_num + 1}/{total_packets} (NAK).")
        else:
            retries_for_current_pkt += 1
            if retries_for_current_pkt >= max_retries:
                callback_status(f"FAILED: No response for {pkt_label} after max retries.", "#f38ba8")
                print(f"Transmission failed on {pkt_label} after max retries.")
                return False
            callback_status(f"Timeout on {pkt_label}. Retrying...", "#fab387")
            print(f"No response heard for {pkt_label}. Retrying...")

    callback_status(f"Transmission SUCCESS! All {total_packets} packet(s) acknowledged.", "#a6e3a1")
    print("All packets transmitted and acknowledged successfully.")
    return True


class ReceiverThread(threading.Thread):
    def __init__(self, callback_packet_received, callback_pixel_received, callback_log):
        super().__init__()
        self.daemon = True
        self.callback_packet_received = callback_packet_received
        self.callback_pixel_received = callback_pixel_received
        self.callback_log = callback_log
        self.running = False

    def _send_control_response(self, stream, ring_buffer, seq_num, control_type):
        """Plays ACK/NAK, waits for room echo to settle, and flushes buffered audio."""
        time.sleep(0.3)
        play_audio_waveform(struct.pack('>H', seq_num), control_type)
        time.sleep(0.15)
        while stream.read_available > 0:
            stream.read(min(stream.read_available, STEP_SAMPLES))
        ring_buffer.fill(0)

    def run(self):
        self.running = True
        self.callback_log("[INFO] Receiver listening for acoustic transmission...\n")
        
        ring_buffer = np.zeros(SYMBOL_SAMPLES * 7, dtype=np.float32)
        state = "SEARCH_SYNC"
        samples_until_next_read = 0
        nibbles = []
        expected_nibbles = None
        payload_len = 0
        is_image_packet = False
        current_seq_num = 0
        
        # Multi-packet reassembly state
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
                    new_samples = audio[:, 0]
                    
                    ring_buffer = np.roll(ring_buffer, -STEP_SAMPLES)
                    ring_buffer[-STEP_SAMPLES:] = new_samples
                    
                    if state == "SEARCH_SYNC":
                        detected_sync = []
                        last_snr = 0.0
                        for k in range(6):
                            end_offset = -((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.010))
                            start_offset = end_offset - WINDOW_SAMPLES
                            freq, snr, _ = analyze_window(ring_buffer[start_offset:end_offset])
                            detected_sync.append(freq)
                            if k == 5:
                                last_snr = snr

                        if detected_sync == expected_sync:
                            if last_snr >= best_sync_snr:
                                best_sync_snr = last_snr
                                sync_armed = True
                            elif sync_armed:
                                self.callback_log("\n[SYNC LOCKED] Receiving packet...\n")
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
                            self.callback_log("\n[SYNC LOCKED] Receiving packet...\n")
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
                            end_offset = -int(SAMPLE_RATE * 0.010)
                            start_offset = end_offset - WINDOW_SAMPLES
                            window = ring_buffer[start_offset:end_offset]
                            
                            _, _, best_data_idx = analyze_window(window)
                            nibbles.append(best_data_idx)
                            current_nibble_count = len(nibbles)

                            # Nibbles 0..3: 2-byte length
                            if current_nibble_count == 4:
                                length_bytes = decode_nibbles_to_bytes(nibbles[:4])
                                payload_len = struct.unpack('>H', length_bytes)[0]
                                expected_nibbles = (2 + payload_len + 4) * 2
                                # Guard against corrupted length headers causing endless waits
                                if payload_len < 5 or expected_nibbles > (2 + 5 + CHUNK_SIZE + 4) * 2:
                                    self.callback_log(f"[ERROR] Corrupted length header! Requesting Packet {expected_seq + 1} (NAK)...\n")
                                    self._send_control_response(stream, ring_buffer, expected_seq, TYPE_NAK)
                                    state = "SEARCH_SYNC"
                                    continue

                            # Nibbles 4..13: 5-byte header [1B Type | 2B SeqNum | 2B TotalPackets]
                            elif current_nibble_count == 14:
                                header_bytes = decode_nibbles_to_bytes(nibbles[4:14])
                                payload_type, seq_num, total_pkts = struct.unpack('>BHH', header_bytes)
                                current_seq_num = seq_num if seq_num < total_pkts else expected_seq
                                
                                if payload_type == TYPE_IMAGE or active_transfer_type == TYPE_IMAGE:
                                    is_image_packet = True
                                self.callback_log(f"  [Header] Type=0x{payload_type:02X}, Pkt {seq_num + 1}/{total_pkts} (unverified)...\n")

                            # Nibbles 14 to (expected_nibbles - 9): Actual data payload (excludes 8-nibble CRC trailer)
                            elif (
                                expected_nibbles is not None
                                and 14 < current_nibble_count <= expected_nibbles - 8
                                and is_image_packet
                            ):
                                nibble_in_chunk = current_nibble_count - 15
                                pixel_idx = (current_seq_num * CHUNK_SIZE * 2) + nibble_in_chunk
                                if pixel_idx < 256:
                                    r = pixel_idx // 16
                                    c = pixel_idx % 16
                                    self.callback_pixel_received(r, c, best_data_idx)

                            if expected_nibbles is not None and current_nibble_count >= expected_nibbles:
                                raw_bytes = decode_nibbles_to_bytes(nibbles)
                                full_payload = raw_bytes[2:-4]
                                received_crc = struct.unpack('>I', raw_bytes[-4:])[0]
                                calculated_crc = zlib.crc32(full_payload) & 0xFFFFFFFF
                                print(f"  Received {len(full_payload)} bytes, CRC: {received_crc:08X}, Calculated CRC: {calculated_crc:08X}")
                                
                                if received_crc == calculated_crc and len(full_payload) >= 5:
                                    p_type, seq_num, total_pkts = struct.unpack('>BHH', full_payload[:5])
                                    chunk_data = full_payload[5:]
                                    
                                    # Reset state if a brand-new transfer starts at packet 0
                                    if seq_num == 0 and expected_seq != 0:
                                        reassembly_buffer.clear()
                                        expected_seq = 0
                                        active_transfer_type = p_type

                                    if active_transfer_type is None:
                                        active_transfer_type = p_type

                                    if seq_num == expected_seq:
                                        reassembly_buffer[seq_num] = chunk_data
                                        self.callback_log(f"[SUCCESS] Packet {seq_num + 1}/{total_pkts} CRC Valid! Sending ACK...\n")
                                        self._send_control_response(stream, ring_buffer, seq_num, TYPE_ACK)
                                        expected_seq += 1

                                        # Check if all packets for this message/file have arrived
                                        if expected_seq == total_pkts:
                                            complete_data = b"".join(reassembly_buffer[i] for i in range(total_pkts))
                                            self.callback_packet_received(bytes([active_transfer_type]) + complete_data)
                                            reassembly_buffer.clear()
                                            expected_seq = 0
                                            active_transfer_type = None
                                    elif seq_num < expected_seq:
                                        # Duplicate packet (our previous ACK was likely missed by sender)
                                        self.callback_log(f"[INFO] Duplicate Packet {seq_num + 1}/{total_pkts}. Re-sending ACK...\n")
                                        self._send_control_response(stream, ring_buffer, seq_num, TYPE_ACK)
                                    else:
                                        # Out-of-order packet received; request the missing expected_seq
                                        self.callback_log(f"[WARN] Out-of-order Packet {seq_num + 1}. Requesting Packet {expected_seq + 1} (NAK)...\n")
                                        self._send_control_response(stream, ring_buffer, expected_seq, TYPE_NAK)
                                else:
                                    # CRC Mismatch: Request exact missing sequence number via NAK
                                    self.callback_log(f"[ERROR] CRC Mismatch! Requesting Packet {expected_seq + 1} resend (NAK)...\n")
                                    self._send_control_response(stream, ring_buffer, expected_seq, TYPE_NAK)
                                    
                                state = "SEARCH_SYNC"
                                continue
                                
                            samples_until_next_read += SYMBOL_SAMPLES
        except Exception as e:
            self.callback_log(f"[INFO] Receiver stream stopped: {e}\n")

    def stop(self):
        self.running = False


# ============================================================================
# GUI APPLICATION
# ============================================================================
class SoundModemApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Acoustic Data & Image Transceiver with Selective Packet ARQ")
        self.geometry("900x680")
        self.configure(bg="#1e1e2e")
        self.resizable(False, False)

        self.container = tk.Frame(self, bg="#1e1e2e")
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (HomePage, SendPage, ReceivePage):
            frame = F(parent=self.container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("HomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e2e")
        self.controller = controller

        title = tk.Label(self, text="SONIC MODEM", font=("Helvetica", 32, "bold"), fg="#cdd6f4", bg="#1e1e2e")
        title.pack(pady=(120, 10))

        subtitle = tk.Label(self, text="Acoustic Transceiver with Packet-Level ARQ", font=("Helvetica", 14), fg="#a6adc8", bg="#1e1e2e")
        subtitle.pack(pady=(0, 60))

        btn_frame = tk.Frame(self, bg="#1e1e2e")
        btn_frame.pack()

        send_btn = tk.Button(
            btn_frame, text="TRANSMIT (SEND)", font=("Helvetica", 16, "bold"),
            width=18, height=3, bg="#89b4fa", fg="#11111b",
            highlightbackground="#1e1e2e", cursor="hand2",
            command=lambda: controller.show_frame("SendPage")
        )
        send_btn.grid(row=0, column=0, padx=20)

        recv_btn = tk.Button(
            btn_frame, text="LISTEN (RECEIVE)", font=("Helvetica", 16, "bold"),
            width=18, height=3, bg="#a6e3a1", fg="#11111b",
            highlightbackground="#1e1e2e", cursor="hand2",
            command=lambda: controller.show_frame("ReceivePage")
        )
        recv_btn.grid(row=0, column=1, padx=20)


class SendPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e2e")
        self.controller = controller

        nav = tk.Frame(self, bg="#181825", height=50)
        nav.pack(fill="x")
        tk.Button(
            nav, text="← Home", font=("Helvetica", 12, "bold"),
            bg="#313244", fg="#11111b", highlightbackground="#181825",
            command=lambda: controller.show_frame("HomePage")
        ).pack(side="left", padx=15, pady=10)
        tk.Label(nav, text="TRANSMIT DATA WITH PACKET ARQ", font=("Helvetica", 14, "bold"), bg="#181825", fg="#cdd6f4").pack(side="left", padx=15)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Helvetica", 11, "bold"), padding=[15, 8])

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        self.tab_text = tk.Frame(self.notebook, bg="#181825")
        self.tab_file = tk.Frame(self.notebook, bg="#181825")
        self.tab_editor = tk.Frame(self.notebook, bg="#181825")

        self.notebook.add(self.tab_text, text="   1. Text Message   ")
        self.notebook.add(self.tab_file, text="   2. Choose File   ")
        self.notebook.add(self.tab_editor, text="   3. 16x16 Pixel Editor   ")

        self._build_text_tab()
        self._build_file_tab()
        self._build_editor_tab()

        bottom = tk.Frame(self, bg="#181825", height=70)
        bottom.pack(fill="x", side="bottom")

        self.status_lbl = tk.Label(bottom, text="Ready to transmit.", font=("Helvetica", 11), bg="#181825", fg="#a6adc8")
        self.status_lbl.pack(side="left", padx=20, pady=15)

        self.tx_btn = tk.Button(
            bottom, text="🔊 TRANSMIT ACTIVE TAB", font=("Helvetica", 13, "bold"),
            bg="#f38ba8", fg="#11111b", highlightbackground="#181825",
            padx=20, pady=8, cursor="hand2", command=self.start_transmission
        )
        self.tx_btn.pack(side="right", padx=20, pady=10)

    def _build_text_tab(self):
        tk.Label(self.tab_text, text="Enter UTF-8 Text Payload:", font=("Helvetica", 12, "bold"), bg="#181825", fg="#cdd6f4").pack(anchor="w", padx=20, pady=(20, 5))
        self.text_input = tk.Text(self.tab_text, height=14, font=("Courier", 13), bg="#313244", fg="#cdd6f4", insertbackground="#cdd6f4", relief="flat")
        self.text_input.pack(fill="x", padx=20, pady=10)
        self.text_input.insert("1.0", "Hello World! Testing chunked acoustic FSK modem with packet-level ARQ.")

    def _build_file_tab(self):
        self.selected_file_path = None
        self.raw_file_bytes = None

        frame = tk.Frame(self.tab_file, bg="#181825")
        frame.pack(expand=True)

        tk.Button(
            frame, text="📁 Browse Device for File...", font=("Helvetica", 14, "bold"),
            bg="#89b4fa", fg="#11111b", highlightbackground="#181825",
            padx=20, pady=10, cursor="hand2", command=self.choose_file
        ).pack(pady=10)

        self.file_info_lbl = tk.Label(frame, text="No file selected.", font=("Helvetica", 12), bg="#181825", fg="#a6adc8")
        self.file_info_lbl.pack(pady=10)

        self.convert_img_btn = tk.Button(
            frame, text="🎨 Load Image into 16x16 Pixel Editor",
            font=("Helvetica", 11, "bold"), bg="#cba6f7", fg="#11111b",
            highlightbackground="#181825", padx=15, pady=6, state="disabled",
            command=self.load_file_into_editor
        )
        self.convert_img_btn.pack(pady=10)

    def choose_file(self):
        path = filedialog.askopenfilename(title="Select a file to transmit")
        if not path:
            return
        self.selected_file_path = path
        with open(path, "rb") as f:
            self.raw_file_bytes = f.read()

        size_kb = len(self.raw_file_bytes) / 1024
        self.file_info_lbl.config(text=f"Selected: {path.split('/')[-1]} ({len(self.raw_file_bytes)} bytes / {size_kb:.2f} KB)")
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            self.convert_img_btn.config(state="normal")
        else:
            self.convert_img_btn.config(state="disabled")

    def load_file_into_editor(self):
        try:
            img = Image.open(self.selected_file_path).convert("RGB")
            img = img.resize((16, 16), Image.Resampling.NEAREST)
            for r in range(16):
                for c in range(16):
                    rgb = img.getpixel((c, r))
                    closest_idx = min(
                        range(16),
                        key=lambda i: sum((rgb[k] - PALETTE_RGB[i][k]) ** 2 for k in range(3))
                    )
                    self.set_grid_pixel(r, c, closest_idx)
            self.notebook.select(self.tab_editor)
            self.status_lbl.config(text="Converted image to 16x16 16-color canvas!")
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not process image:\n{e}")

    def _build_editor_tab(self):
        self.grid_size = 16
        self.cell_size = 24
        self.current_color_idx = 8
        self.pixel_data = [[0 for _ in range(16)] for _ in range(16)]
        self.rect_ids = [[None for _ in range(16)] for _ in range(16)]

        editor_container = tk.Frame(self.tab_editor, bg="#181825")
        editor_container.pack(expand=True, fill="both", padx=20, pady=10)

        canvas_size = self.grid_size * self.cell_size
        self.canvas = tk.Canvas(editor_container, width=canvas_size, height=canvas_size, bg="#000000", highlightthickness=2, highlightbackground="#585b70")
        self.canvas.pack(side="left", padx=(20, 30), pady=5)

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
        tools_frame.pack(side="left", fill="y", pady=10)

        tk.Label(tools_frame, text="16-Color Palette (4-bit)", font=("Helvetica", 12, "bold"), bg="#181825", fg="#cdd6f4").pack(anchor="w", pady=(0, 10))
        palette_grid = tk.Frame(tools_frame, bg="#181825")
        palette_grid.pack()

        self.swatch_widgets = []
        for idx, hex_color in enumerate(PALETTE):
            is_selected = (idx == self.current_color_idx)
            swatch = tk.Canvas(palette_grid, width=38, height=32, bg=hex_color, highlightthickness=3, highlightbackground="#ffffff" if is_selected else "#181825", cursor="hand2")
            swatch.grid(row=idx // 4, column=idx % 4, padx=4, pady=4)
            swatch.bind("<Button-1>", lambda e, i=idx: self.select_color(i))
            self.swatch_widgets.append(swatch)

        self.active_color_lbl = tk.Label(tools_frame, text=f"Selected: Color #{self.current_color_idx} ({PALETTE[self.current_color_idx]})", font=("Helvetica", 10), bg="#181825", fg="#a6adc8")
        self.active_color_lbl.pack(pady=10)

        tk.Button(tools_frame, text="Fill Canvas with Selected", font=("Helvetica", 10, "bold"), highlightbackground="#181825", width=22, pady=5, cursor="hand2", command=self.fill_canvas).pack(pady=4)
        tk.Button(tools_frame, text="Clear Canvas (Black)", font=("Helvetica", 10, "bold"), highlightbackground="#181825", width=22, pady=5, cursor="hand2", command=lambda: self.fill_canvas(0)).pack(pady=4)
        tk.Button(tools_frame, text="Load Test Pattern", font=("Helvetica", 10, "bold"), highlightbackground="#181825", width=22, pady=5, cursor="hand2", command=self.load_test_pattern).pack(pady=4)
        tk.Button(
            tools_frame, text="💾 Save Drawn Image (.png)", font=("Helvetica", 10, "bold"),
            bg="#89b4fa", fg="#11111b", highlightbackground="#181825", width=22, pady=6, cursor="hand2",
            command=self.save_drawn_image
        ).pack(pady=(15, 4))

    def select_color(self, idx):
        self.current_color_idx = idx
        for i, swatch in enumerate(self.swatch_widgets):
            swatch.config(highlightbackground="#ffffff" if i == idx else "#181825")
        self.active_color_lbl.config(text=f"Selected: Color #{idx} ({PALETTE[idx]})")

    def set_grid_pixel(self, r, c, color_idx):
        if 0 <= r < 16 and 0 <= c < 16:
            self.pixel_data[r][c] = color_idx
            self.canvas.itemconfig(self.rect_ids[r][c], fill=PALETTE[color_idx])

    def _paint_event(self, event):
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        self.set_grid_pixel(r, c, self.current_color_idx)

    def _erase_event(self, event):
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        self.set_grid_pixel(r, c, 0)

    def fill_canvas(self, color_idx=None):
        idx = self.current_color_idx if color_idx is None else color_idx
        for r in range(16):
            for c in range(16):
                self.set_grid_pixel(r, c, idx)

    def load_test_pattern(self):
        for r in range(16):
            for c in range(16):
                self.set_grid_pixel(r, c, ((r // 4) * 4 + (c // 4)) % 16)

    def save_drawn_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            img = Image.new("RGB", (16, 16))
            for r in range(16):
                for c in range(16):
                    color_idx = self.pixel_data[r][c]
                    rgb = hex_to_rgb(PALETTE[color_idx])
                    img.putpixel((c, r), rgb)
            img_scaled = img.resize((256, 256), Image.Resampling.NEAREST)
            img_scaled.save(file_path)
            messagebox.showinfo("Success", f"Image successfully saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save image:\n{e}")

    def get_packed_canvas_bytes(self) -> bytes:
        flat_nibbles = [self.pixel_data[r][c] for r in range(16) for c in range(16)]
        packed = bytearray()
        for i in range(0, len(flat_nibbles), 2):
            high = flat_nibbles[i] & 0x0F
            low = flat_nibbles[i + 1] & 0x0F
            packed.append((high << 4) | low)
        return bytes(packed)

    def update_status(self, text, color):
        def _set():
            self.status_lbl.config(text=text, fg=color)
        self.after(0, _set)

    def start_transmission(self):
        active_tab = self.notebook.index(self.notebook.select())
        if active_tab == 0:
            text = self.text_input.get("1.0", "end-1c").strip()
            payload = text.encode("utf-8")
            payload_type = TYPE_TEXT
        elif active_tab == 1:
            if not self.raw_file_bytes:
                messagebox.showwarning("No File", "Please choose a file first.")
                return
            payload = self.raw_file_bytes
            payload_type = TYPE_FILE
        elif active_tab == 2:
            payload = self.get_packed_canvas_bytes()
            payload_type = TYPE_IMAGE
        else:
            return

        threading.Thread(
            target=lambda: transmit_with_arq(payload, payload_type, self.update_status, max_retries=3),
            daemon=True
        ).start()


class ReceivePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#1e1e2e")
        self.controller = controller
        self.is_listening = False
        self.receiver_thread = None
        self.rx_pixel_data = [[0 for _ in range(16)] for _ in range(16)]

        nav = tk.Frame(self, bg="#181825", height=50)
        nav.pack(fill="x")
        tk.Button(
            nav, text="← Home", font=("Helvetica", 12, "bold"),
            bg="#313244", fg="#11111b", highlightbackground="#181825",
            command=lambda: controller.show_frame("HomePage")
        ).pack(side="left", padx=15, pady=10)
        tk.Label(nav, text="RECEIVER & LIVE 16x16 DECODER", font=("Helvetica", 14, "bold"), bg="#181825", fg="#cdd6f4").pack(side="left", padx=15)

        main_body = tk.Frame(self, bg="#1e1e2e")
        main_body.pack(fill="both", expand=True, padx=25, pady=20)

        left_col = tk.Frame(main_body, bg="#181825", padx=15, pady=15)
        left_col.pack(side="left", fill="y", padx=(0, 15))

        tk.Label(left_col, text="Live 16x16 Image Feed", font=("Helvetica", 12, "bold"), bg="#181825", fg="#cdd6f4").pack(pady=(0, 10))

        self.rx_cell_size = 24
        canvas_dim = 16 * self.rx_cell_size
        self.rx_canvas = tk.Canvas(left_col, width=canvas_dim, height=canvas_dim, bg="#000000", highlightthickness=2, highlightbackground="#585b70")
        self.rx_canvas.pack()

        self.rx_rects = [[None for _ in range(16)] for _ in range(16)]
        for r in range(16):
            for c in range(16):
                x1, y1 = c * self.rx_cell_size, r * self.rx_cell_size
                x2, y2 = x1 + self.rx_cell_size, y1 + self.rx_cell_size
                self.rx_rects[r][c] = self.rx_canvas.create_rectangle(x1, y1, x2, y2, fill="#000000", outline="#11111b")

        tk.Button(
            left_col, text="💾 Save Received Image (.png)", font=("Helvetica", 10, "bold"),
            bg="#89b4fa", fg="#11111b", highlightbackground="#181825", width=24, pady=6, cursor="hand2",
            command=self.save_received_image
        ).pack(pady=(15, 0))

        right_col = tk.Frame(main_body, bg="#181825", padx=15, pady=15)
        right_col.pack(side="left", fill="both", expand=True)

        self.listen_btn = tk.Button(
            right_col, text="🎙️ START LISTENING", font=("Helvetica", 14, "bold"),
            bg="#a6e3a1", fg="#11111b", highlightbackground="#181825", pady=10,
            cursor="hand2", command=self.toggle_listening
        )
        self.listen_btn.pack(fill="x", pady=(0, 15))

        tk.Label(right_col, text="Decoded Text / Packet Log:", font=("Helvetica", 11, "bold"), bg="#181825", fg="#cdd6f4").pack(anchor="w")

        self.rx_text_box = tk.Text(right_col, height=16, font=("Courier", 11), bg="#313244", fg="#cdd6f4", relief="flat")
        self.rx_text_box.pack(fill="both", expand=True, pady=(5, 0))

    def toggle_listening(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.listen_btn.config(text="⏹️ STOP LISTENING", bg="#f38ba8")
            self.receiver_thread = ReceiverThread(
                callback_packet_received=self.on_packet_received,
                callback_pixel_received=self.update_rx_pixel_safely,
                callback_log=self.append_log
            )
            self.receiver_thread.start()
        else:
            self.listen_btn.config(text="🎙️ START LISTENING", bg="#a6e3a1")
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

    def save_received_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            img = Image.new("RGB", (16, 16))
            for r in range(16):
                for c in range(16):
                    color_idx = self.rx_pixel_data[r][c]
                    rgb = hex_to_rgb(PALETTE[color_idx])
                    img.putpixel((c, r), rgb)
            img_scaled = img.resize((256, 256), Image.Resampling.NEAREST)
            img_scaled.save(file_path)
            messagebox.showinfo("Success", f"Received image successfully saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save received image:\n{e}")

    def on_packet_received(self, full_payload):
        def _handle():
            payload_type = full_payload[0]
            actual_data = full_payload[1:]

            if payload_type == TYPE_IMAGE:
                self.append_log(f"[SUCCESS] All 16x16 Image packets reassembled & verified!\n\n")
            elif payload_type == TYPE_TEXT:
                try:
                    text_msg = actual_data.decode("utf-8")
                    self.append_log(f"Decoded Message:\n{text_msg}\n[SUCCESS] Full message verified.\n\n")
                except UnicodeDecodeError:
                    self.append_log(f"[ERROR] Text payload contained invalid UTF-8 bytes.\n")
            elif payload_type == TYPE_FILE:
                self.append_log(f"Received complete binary file ({len(actual_data)} bytes). Saving to ./received/...\n")
                os.makedirs("received", exist_ok=True)
                filename = datetime.now().strftime("received_%Y%m%d_%H%M%S.bin")
                filepath = os.path.join("received", filename)
                with open(filepath, "wb") as f:
                    f.write(actual_data)
                self.append_log(f"Saved to {filepath}\n\n")
        self.after(0, _handle)


if __name__ == "__main__":
    app = SoundModemApp()
    app.mainloop()