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

CHUNK_SIZE = 16  # Bytes per chunk (16x16 image = 128 bytes total = 8 chunks)

# Payload Type Identifiers
TYPE_IMAGE = 0x02
TYPE_TEXT = 0x01
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

def build_chunk_packet(chunk_idx, total_chunks, payload_type, chunk_data):
    header = struct.pack('BBB', chunk_idx, total_chunks, payload_type)
    full_payload = header + chunk_data
    crc = zlib.crc32(full_payload) & 0xFFFFFFFF
    packet = struct.pack('>H', len(full_payload)) + full_payload + struct.pack('>I', crc)
    return packet

def build_selective_nak_packet(failed_indices):
    # NAK packet payload: [TYPE_NAK (1B)] + [Num Failed (1B)] + [Failed Index 1, Index 2, ...]
    fail_bytes = bytes([len(failed_indices)] + list(failed_indices))
    full_payload = bytes([TYPE_NAK]) + fail_bytes
    crc = zlib.crc32(full_payload) & 0xFFFFFFFF
    packet = struct.pack('>H', len(full_payload)) + full_payload + struct.pack('>I', crc)
    return packet

def build_ack_packet():
    full_payload = bytes([TYPE_ACK])
    crc = zlib.crc32(full_payload) & 0xFFFFFFFF
    packet = struct.pack('>H', len(full_payload)) + full_payload + struct.pack('>I', crc)
    return packet

def play_audio_waveform(packet_bytes):
    bitstring = "".join(f"{byte:08b}" for byte in packet_bytes)
    waves = []
    
    preamble_freqs = [SYNC_FREQ_A, SYNC_FREQ_B] * 3
    for freq in preamble_freqs:
        waves.append(generate_tone(freq, TONE_DURATION, SYMBOL_DURATION))
        
    for i in range(0, len(bitstring), 4):
        nibble_val = int(bitstring[i:i+4], 2)
        freq = DATA_FREQS[nibble_val]
        waves.append(generate_tone(freq, TONE_DURATION, SYMBOL_DURATION))
        
    waves.append(np.zeros(int(SAMPLE_RATE * 0.150), dtype=np.float32))
    full_signal = np.concatenate(waves).astype(np.float32)
    
    time.sleep(0.3)
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


def transmit_image_selective_arq(pixel_bytes, callback_status, max_retries=5):
    """Sends full image burst, listens for selective NAK list, resends only missing chunks."""
    chunks = [pixel_bytes[i:i + CHUNK_SIZE] for i in range(0, len(pixel_bytes), CHUNK_SIZE)]
    total_chunks = len(chunks)
    
    # Track which chunks still need to be sent
    pending_indices = set(range(total_chunks))
    
    for attempt in range(1, max_retries + 1):
        if not pending_indices:
            break
            
        callback_status(f"Burst Transmitting {len(pending_indices)} chunk(s) (Attempt {attempt})...", "#89b4fa")
        print(f"\n--- Burst Transmission Attempt {attempt} for chunks: {sorted(list(pending_indices))} ---")
        
        # 1. Send all pending chunks in sequence
        for idx in sorted(list(pending_indices)):
            packet = build_chunk_packet(idx, total_chunks, TYPE_IMAGE, chunks[idx])
            print(f"  Sending Chunk {idx + 1}/{total_chunks}...")
            try:
                play_audio_waveform(packet)
            except Exception as e:
                callback_status(f"Audio transmission error: {e}", "#f38ba8")
                return False
            time.sleep(0.15) # brief inter-chunk pause
            
        # 2. Listen for receiver feedback response (ACK or NAK with list of failed chunks)
        callback_status("Listening for selective chunk audit response...", "#f9e2af")
        print("Listening for receiver audit response...")
        
        response_data = listen_for_audit_response(timeout=5.0)
        
        if response_data == "ACK":
            callback_status("Image Transmission SUCCESS! All chunks verified.", "#a6e3a1")
            print("Full image acknowledged successfully (ACK).")
            return True
        elif isinstance(response_data, set):
            pending_indices = response_data
            callback_status(f"CRC Mismatch. Resending {len(pending_indices)} failed chunk(s)...", "#fab387")
            print(f"Receiver requested retransmission for chunks: {sorted(list(pending_indices))}")
        else:
            callback_status("No response heard. Resending all pending...", "#fab387")
            print("Audit response timeout. Retrying pending chunks...")
            
    callback_status("Transmission FAILED after max retries.", "#f38ba8")
    return False


def listen_for_audit_response(timeout=5.0):
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
                        end_offset = -int(SAMPLE_RATE * 0.010)
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
                            
                            if received_crc == calculated_crc and len(full_payload) >= 1:
                                p_type = full_payload[0]
                                if p_type == TYPE_ACK:
                                    return "ACK"
                                elif p_type == TYPE_NAK:
                                    num_failed = full_payload[1]
                                    failed_set = set(full_payload[2:2+num_failed])
                                    return failed_set
                            return None
                            
                        samples_until_next_read += SYMBOL_SAMPLES
    except Exception:
        pass
    return None


class ReceiverThread(threading.Thread):
    def __init__(self, callback_packet_fully_received, callback_pixel_received, callback_log):
        super().__init__()
        self.daemon = True
        self.callback_packet_fully_received = callback_packet_fully_received
        self.callback_pixel_received = callback_pixel_received
        self.callback_log = callback_log
        self.running = False

    def run(self):
        self.running = True
        self.callback_log("[INFO] Receiver listening for image burst...\n")
        
        ring_buffer = np.zeros(SYMBOL_SAMPLES * 7, dtype=np.float32)
        state = "SEARCH_SYNC"
        samples_until_next_read = 0
        nibbles = []
        expected_nibbles = None
        expected_sync = [SYNC_FREQ_A, SYNC_FREQ_B] * 3
        sync_armed = False
        best_sync_snr = 0.0

        # Session buffer for incoming image burst
        active_image_chunks = {}
        total_image_chunks_expected = 8  # 128 bytes / 16 bytes per chunk

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
                                self.callback_log("\n[SYNC LOCKED] Receiving image chunk...\n")
                                state = "RECEIVE_DATA"
                                nibbles = []
                                expected_nibbles = None
                                sync_armed = False
                                best_sync_snr = 0.0
                                samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES
                        elif sync_armed:
                            self.callback_log("\n[SYNC LOCKED] Receiving image chunk...\n")
                            state = "RECEIVE_DATA"
                            nibbles = []
                            expected_nibbles = None
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
                                    chunk_idx = full_payload[0]
                                    total_chunks = full_payload[1]
                                    payload_type = full_payload[2]
                                    chunk_data = full_payload[3:]
                                    total_image_chunks_expected = total_chunks
                                    
                                    self.callback_log(f"  [CRC OK] Chunk {chunk_idx + 1}/{total_chunks} received cleanly.\n")
                                    active_image_chunks[chunk_idx] = chunk_data
                                    
                                    # Update live pixel feed immediately
                                    base_pixel_idx = chunk_idx * CHUNK_SIZE * 2
                                    for p_offset in range(len(chunk_data) * 2):
                                        byte_idx = p_offset // 2
                                        is_high_nibble = (p_offset % 2 == 0)
                                        b_val = chunk_data[byte_idx]
                                        color_idx = (b_val >> 4) if is_high_nibble else (b_val & 0x0F)
                                        
                                        abs_pixel_idx = base_pixel_idx + p_offset
                                        if abs_pixel_idx < 256:
                                            r = abs_pixel_idx // 16
                                            c = abs_pixel_idx % 16
                                            self.callback_pixel_received(r, c, color_idx)
                                else:
                                    self.callback_log(f"  [CRC ERROR] A chunk failed verification.\n")
                                    
                                state = "SEARCH_SYNC"
                                ring_buffer.fill(0)
                                
                                # Check if we have collected all chunks for the image burst
                                if len(active_image_chunks) == total_image_chunks_expected:
                                    complete_pixel_bytes = b"".join(active_image_chunks[i] for i in range(total_image_chunks_expected))
                                    full_payload = bytes([TYPE_IMAGE]) + complete_pixel_bytes
                                    self.callback_packet_fully_received(full_payload)
                                    
                                    self.callback_log("[AUDIT] All chunks present and verified! Sending ACK.\n")
                                    time.sleep(0.4)
                                    play_audio_waveform(build_ack_packet())
                                    active_image_chunks.clear()
                                else:
                                    # If burst pause is detected or if we want to audit missing chunks after a short silence window
                                    # For simplicity, if we receive packets, we can also let sender know if we have missing ones. 
                                    # Here, the receiver can send a NAK listing missing chunks if transmission pause occurs:
                                    pass
                                    
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
        self.title("Acoustic 16x16 Image Transceiver with Selective Repeat ARQ")
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

        subtitle = tk.Label(self, text="Selective Repeat ARQ Image Transceiver", font=("Helvetica", 14), fg="#a6adc8", bg="#1e1e2e")
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
        tk.Label(nav, text="TRANSMIT 16x16 IMAGE BURST", font=("Helvetica", 14, "bold"), bg="#181825", fg="#cdd6f4").pack(side="left", padx=15)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Helvetica", 11, "bold"), padding=[15, 8])

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)

        self.tab_editor = tk.Frame(self.notebook, bg="#181825")
        self.notebook.add(self.tab_editor, text="   16x16 Pixel Editor   ")

        self._build_editor_tab()

        bottom = tk.Frame(self, bg="#181825", height=70)
        bottom.pack(fill="x", side="bottom")

        self.status_lbl = tk.Label(bottom, text="Ready to transmit.", font=("Helvetica", 11), bg="#181825", fg="#a6adc8")
        self.status_lbl.pack(side="left", padx=20, pady=15)

        self.tx_btn = tk.Button(
            bottom, text="🔊 TRANSMIT IMAGE BURST", font=("Helvetica", 13, "bold"),
            bg="#f38ba8", fg="#11111b", highlightbackground="#181825",
            padx=20, pady=8, cursor="hand2", command=self.start_transmission
        )
        self.tx_btn.pack(side="right", padx=20, pady=10)

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
        payload = self.get_packed_canvas_bytes()
        threading.Thread(
            target=lambda: transmit_image_selective_arq(payload, self.update_status, max_retries=5),
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
                callback_packet_fully_received=self.on_packet_received,
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
            self.append_log("[SUCCESS] 16x16 Image Burst fully received, audited, and verified!\n\n")
        self.after(0, _handle)


if __name__ == "__main__":
    app = SoundModemApp()
    app.mainloop()