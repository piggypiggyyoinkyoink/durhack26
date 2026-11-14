import struct
import zlib
import random
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
SYMBOL_DURATION = 0.060  # 60 ms total per symbol
TONE_DURATION = 0.050    # 50 ms tone + 10 ms guard silence
SYMBOL_SAMPLES = int(SAMPLE_RATE * SYMBOL_DURATION)
WINDOW_SAMPLES = int(SAMPLE_RATE * 0.040)  # 40ms analysis window
STEP_SAMPLES = int(SAMPLE_RATE * 0.010)    # 10ms step during sync search
GUARD_SAMPLES = int(SAMPLE_RATE * 0.010)   # analysis window ends 10 ms before "now"

CHUNK_SIZE = 32  # data bytes per chunk

# Payload Type Identifiers
TYPE_TEXT = 0x01
TYPE_IMAGE = 0x02
TYPE_FILE = 0x03
TYPE_ACK = 0x04
TYPE_NAK = 0x05

# Packet kind byte for data chunks (first byte of a chunk's payload).
# ACK/NAK start with TYPE_ACK/TYPE_NAK, so the receiver can never mistake
# its own (or anybody's) ACK/NAK for a data chunk, and vice versa.
PKT_DATA = 0x10
HEADER_LEN = 5                      # [PKT_DATA, msg_id, idx, total, ptype]
MAX_CHUNK_PAYLOAD = HEADER_LEN + CHUNK_SIZE

# 16 Data Frequencies: 8000 Hz to 11750 Hz (250 Hz spacing)
DATA_FREQS = np.array([8000 + i * 250 for i in range(16)])
SYNC_FREQ_A = 12250
SYNC_FREQ_B = 12750
ALL_FREQS = np.append(DATA_FREQS, [SYNC_FREQ_A, SYNC_FREQ_B])
EXPECTED_SYNC = [SYNC_FREQ_A, SYNC_FREQ_B] * 3

FFT_SIZE = 8192
HANN = np.hanning(WINDOW_SAMPLES)
FFT_FREQS = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
FREQ_BINS = [np.argmin(np.abs(FFT_FREQS - f)) for f in ALL_FREQS]
SNR_THRESHOLD = 6.0

# ---- Protocol timing -------------------------------------------------------
# Time one full chunk occupies the air (preamble + data + trailing gaps).
CHUNK_AIR_TIME = (6 + 2 * (2 + MAX_CHUNK_PAYLOAD + 4)) * SYMBOL_DURATION + 0.5
# The receiver only talks back after the channel has been silent for longer
# than a whole chunk, so it can never talk over a transmitter that is still
# sending (e.g. when it missed a chunk's preamble).
IDLE_TIMEOUT = CHUNK_AIR_TIME + 1.5
REPLY_DELAY = 0.6        # give the transmitter time to open its mic stream
SESSION_TIMEOUT = 60.0   # drop a half-received message after this much silence
DUP_WINDOW = 60.0        # remember a finished message id this long (for re-ACK)

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
    fade_len = int(sample_rate * 0.003)

    envelope = np.ones(tone_samples)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)

    wave = amplitude * np.sin(2 * np.pi * freq * t) * envelope
    guard_samples = total_samples - tone_samples
    if guard_samples > 0:
        wave = np.pad(wave, (0, guard_samples), mode='constant')
    return wave


def _frame(payload: bytes) -> bytes:
    """[len:2][payload][crc32:4]"""
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return struct.pack('>H', len(payload)) + payload + struct.pack('>I', crc)

def build_chunk_packet(msg_id, chunk_idx, total_chunks, payload_type, chunk_data):
    header = bytes([PKT_DATA, msg_id, chunk_idx, total_chunks, payload_type])
    return _frame(header + chunk_data)

def build_ack_packet(msg_id):
    return _frame(bytes([TYPE_ACK, msg_id]))

def build_nak_packet(msg_id, missing_indices):
    missing = sorted(missing_indices)[:255]
    return _frame(bytes([TYPE_NAK, msg_id, len(missing)] + missing))

def packet_to_waveform(packet_bytes):
    bitstring = "".join(f"{byte:08b}" for byte in packet_bytes)
    waves = []
    for freq in EXPECTED_SYNC:
        waves.append(generate_tone(freq, TONE_DURATION, SYMBOL_DURATION))
    for i in range(0, len(bitstring), 4):
        nibble_val = int(bitstring[i:i+4], 2)
        waves.append(generate_tone(DATA_FREQS[nibble_val], TONE_DURATION, SYMBOL_DURATION))
    waves.append(np.zeros(int(SAMPLE_RATE * 0.100), dtype=np.float32))
    return np.concatenate(waves).astype(np.float32)

def play_audio_waveform(packet_bytes):
    full_signal = packet_to_waveform(packet_bytes)
    time.sleep(0.25)  # Warmup pause for receiver input stream
    sd.play(full_signal, samplerate=SAMPLE_RATE)
    sd.wait()


# ============================================================================
# DECODER (one streaming demodulator shared by receiver AND ARQ listener)
# ============================================================================
def analyze_window(samples):

    samples = samples - np.mean(samples)

    windowed = samples * HANN

    spectrum = np.abs(
        np.fft.rfft(windowed, n=FFT_SIZE)
    )

    band_mask = (
        (FFT_FREQS >= 7500) &
        (FFT_FREQS <= 13500)
    )

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



class PacketDemodulator:
    """
    Streaming FSK demodulator.

    Feed STEP_SAMPLES (10 ms) of audio at a time.
    Returns:
        ("sync", None)    Preamble detected
        ("ok", payload)   Packet successfully decoded and CRC verified
        ("bad", None)     Invalid packet or CRC failure
    """

    def __init__(self, max_payload_len):
        self.max_payload_len = max_payload_len
        self.reset()

    def reset(self):
        self.ring = np.zeros(SYMBOL_SAMPLES * 7, dtype=np.float32)
        self.state = "SEARCH_SYNC"
        self.samples_until_next_read = 0
        self.nibbles = []
        self.expected_nibbles = None
        self.sync_armed = False

    @property
    def idle(self):
        return self.state == "SEARCH_SYNC" and not self.sync_armed

    def _lock(self):
        self.state = "RECEIVE_DATA"
        self.nibbles = []
        self.expected_nibbles = None
        self.sync_armed = False

        # Wait for the remainder of the first data symbol.
        self.samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES

    def feed(self, new_samples):

        self.ring = np.roll(self.ring, -STEP_SAMPLES)
        self.ring[-STEP_SAMPLES:] = new_samples

        # --------------------------------------------------
        # SEARCH FOR SYNCHRONIZATION PREAMBLE
        # --------------------------------------------------
        if self.state == "SEARCH_SYNC":

            detected = []

            for k in range(6):

                end = -((5 - k) * SYMBOL_SAMPLES + GUARD_SAMPLES)

                freq, snr, _ = analyze_window(
                    self.ring[end - WINDOW_SAMPLES:end]
                )

                detected.append(freq)

            if detected == EXPECTED_SYNC:

                self.sync_armed = True
                return None

            elif self.sync_armed:

                self._lock()
                return ("sync", None)

            return None

        # --------------------------------------------------
        # RECEIVE DATA SYMBOLS
        # --------------------------------------------------
        self.samples_until_next_read -= STEP_SAMPLES

        if self.samples_until_next_read > 0:
            return None

        self.samples_until_next_read += SYMBOL_SAMPLES

        window = self.ring[
            -GUARD_SAMPLES - WINDOW_SAMPLES:-GUARD_SAMPLES
        ]

        freq, snr, best_data_idx = analyze_window(window)

        # Reject noise and invalid frequencies.
        if freq is None or freq not in DATA_FREQS:
            self.reset()
            return ("bad", None)

        self.nibbles.append(best_data_idx)

        # --------------------------------------------------
        # DECODE PACKET LENGTH
        # --------------------------------------------------
        if len(self.nibbles) == 4:

            length_bytes = decode_nibbles_to_bytes(self.nibbles)

            length = struct.unpack(">H", length_bytes)[0]

            if length < 1 or length > self.max_payload_len:
                self.reset()
                return ("bad", None)

            self.expected_nibbles = (2 + length + 4) * 2

        # --------------------------------------------------
        # COMPLETE PACKET AND VERIFY CRC
        # --------------------------------------------------
        if (
            self.expected_nibbles is not None
            and len(self.nibbles) >= self.expected_nibbles
        ):

            raw = decode_nibbles_to_bytes(self.nibbles)

            payload = raw[2:-4]

            received_crc = struct.unpack(">I", raw[-4:])[0]

            calculated_crc = zlib.crc32(payload) & 0xFFFFFFFF

            self.reset()

            if received_crc == calculated_crc:
                return ("ok", payload)

            return ("bad", None)

        return None



# ============================================================================
# TRANSMITTER: burst + selective-repeat ARQ
# ============================================================================
_msg_counter = random.randint(1, 255)

def _next_msg_id():
    global _msg_counter
    _msg_counter = _msg_counter % 255 + 1   # 1..255, never repeats back-to-back
    return _msg_counter


def listen_for_audit_response(msg_id, total_chunks, timeout):
    """Returns "ACK", a set of missing chunk indices (NAK), or None (nothing heard)."""
    demod = PacketDemodulator(max_payload_len=3 + total_chunks)
    deadline = time.time() + timeout
    hard_deadline = deadline + 15.0
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=STEP_SAMPLES) as stream:
            # keep going past the deadline only to finish a packet already in progress
            while time.time() < deadline or (not demod.idle and time.time() < hard_deadline):
                audio, _ = stream.read(STEP_SAMPLES)
                ev = demod.feed(audio[:, 0])
                if ev is None or ev[0] != "ok":
                    continue
                p = ev[1]
                if len(p) < 2 or p[1] != msg_id:
                    continue  # not a reply to *this* message
                if p[0] == TYPE_ACK:
                    return "ACK"
                if p[0] == TYPE_NAK and len(p) >= 3:
                    n = p[2]
                    return set(p[3:3 + n])
    except Exception as e:
        print(f"[listen_for_audit_response] audio error: {e}")
    return None


def transmit_payload_selective_arq(payload_bytes, payload_type, callback_status, max_retries=6):
    """Send all chunks as a burst, listen for ACK/NAK, resend only what is missing."""
    if not payload_bytes:
        callback_status("Nothing to send (payload is empty).", "#f38ba8")
        return False

    chunks = [payload_bytes[i:i + CHUNK_SIZE] for i in range(0, len(payload_bytes), CHUNK_SIZE)]
    total_chunks = len(chunks)
    if total_chunks > 255:
        callback_status(f"Payload too large ({total_chunks} chunks, max 255 = {255 * CHUNK_SIZE} bytes).", "#f38ba8")
        return False

    msg_id = _next_msg_id()
    pending = set(range(total_chunks))
    listen_timeout = IDLE_TIMEOUT + 2.5 + 0.13 * total_chunks

    for attempt in range(1, max_retries + 1):
        callback_status(f"Burst transmitting {len(pending)} chunk(s) (attempt {attempt})...", "#89b4fa")
        print(f"\n--- Attempt {attempt}: msg {msg_id}, chunks {sorted(pending)} of {total_chunks} ---")

        for idx in sorted(pending):
            packet = build_chunk_packet(msg_id, idx, total_chunks, payload_type, chunks[idx])
            print(f"  Sending chunk {idx + 1}/{total_chunks}...")
            try:
                play_audio_waveform(packet)
            except Exception as e:
                callback_status(f"Transmission error: {e}", "#f38ba8")
                return False
            time.sleep(0.10)

        callback_status("Listening for audit response...", "#f9e2af")
        response = listen_for_audit_response(msg_id, total_chunks, listen_timeout)

        if response == "ACK":
            callback_status("Transmission SUCCESS! All chunks verified.", "#a6e3a1")
            print("ACK received.")
            return True
        elif isinstance(response, set):
            requested = {i for i in response if i < total_chunks}
            if requested:
                pending = requested
            callback_status(f"Receiver is missing {len(pending)} chunk(s). Resending them...", "#fab387")
            print(f"NAK received, resending: {sorted(pending)}")
        else:
            callback_status("No response heard. Retrying pending chunks...", "#fab387")
            print("No response heard. Retrying pending chunks...")

    callback_status("Transmission FAILED after max retries.", "#f38ba8")
    return False


# ============================================================================
# RECEIVER
# ============================================================================
class ReceiverThread(threading.Thread):
    def __init__(self, callback_packet_fully_received, callback_pixel_received, callback_log):
        super().__init__()
        self.daemon = True
        self.callback_packet_fully_received = callback_packet_fully_received
        self.callback_pixel_received = callback_pixel_received
        self.callback_log = callback_log
        self.running = False

    @staticmethod
    def _drain(stream):
        """Throw away audio captured while we were talking (it contains our own reply)."""
        try:
            while stream.read_available >= STEP_SAMPLES:
                stream.read(STEP_SAMPLES)
        except Exception:
            pass

    def _reply(self, packet, stream, demod):
        time.sleep(REPLY_DELAY)
        try:
            play_audio_waveform(packet)
        except Exception as e:
            self.callback_log(f"[ERROR] Could not play reply: {e}\n")
        self._drain(stream)
        demod.reset()

    def run(self):
        self.running = True
        self.callback_log("[INFO] Receiver listening for burst transmission...\n")

        demod = PacketDemodulator(max_payload_len=MAX_CHUNK_PAYLOAD)
        session = None          # {"msg_id","total","ptype","chunks":{idx: bytes}}
        last_done_id = None
        last_done_time = 0.0
        dup_seen = False
        last_event = time.time()   # last packet decoded or reply sent
        last_rx = last_event       # last valid chunk received

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=STEP_SAMPLES) as stream:
                while self.running:
                    audio, _ = stream.read(STEP_SAMPLES)
                    ev = demod.feed(audio[:, 0])
                    now = time.time()

                    if ev is not None:
                        kind, payload = ev
                        if kind == "sync":
                            self.callback_log("\n[SYNC LOCKED] Receiving chunk...\n")
                        elif kind == "bad":
                            last_event = now
                            self.callback_log("  [CRC ERROR] A chunk failed verification.\n")
                        elif len(payload) >= HEADER_LEN and payload[0] == PKT_DATA:
                            last_event = now
                            msg_id, idx, total, ptype = payload[1], payload[2], payload[3], payload[4]
                            data = payload[HEADER_LEN:]

                            if not (idx < total and ptype in (TYPE_TEXT, TYPE_IMAGE, TYPE_FILE)):
                                self.callback_log("  [IGNORED] Malformed chunk header.\n")

                            elif last_done_id == msg_id and now - last_done_time < DUP_WINDOW:
                                # Transmitter never heard our ACK and is re-sending: re-ACK once it goes quiet.
                                dup_seen = True
                                self.callback_log(f"  [DUP] Chunk {idx + 1}/{total} of an already-delivered message.\n")

                            else:
                                if session is None or session["msg_id"] != msg_id or session["total"] != total:
                                    session = {"msg_id": msg_id, "total": total, "ptype": ptype, "chunks": {}}
                                    self.callback_log(f"[NEW MESSAGE] id {msg_id}, {total} chunk(s).\n")
                                session["chunks"][idx] = data
                                last_rx = now
                                self.callback_log(f"  [CRC OK] Chunk {idx + 1}/{total} received "
                                                  f"(Type {ptype}, {len(session['chunks'])}/{total} have).\n")

                                if ptype == TYPE_IMAGE:
                                    base_pixel_idx = idx * CHUNK_SIZE * 2
                                    for p_offset in range(len(data) * 2):
                                        b_val = data[p_offset // 2]
                                        color_idx = (b_val >> 4) if p_offset % 2 == 0 else (b_val & 0x0F)
                                        abs_pixel_idx = base_pixel_idx + p_offset
                                        if abs_pixel_idx < 256:
                                            self.callback_pixel_received(abs_pixel_idx // 16, abs_pixel_idx % 16, color_idx)

                                # Everything here? Deliver and ACK right away.
                                if len(session["chunks"]) == total:
                                    complete = b"".join(session["chunks"][i] for i in range(total))
                                    self.callback_packet_fully_received(bytes([session["ptype"]]) + complete)
                                    self.callback_log("[AUDIT] All chunks present and verified! Sending ACK.\n")
                                    self._reply(build_ack_packet(msg_id), stream, demod)
                                    last_done_id, last_done_time = msg_id, time.time()
                                    session, dup_seen = None, False
                                    last_event = time.time()
                                    continue
                        # else: a valid packet that is not a data chunk (e.g. an echo of ACK/NAK): ignore.

                    # Channel silent for longer than a whole chunk and we are NOT mid-packet:
                    # the burst is over, so it is safe to talk.
                    if demod.idle and now - last_event > IDLE_TIMEOUT:
                        if dup_seen:
                            self.callback_log("[AUDIT] Duplicate burst of a delivered message. Re-sending ACK.\n")
                            self._reply(build_ack_packet(last_done_id), stream, demod)
                            dup_seen = False
                            last_event = time.time()
                        elif session is not None:
                            missing = sorted(set(range(session["total"])) - set(session["chunks"]))
                            self.callback_log(f"[AUDIT] Burst ended. Missing chunks: {[m + 1 for m in missing]}. Sending NAK.\n")
                            # NOTE: session is KEPT, so retransmitted chunks complete the message.
                            self._reply(build_nak_packet(session["msg_id"], missing), stream, demod)
                            last_event = time.time()

                    if session is not None and now - last_rx > SESSION_TIMEOUT:
                        self.callback_log("[INFO] Sender went silent. Discarding incomplete message.\n")
                        session = None
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
        self.title("Acoustic Data & Image Transceiver with Selective ARQ")
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

        subtitle = tk.Label(self, text="Selective Repeat ARQ Acoustic Transceiver", font=("Helvetica", 14), fg="#a6adc8", bg="#1e1e2e")
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
        tk.Label(nav, text="TRANSMIT DATA / IMAGE WITH SELECTIVE ARQ", font=("Helvetica", 14, "bold"), bg="#181825", fg="#cdd6f4").pack(side="left", padx=15)

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
        self.text_input.insert("1.0", "Hello World! Testing acoustic modem with Selective ARQ.")

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
        self.file_info_lbl.config(text=f"Selected: {os.path.basename(path)} ({len(self.raw_file_bytes)} bytes / {size_kb:.2f} KB)")
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
        # The receiver would hear our own transmission and answer it with ACK/NAKs.
        if self.controller.frames["ReceivePage"].is_listening:
            messagebox.showwarning(
                "Receiver active",
                "Stop listening on the Receive page before transmitting from this same app."
            )
            return

        active_tab = self.notebook.index(self.notebook.select())
        if active_tab == 0:
            text = self.text_input.get("1.0", "end-1c").strip()
            if not text:
                messagebox.showwarning("Empty", "Please enter some text first.")
                return
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

        self.tx_btn.config(state="disabled")

        def worker():
            try:
                transmit_payload_selective_arq(payload, payload_type, self.update_status)
            finally:
                self.after(0, lambda: self.tx_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()


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
        tk.Label(nav, text="RECEIVER & LIVE DECODER", font=("Helvetica", 14, "bold"), bg="#181825", fg="#cdd6f4").pack(side="left", padx=15)

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
            payload_type = full_payload[0]
            actual_data = full_payload[1:]

            if payload_type == TYPE_IMAGE:
                self.append_log("[SUCCESS] 16x16 Image Burst fully received and verified!\n\n")
            elif payload_type == TYPE_TEXT:
                try:
                    text_msg = actual_data.decode("utf-8")
                    self.append_log(f"Decoded Message Burst:\n{text_msg}\n[SUCCESS] Verified.\n\n")
                except UnicodeDecodeError:
                    self.append_log("[ERROR] Text payload contained invalid UTF-8 bytes.\n")
            elif payload_type == TYPE_FILE:
                self.append_log(f"Received binary file burst ({len(actual_data)} bytes). Saving to ./received/\n")
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