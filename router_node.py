import struct
import zlib
import threading
import time
import os
from datetime import datetime
import numpy as np
import sounddevice as sd
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image

SAMPLE_RATE = 44100
SYMBOL_DURATION = 0.100
TONE_DURATION = 0.080
SYMBOL_SAMPLES = int(SAMPLE_RATE * SYMBOL_DURATION)
WINDOW_SAMPLES = int(SAMPLE_RATE * 0.060)
STEP_SAMPLES = int(SAMPLE_RATE * 0.010)
CHUNK_SIZE = 32

TYPE_TEXT = 0x01
TYPE_IMAGE = 0x02
TYPE_FILE = 0x03
TYPE_ACK = 0x04
TYPE_NAK = 0x05
TYPE_HS_REQ = 0x10
TYPE_HS_RESP = 0x11
TYPE_TARGET_REQ = 0x12
TYPE_TARGET_RESP = 0x13

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

PALETTE_RGB = [hex_to_rgb(c) for c in PALETTE]

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
    if payload_type in (TYPE_ACK, TYPE_NAK, TYPE_HS_RESP, TYPE_TARGET_RESP):
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

class JupyterRouterThread(threading.Thread):
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

    def run(self):
        self.running = True
        self.log_callback(f"[ROUTER START] Listening on Router Address: {self.router_addr}")
        
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
                                            self.log_callback(f"[HANDSHAKE] Verified connection from Sender {s_addr}. Repeating confirmation...")
                                            self._send_response(stream, ring_buffer, full_payload[1:5], TYPE_HS_RESP)
                                    
                                    # Target Designation
                                    elif p_type == TYPE_TARGET_REQ and len(full_payload) >= 5:
                                        r_addr, t_addr = struct.unpack('>HH', full_payload[1:5])
                                        if r_addr == self.router_addr:
                                            self.active_target = t_addr
                                            self.log_callback(f"[ROUTING] Target designated: {t_addr}. Acknowledging...")
                                            self._send_response(stream, ring_buffer, full_payload[1:5], TYPE_TARGET_RESP)

                                    # Data Packets (Text, Image, File)
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
                                            self.log_callback(f"[ROUTER RX] Packet {seq_num + 1}/{total_pkts} received from {self.active_sender}. Sending ACK.")
                                            self._send_response(stream, ring_buffer, struct.pack('>H', seq_num), TYPE_ACK)
                                            expected_seq += 1

                                            if expected_seq == total_pkts:
                                                complete_data = b"".join(reassembly_buffer[i] for i in range(total_pkts))
                                                self.log_callback(f"[ROUTER SUCCESS] All packets gathered from {self.active_sender} for target {self.active_target}!")
                                                self.data_callback(active_transfer_type, complete_data, self.active_sender, self.active_target)
                                                
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
            self.log_callback(f"[ROUTER ERROR] {e}")

    def stop(self):
        self.running = False

# --- Jupyter ipywidgets Dashboard ---
router_id_input = widgets.IntText(value=999, description='Router ID:')
toggle_btn = widgets.Button(description='🚀 Start Router', button_style='success')
status_label = widgets.HTML(value='<b>Status:</b> Offline')

log_output = widgets.Output(layout={'border': '1px solid #ccc', 'height': '180px', 'overflow_y': 'scroll', 'background-color': '#1e1e2e', 'color': '#cdd6f4'})
data_output = widgets.Output(layout={'border': '1px solid #ccc', 'height': '280px', 'overflow_y': 'scroll', 'padding': '10px'})

router_thread = None

def log_msg(msg):
    with log_output:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def display_received_data(p_type, data, sender, target):
    with data_output:
        clear_output(wait=True)
        print(f"--- Stored Data Ready to Forward (From: {sender} -> Target: {target}) ---")
        if p_type == TYPE_TEXT:
            try:
                print(f"Text Message:\n{data.decode('utf-8')}")
            except Exception as e:
                print(f"Decoded binary text error: {e}")
        elif p_type == TYPE_IMAGE:
            print("Reconstructed 16x16 Pixel Image:")
            img = Image.new("RGB", (16, 16))
            flat_nibbles = []
            for b in data:
                flat_nibbles.append((b >> 4) & 0x0F)
                flat_nibbles.append(b & 0x0F)
            for idx, color_idx in enumerate(flat_nibbles[:256]):
                r = idx // 16
                c = idx % 16
                img.putpixel((c, r), hex_to_rgb(PALETTE[color_idx]))
            display(img.resize((200, 200), Image.Resampling.NEAREST))
        elif p_type == TYPE_FILE:
            os.makedirs("router_stored", exist_ok=True)
            filename = datetime.now().strftime("forward_%Y%m%d_%H%M%S.bin")
            filepath = os.path.join("router_stored", filename)
            with open(filepath, "wb") as f:
                f.write(data)
            print(f"Stored Binary File ({len(data)} bytes) saved locally to: {filepath}")

def on_toggle_clicked(b):
    global router_thread
    if toggle_btn.description.startswith('🚀'):
        r_addr = router_id_input.value
        router_thread = JupyterRouterThread(router_addr=r_addr, log_callback=log_msg, data_callback=display_received_data)
        router_thread.start()
        toggle_btn.description = '⏹️ Stop Router'
        toggle_btn.button_style = 'danger'
        status_label.value = f'<b>Status:</b> Listening on ID {r_addr}'
    else:
        if router_thread:
            router_thread.stop()
        toggle_btn.description = '🚀 Start Router'
        toggle_btn.button_style = 'success'
        status_label.value = '<b>Status:</b> Offline'
        log_msg("Router stopped.")

toggle_btn.on_click(on_toggle_clicked)

dashboard = widgets.VBox([
    widgets.HBox([router_id_input, toggle_btn, status_label]),
    widgets.Label("Live Router Event Log:"),
    log_output,
    widgets.Label("Stored Data / Reassembled Payload Preview:"),
    data_output
])

display(dashboard)