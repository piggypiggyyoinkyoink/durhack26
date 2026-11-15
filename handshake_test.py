import struct
import zlib
import time
import numpy as np
import sounddevice as sd

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLE_RATE = 44100
SYMBOL_DURATION = 0.100  # 100 ms per symbol
TONE_DURATION = 0.080    # 80 ms tone + 20 ms guard
SYMBOL_SAMPLES = int(SAMPLE_RATE * SYMBOL_DURATION)
WINDOW_SAMPLES = int(SAMPLE_RATE * 0.060)
STEP_SAMPLES = int(SAMPLE_RATE * 0.010)

TYPE_HS_REQ = 0x10
TYPE_HS_RESP = 0x11

DATA_FREQS = np.array([8000 + i * 250 for i in range(16)])
SYNC_FREQ_A = 12250
SYNC_FREQ_B = 12750
ALL_FREQS = np.append(DATA_FREQS, [SYNC_FREQ_A, SYNC_FREQ_B])

FFT_SIZE = 8192
HANN = np.hanning(WINDOW_SAMPLES)
FFT_FREQS = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
FREQ_BINS = [np.argmin(np.abs(FFT_FREQS - f)) for f in ALL_FREQS]
SNR_THRESHOLD = 4.0  # Lowered slightly for easier diagnostic testing

def generate_tone(freq, duration, total_slot_duration):
    tone_samples = int(SAMPLE_RATE * duration)
    total_samples = int(SAMPLE_RATE * total_slot_duration)
    t = np.linspace(0, duration, tone_samples, endpoint=False)
    envelope = np.ones(tone_samples)
    fade_len = int(SAMPLE_RATE * 0.005)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    wave = 0.8 * np.sin(2 * np.pi * freq * t) * envelope
    if total_samples > tone_samples:
        wave = np.pad(wave, (0, total_samples - tone_samples), mode='constant')
    return wave

def build_packet(data_payload, payload_type):
    full_payload = bytes([payload_type]) + data_payload
    crc = zlib.crc32(full_payload) & 0xFFFFFFFF
    return struct.pack('>H', len(full_payload)) + full_payload + struct.pack('>I', crc)

def play_waveform(data_payload, payload_type):
    packet = build_packet(data_payload, payload_type)
    bitstring = "".join(f"{byte:08b}" for byte in packet)
    
    waves = []
    # Preamble sync tones
    for freq in [SYNC_FREQ_A, SYNC_FREQ_B] * 3:
        waves.append(generate_tone(freq, TONE_DURATION, SYMBOL_DURATION))
    # Data nibbles
    for i in range(0, len(bitstring), 4):
        nibble = int(bitstring[i:i+4], 2)
        waves.append(generate_tone(DATA_FREQS[nibble], TONE_DURATION, SYMBOL_DURATION))
    # Postamble sync tone
    waves.append(generate_tone(SYNC_FREQ_A, TONE_DURATION, SYMBOL_DURATION))
    waves.append(np.zeros(int(SAMPLE_RATE * 0.150), dtype=np.float32))
    
    signal = np.concatenate(waves).astype(np.float32)
    print("[AUDIO] Playing transmission waveform...")
    sd.play(signal, samplerate=SAMPLE_RATE)
    sd.wait()
    print("[AUDIO] Playback complete.")

def analyze_window(samples):
    samples = samples - np.mean(samples)
    spectrum = np.abs(np.fft.rfft(samples * HANN, n=FFT_SIZE))
    noise_floor = np.median(spectrum[(FFT_FREQS >= 7500) & (FFT_FREQS <= 13500)]) + 1e-9
    strengths = spectrum[FREQ_BINS]
    best_idx = int(np.argmax(strengths))
    snr = strengths[best_idx] / noise_floor
    if snr < SNR_THRESHOLD:
        return None, snr
    return ALL_FREQS[best_idx], snr

def decode_nibbles(nibbles):
    bitstring = "".join(format(n, "04b") for n in nibbles)
    return bytes([int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)])

def run_receiver(router_addr=999):
    print(f"\n--- DIAGNOSTIC RECEIVER (Listening on ID {router_addr}) ---")
    print("Make sure your speakers/mic are active and volume is turned up.")
    
    ring_buffer = np.zeros(SYMBOL_SAMPLES * 7, dtype=np.float32)
    state = "SEARCH_SYNC"
    samples_left = 0
    nibbles = []
    expected_nibbles = None
    expected_sync = [SYNC_FREQ_A, SYNC_FREQ_B] * 3
    sync_armed = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=STEP_SAMPLES) as stream:
        while True:
            audio, _ = stream.read(STEP_SAMPLES)
            ring_buffer = np.roll(ring_buffer, -STEP_SAMPLES)
            ring_buffer[-STEP_SAMPLES:] = audio[:, 0]
            
            if state == "SEARCH_SYNC":
                detected_sync = []
                max_snr = 0.0
                for k in range(6):
                    offset = -((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.010))
                    freq, snr = analyze_window(ring_buffer[offset - WINDOW_SAMPLES:offset])
                    detected_sync.append(freq)
                    max_snr = max(max_snr, snr)

                if max_snr > 2.0:
                    print(f"[DEBUG] Listening... Max SNR in window: {max_snr:.2f} (Threshold: {SNR_THRESHOLD})", end='\r')

                if detected_sync == expected_sync:
                    print(f"\n[SYNC] Preamble matched! SNR: {max_snr:.2f}")
                    state = "RECEIVE_DATA"
                    nibbles = []
                    expected_nibbles = None
                    sync_armed = False
                    samples_left = SYMBOL_SAMPLES - STEP_SAMPLES
                    
            elif state == "RECEIVE_DATA":
                samples_left -= STEP_SAMPLES
                if samples_left <= 0:
                    offset = -int(SAMPLE_RATE * 0.010)
                    _, _, best_data = analyze_window_data(ring_buffer[offset - WINDOW_SAMPLES:offset])
                    nibbles.append(best_data)
                    
                    if len(nibbles) == 4:
                        length_bytes = decode_nibbles(nibbles[:4])
                        payload_len = struct.unpack('>H', length_bytes)[0]
                        expected_nibbles = (2 + payload_len + 4) * 2

                    if expected_nibbles is not None and len(nibbles) >= expected_nibbles:
                        raw = decode_nibbles(nibbles)
                        full_payload = raw[2:-4]
                        received_crc = struct.unpack('>I', raw[-4:])[0]
                        
                        if received_crc == (zlib.crc32(full_payload) & 0xFFFFFFFF) and len(full_payload) >= 5:
                            p_type = full_payload[0]
                            if p_type == TYPE_HS_REQ:
                                r_addr, s_addr = struct.unpack('>HH', full_payload[1:5])
                                print(f"[SUCCESS] Handshake received! Router Addr: {r_addr}, Sender Addr: {s_addr}")
                                
                                # Send Response Back
                                print("[AUDIO] Sending handshake response (HS_RESP)...")
                                play_waveform(struct.pack('>HH', r_addr, s_addr), TYPE_HS_RESP)
                                print("[SUCCESS] Response sent. Returning to listen mode.")
                        else:
                            print("[ERROR] CRC validation failed on packet.")
                            
                        state = "SEARCH_SYNC"
                        continue
                    samples_left += SYMBOL_SAMPLES

def analyze_window_data(samples):
    samples = samples - np.mean(samples)
    spectrum = np.abs(np.fft.rfft(samples * HANN, n=FFT_SIZE))
    noise_floor = np.median(spectrum[(FFT_FREQS >= 7500) & (FFT_FREQS <= 13500)]) + 1e-9
    strengths = spectrum[FREQ_BINS]
    best_idx = int(np.argmax(strengths))
    snr = strengths[best_idx] / noise_floor
    return ALL_FREQS[best_idx], snr, int(np.argmax(strengths[:16]))

def run_sender(router_addr=999, my_addr=101):
    print(f"\n--- DIAGNOSTIC SENDER (Sending HS_REQ to Router {router_addr} from {my_addr}) ---")
    payload = struct.pack('>HH', router_addr, my_addr)
    play_waveform(payload, TYPE_HS_REQ)
    print("[SENDER] Handshake request transmitted.")

if __name__ == "__main__":
    choice = input("Select mode:\n1. Receiver (Router side)\n2. Sender (Endpoint side)\nEnter 1 or 2: ").strip()
    if choice == "1":
        run_receiver()
    elif choice == "2":
        run_sender()
    else:
        print("Invalid choice.")