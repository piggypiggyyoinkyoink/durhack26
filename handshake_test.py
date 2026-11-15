import struct
import zlib
import time
import numpy as np
import sounddevice as sd

# ============================================================================
# CONFIGURATION (Aligned with sound_modem_3.py robust standards)
# ============================================================================
SAMPLE_RATE = 44100
SYMBOL_DURATION = 0.100  # 100 ms per symbol
TONE_DURATION = 0.080    # 80 ms tone + 20 ms guard silence
SYMBOL_SAMPLES = int(SAMPLE_RATE * SYMBOL_DURATION)
WINDOW_SAMPLES = int(SAMPLE_RATE * 0.060)  # 60ms analysis window
STEP_SAMPLES = int(SAMPLE_RATE * 0.010)    # 10ms step during sync search

# Payload Types
TYPE_HS_REQ = 0x10
TYPE_HS_RESP = 0x11
TYPE_ACK = 0x04
TYPE_NAK = 0x05

DATA_FREQS = np.array([8000 + i * 250 for i in range(16)])
SYNC_FREQ_A = 12250
SYNC_FREQ_B = 12750
ALL_FREQS = np.append(DATA_FREQS, [SYNC_FREQ_A, SYNC_FREQ_B])

FFT_SIZE = 8192
HANN = np.hanning(WINDOW_SAMPLES)
FFT_FREQS = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
FREQ_BINS = [np.argmin(np.abs(FFT_FREQS - f)) for f in ALL_FREQS]
SNR_THRESHOLD = 6.0

def generate_tone(freq, duration, total_slot_duration, amplitude=0.8):
    tone_samples = int(SAMPLE_RATE * duration)
    total_samples = int(SAMPLE_RATE * total_slot_duration)
    t = np.linspace(0, duration, tone_samples, endpoint=False)
    fade_len = int(SAMPLE_RATE * 0.005)
    
    envelope = np.ones(tone_samples)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    
    wave = amplitude * np.sin(2 * np.pi * freq * t) * envelope
    if (total_samples - tone_samples) > 0:
        wave = np.pad(wave, (0, total_samples - tone_samples), mode='constant')
    return wave

def build_packet(data_payload, payload_type):
    """
    Robust packet format:
    [2B Length] + [1B Type | N-B Data] + [4B CRC32]
    """
    full_payload = bytes([payload_type]) + data_payload
    crc = zlib.crc32(full_payload) & 0xFFFFFFFF
    return struct.pack('>H', len(full_payload)) + full_payload + struct.pack('>I', crc)

def play_audio_waveform(data_payload, payload_type):
    packet = build_packet(data_payload, payload_type)
    bitstring = "".join(f"{byte:08b}" for byte in packet)
    
    waves = []
    for freq in [SYNC_FREQ_A, SYNC_FREQ_B] * 3:
        waves.append(generate_tone(freq, TONE_DURATION, SYMBOL_DURATION))
    for i in range(0, len(bitstring), 4):
        waves.append(generate_tone(DATA_FREQS[int(bitstring[i:i+4], 2)], TONE_DURATION, SYMBOL_DURATION))
    waves.append(generate_tone(SYNC_FREQ_A, TONE_DURATION, SYMBOL_DURATION)) # Postamble guard tone
    waves.append(np.zeros(int(SAMPLE_RATE * 0.150), dtype=np.float32))
    
    sd.play(np.concatenate(waves).astype(np.float32), samplerate=SAMPLE_RATE)
    sd.wait()

def analyze_window(samples):
    samples = samples - np.mean(samples)
    windowed = samples * HANN
    spectrum = np.abs(np.fft.rfft(windowed, n=FFT_SIZE))
    
    band_mask = (FFT_FREQS >= 7500) & (FFT_FREQS <= 13500)
    noise_floor = np.median(spectrum[band_mask]) + 1e-9
    
    strengths = spectrum[FREQ_BINS]
    best_idx = int(np.argmax(strengths))
    snr = strengths[best_idx] / noise_floor
    best_data_idx = int(np.argmax(strengths[:16]))
    
    if snr < SNR_THRESHOLD:
        return None, snr, best_data_idx
    return ALL_FREQS[best_idx], snr, best_data_idx

def decode_nibbles_to_bytes(nibbles):
    bitstring = "".join(format(n, "04b") for n in nibbles)
    return bytes([int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)])

def listen_for_packet(timeout=4.0):
    """Robust listener using 25ms offset and full CRC verification."""
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
                        end_offset = -((5 - k) * SYMBOL_SAMPLES + int(SAMPLE_RATE * 0.025)) # 25ms sweet spot
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
                        end_offset = -int(SAMPLE_RATE * 0.025) # 25ms sweet spot
                        start_offset = end_offset - WINDOW_SAMPLES
                        _, _, best_data_idx = analyze_window(ring_buffer[start_offset:end_offset])
                        nibbles.append(best_data_idx)
                        
                        if len(nibbles) == 4:
                            length_bytes = decode_nibbles_to_bytes(nibbles[:4])
                            payload_len = struct.unpack('>H', length_bytes)[0]
                            expected_nibbles = (2 + payload_len + 4) * 2

                        if expected_nibbles is not None and len(nibbles) >= expected_nibbles:
                            raw_bytes = decode_nibbles_to_bytes(nibbles)
                            full_payload = raw_bytes[2:-4]
                            received_crc = struct.unpack('>I', raw_bytes[-4:])[0]
                            calculated_crc = zlib.crc32(full_payload) & 0xFFFFFFFF
                            
                            if received_crc == calculated_crc and len(full_payload) >= 1:
                                return full_payload[0], full_payload[1:]
                            return None, None
                            
                        samples_until_next_read += SYMBOL_SAMPLES
    except Exception:
        pass
    return None, None

def run_receiver(router_addr=999):
    print(f"\n--- ROBUST DIAGNOSTIC RECEIVER (Router ID {router_addr}) ---")
    print("Listening for robust handshake requests with ACK/NAK feedback...")
    
    while True:
        p_type, payload = listen_for_packet(timeout=10.0)
        if p_type == TYPE_HS_REQ and len(payload) >= 4:
            r_addr, s_addr = struct.unpack('>HH', payload[:4])
            if r_addr == router_addr:
                print(f"[SUCCESS] Valid Handshake Request verified from Endpoint {s_addr}!")
                
                # Send ACK response back robustly
                print("[AUDIO] Sending robust Handshake Response (HS_RESP)...")
                time.sleep(0.3)
                play_audio_waveform(struct.pack('>HH', r_addr, s_addr), TYPE_HS_RESP)
                print("[SUCCESS] Response sent successfully. Ready for next request.")

def run_sender(router_addr=999, my_addr=101, max_retries=3):
    print(f"\n--- ROBUST DIAGNOSTIC SENDER (Endpoint {my_addr} -> Router {router_addr}) ---")
    payload = struct.pack('>HH', router_addr, my_addr)
    
    for attempt in range(1, max_retries + 1):
        print(f"Transmitting Handshake Request (Attempt {attempt}/{max_retries})...")
        play_audio_waveform(payload, TYPE_HS_REQ)
        
        print("Listening for Router response...")
        time.sleep(0.3)
        p_type, resp_payload = listen_for_packet(timeout=4.5)
        
        if p_type == TYPE_HS_RESP:
            print("[SUCCESS] Handshake fully acknowledged and verified by Router!")
            return True
        else:
            print(f"[WARN] Timeout or CRC mismatch on attempt {attempt}. Retrying...")
            
    print("[FAIL] Handshake failed after maximum retries.")
    return False

if __name__ == "__main__":
    choice = input("Select mode:\n1. Receiver (Router side)\n2. Sender (Endpoint side)\nEnter 1 or 2: ").strip()
    if choice == "1":
        run_receiver()
    elif choice == "2":
        run_sender()
    else:
        print("Invalid choice.")