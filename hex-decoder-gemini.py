import struct
import zlib
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
SYMBOL_SAMPLES = int(SAMPLE_RATE * 0.100)  # 4410 samples per 100ms slot
WINDOW_SAMPLES = int(SAMPLE_RATE * 0.060)  # 2646 samples (60ms analysis window)
STEP_SAMPLES = int(SAMPLE_RATE * 0.010)    # 441 samples (10ms step during sync search)

DATA_FREQS = np.array([8000 + i * 250 for i in range(16)])
SYNC_FREQ_A = 12250
SYNC_FREQ_B = 12750
ALL_FREQS = np.append(DATA_FREQS, [SYNC_FREQ_A, SYNC_FREQ_B])

FFT_SIZE = 8192
HANN = np.hanning(WINDOW_SAMPLES)
FFT_FREQS = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
FREQ_BINS = [np.argmin(np.abs(FFT_FREQS - f)) for f in ALL_FREQS]

SNR_THRESHOLD = 6.0

def analyze_window(samples):
    """Returns (detected_freq, snr_ratio, data_nibble_idx)."""
    samples = samples - np.mean(samples)
    windowed = samples * HANN
    spectrum = np.abs(np.fft.rfft(windowed, n=FFT_SIZE))
    
    # Local noise floor in target band (7.5 kHz - 13.5 kHz)
    band_mask = (FFT_FREQS >= 7500) & (FFT_FREQS <= 13500)
    noise_floor = np.median(spectrum[band_mask]) + 1e-9
    
    strengths = spectrum[FREQ_BINS]
    best_idx = int(np.argmax(strengths))
    peak_val = strengths[best_idx]
    
    snr = peak_val / noise_floor
    
    # Best data-only candidate (indices 0..15)
    best_data_idx = int(np.argmax(strengths[:16]))
    
    if snr < SNR_THRESHOLD:
        return None, snr, best_data_idx
        
    return ALL_FREQS[best_idx], snr, best_data_idx

def decode_nibbles_to_bytes(nibbles):
    bitstring = "".join(format(n, "04b") for n in nibbles)
    byte_list = [int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)]
    return bytes(byte_list)

def main():
    print("Listening for acoustic transmission... (Press CTRL+C to stop)")
    
    # Hold 7 full symbol slots so we can inspect all 6 preamble tones cleanly
    ring_buffer = np.zeros(SYMBOL_SAMPLES * 7, dtype=np.float32)
    state = "SEARCH_SYNC"
    samples_until_next_read = 0
    
    nibbles = []
    expected_nibbles = None
    expected_sync = [SYNC_FREQ_A, SYNC_FREQ_B] * 3  # All 6 preamble tones
    # Add these two tracking variables right above `with sd.InputStream(...)`:
    sync_armed = False
    best_sync_snr = 0.0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=STEP_SAMPLES) as stream:
        while True:
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
                    # Wait until the 6th sync tone reaches its peak energy
                    if last_snr >= best_sync_snr:
                        best_sync_snr = last_snr
                        sync_armed = True
                    elif sync_armed:
                        # Energy just peaked on the previous 10ms step! Lock now.
                        print("\n[SYNC LOCKED] Receiving packet...")
                        state = "RECEIVE_DATA"
                        nibbles = []
                        expected_nibbles = None
                        sync_armed = False
                        best_sync_snr = 0.0
                        # Compensate for the 1 step (10ms) we waited past peak
                        samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES
                elif sync_armed:
                    # Pattern broke right after peaking -> lock immediately
                    print("\n[SYNC LOCKED] Receiving packet...")
                    state = "RECEIVE_DATA"
                    nibbles = []
                    expected_nibbles = None
                    sync_armed = False
                    best_sync_snr = 0.0
                    samples_until_next_read = SYMBOL_SAMPLES - STEP_SAMPLES        
            elif state == "RECEIVE_DATA":
                samples_until_next_read -= STEP_SAMPLES
                if samples_until_next_read <= 0:
                    # Read the centered 60ms window (10ms before end of slot)
                    end_offset = -int(SAMPLE_RATE * 0.010)
                    start_offset = end_offset - WINDOW_SAMPLES
                    window = ring_buffer[start_offset:end_offset]
                    
                    freq, snr, best_data_idx = analyze_window(window)
                    
                    if freq in DATA_FREQS:
                        nibbles.append(best_data_idx)
                    else:
                        nibbles.append(best_data_idx)
                        print(f"  [!] Low/Unexpected SNR ({snr:.1f}, peak={freq}) on nibble {len(nibbles)}, guessed {best_data_idx:04b}")

                    # First 2 nibbles = 1 byte payload length
                    if len(nibbles) == 2:
                        payload_len = decode_nibbles_to_bytes(nibbles)[0]
                        expected_nibbles = (1 + payload_len + 4) * 2
                        print(f"  Header decoded: expecting {payload_len} bytes...")

                    # Check if full packet received
                    if expected_nibbles is not None and len(nibbles) >= expected_nibbles:
                        raw_bytes = decode_nibbles_to_bytes(nibbles)
                        payload = raw_bytes[1:-4]
                        received_crc = struct.unpack('>I', raw_bytes[-4:])[0]
                        calculated_crc = zlib.crc32(payload) & 0xFFFFFFFF
                        
                        if received_crc == calculated_crc:
                            print(f"[SUCCESS] CRC Valid! Message:\n{payload.decode('utf-8', errors='replace')}\n")
                        else:
                            print(f"[ERROR] CRC Mismatch! (Got {received_crc:#x}, expected {calculated_crc:#x})")
                            print(f"  Raw decoded bytes: {payload}\n")
                            
                        state = "SEARCH_SYNC"
                        # Clear ring buffer so old tones don't re-trigger sync
                        ring_buffer.fill(0)
                        
                    samples_until_next_read += SYMBOL_SAMPLES

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nReceiver stopped.")