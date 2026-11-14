import struct
import zlib
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
SYMBOL_DURATION = 0.100  # 100 ms total per symbol
TONE_DURATION = 0.080    # 80 ms tone + 20 ms guard silence (prevents echo overlap)

# 16 Data Frequencies: 8000 Hz to 11750 Hz (250 Hz spacing)
DATA_FREQS = np.array([8000 + i * 250 for i in range(16)])

# 2 Dedicated Preamble Frequencies (outside data band)
SYNC_FREQ_A = 12250
SYNC_FREQ_B = 12750

def generate_tone(freq, duration, total_slot_duration, sample_rate=SAMPLE_RATE, amplitude=0.8):
    tone_samples = int(sample_rate * duration)
    total_samples = int(sample_rate * total_slot_duration)
    
    t = np.linspace(0, duration, tone_samples, endpoint=False)
    fade_len = int(sample_rate * 0.005)  # 5 ms fade to eliminate speaker clicks
    
    envelope = np.ones(tone_samples)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    
    wave = amplitude * np.sin(2 * np.pi * freq * t) * envelope
    
    # Pad with guard silence to match total_slot_duration
    guard_samples = total_samples - tone_samples
    if guard_samples > 0:
        wave = np.pad(wave, (0, guard_samples), mode='constant')
        
    return wave

def build_packet(file_path):
    with open(file_path, 'rb') as f:
        payload = f.read()
        
    if len(payload) > 255:
        raise ValueError("Payload too large for 1-byte length header (max 255 bytes).")
        
    # Packet structure: [Length (1B)] + [Payload] + [CRC32 (4B)]
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    packet = struct.pack('B', len(payload)) + payload + struct.pack('>I', crc)
    return packet

def transmit_file(file_path):
    packet = build_packet(file_path)
    bitstring = "".join(f"{byte:08b}" for byte in packet)
    
    waves = []
    
    # 1. Preamble: 6 alternating sync tones for clock & frame lock
    preamble_freqs = [SYNC_FREQ_A, SYNC_FREQ_B] * 3
    for freq in preamble_freqs:
        waves.append(generate_tone(freq, TONE_DURATION, SYMBOL_DURATION))
        
    # 2. Data nibbles (4 bits per symbol)
    for i in range(0, len(bitstring), 4):
        nibble_val = int(bitstring[i:i+4], 2)
        freq = DATA_FREQS[nibble_val]
        waves.append(generate_tone(freq, TONE_DURATION, SYMBOL_DURATION))
    # 3. Tail silence so sounddevice doesn't truncate the final CRC symbol
    waves.append(np.zeros(int(SAMPLE_RATE * 0.150), dtype=np.float32))
    full_signal = np.concatenate(waves).astype(np.float32)
    
    print(f"Transmitting {len(packet)} bytes ({len(waves)} symbols, {len(full_signal)/SAMPLE_RATE:.2f}s)...")
    sd.play(full_signal, samplerate=SAMPLE_RATE)
    sd.wait()
    print("Transmission complete.")

if __name__ == "__main__":
    transmit_file("test.txt")