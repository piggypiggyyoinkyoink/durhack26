import struct
import zlib
import numpy as np
import sounddevice as sd

PREAMBLE_BITS = "101010101010"   # 12-bit clock training
SYNC_BYTE = b"\x7e"              # 01111110 frame delimiter


def build_frame(payload: bytes) -> str:
    """Frame format: [PREAMBLE] [SYNC 0x7E] [LEN (2B)] [PAYLOAD] [CRC32 (4B)]"""
    header = struct.pack(">H", len(payload))
    crc = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    frame_bytes = SYNC_BYTE + header + payload + crc
    frame_bits = "".join(f"{b:08b}" for b in frame_bytes)
    return PREAMBLE_BITS + frame_bits


def transmit_bitstring(
    bitstring: str,
    f0: float = 5000.0,
    f1: float = 6500.0,
    duration: float = 0.06,
    sample_rate: int = 44100,
    amplitude: float = 0.8,
):
    spb = int(round(sample_rate * duration))
    t = np.arange(spb) / sample_rate

    # Smooth raised-cosine edges (3 ms) to prevent speaker clicks without deep dips
    fade_len = int(sample_rate * 0.003)
    envelope = np.ones(spb)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len) / fade_len))
    envelope[:fade_len] = ramp
    envelope[-fade_len:] = ramp[::-1]

    # Pre-render the two bit waveforms
    wave_0 = amplitude * np.sin(2 * np.pi * f0 * t) * envelope
    wave_1 = amplitude * np.sin(2 * np.pi * f1 * t) * envelope

    # Pad with 0.15 s of silence before/after so OS audio drivers don't clip the start
    pad = np.zeros(int(sample_rate * 0.15))
    signal = np.concatenate(
        [pad] + [wave_1 if b == "1" else wave_0 for b in bitstring] + [pad]
    )

    sd.play(signal.astype(np.float32), samplerate=sample_rate)
    sd.wait()

if __name__ == "__main__":
    with open("test.txt", "rb") as f:
        data = f.read()

    bits = build_frame(data)
    print(f"Transmitting {len(data)} bytes ({len(bits)} bits, ~{len(bits)*0.06:.1f}s)...")
    transmit_bitstring(bits)
    print("Done.")