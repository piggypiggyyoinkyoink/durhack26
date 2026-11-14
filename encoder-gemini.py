import struct
import zlib
import numpy as np
import sounddevice as sd

PREAMBLE_BITS = "101010101010"   # 12-bit clock training
SYNC_BYTE = b"\x7e"              # 01111110 frame delimiter


def hamming74_encode_nibble(nibble: int) -> str:
    """Encode a 4-bit integer (0..15) into a 7-bit Hamming(7,4) codeword string."""
    d1 = (nibble >> 3) & 1
    d2 = (nibble >> 2) & 1
    d3 = (nibble >> 1) & 1
    d4 = nibble & 1

    # Standard even-parity Hamming(7,4) positions: [p1, p2, d1, p3, d2, d3, d4]
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4

    return f"{p1}{p2}{d1}{p3}{d2}{d3}{d4}"


def hamming74_encode_bytes(data: bytes) -> str:
    """Encode each byte into two 7-bit Hamming(7,4) codewords (14 bits/byte)."""
    bits = []
    for byte in data:
        high = (byte >> 4) & 0x0F
        low = byte & 0x0F
        bits.append(hamming74_encode_nibble(high))
        bits.append(hamming74_encode_nibble(low))
    return "".join(bits)


def build_frame(payload: bytes) -> str:
    """
    Frame format:
      Raw:     [PREAMBLE (12b)] [SYNC 0x7E (8b)]
      Encoded: Hamming(7,4) over [LEN (2B)] [PAYLOAD] [CRC32 (4B)]
    """
    header = struct.pack(">H", len(payload))
    crc = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    protected_bytes = header + payload + crc

    sync_bits = "".join(f"{b:08b}" for b in SYNC_BYTE)
    encoded_body = hamming74_encode_bytes(protected_bytes)
    return PREAMBLE_BITS + sync_bits + encoded_body


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
    print(f"Transmitting {len(data)} bytes ({len(bits)} bits with Hamming(7,4), ~{len(bits)*0.06:.1f}s)...")
    transmit_bitstring(bits)
    print("Done.")