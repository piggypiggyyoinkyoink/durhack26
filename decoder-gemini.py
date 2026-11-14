import argparse
import os
import queue
import struct
import zlib
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd

PREAMBLE_BITS = "101010101010"
SYNC_BITS = "01111110"
HEADER_BITS = PREAMBLE_BITS + SYNC_BITS  # 20 known bits for exact sample lock


def hamming74_decode_nibble(codeword: str) -> tuple[int, bool]:
    """
    Decode a 7-bit Hamming(7,4) codeword string into a 4-bit nibble.
    Returns (nibble, corrected) where corrected is True if a 1-bit error was fixed.
    """
    b = [int(c) for c in codeword]
    # Positions (1-indexed): 1:p1, 2:p2, 3:d1, 4:p3, 5:d2, 6:d3, 7:d4
    p1, p2, d1, p3, d2, d3, d4 = b

    # Compute 3-bit syndrome (s3 s2 s1) pointing to 1-based error index
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = (s3 << 2) | (s2 << 1) | s1

    corrected = False
    if syndrome != 0:
        b[syndrome - 1] ^= 1
        corrected = True
        _, _, d1, _, d2, d3, d4 = b

    nibble = (d1 << 3) | (d2 << 2) | (d3 << 1) | d4
    return nibble, corrected


def hamming74_decode_bytes(encoded_bits: str) -> tuple[bytes, int]:
    """
    Decode a multiple-of-14 bitstring into raw bytes.
    Returns (decoded_bytes, total_corrected_bits).
    """
    out = bytearray()
    corrected_count = 0
    for i in range(0, len(encoded_bits), 14):
        high, c1 = hamming74_decode_nibble(encoded_bits[i : i + 7])
        low, c2 = hamming74_decode_nibble(encoded_bits[i + 7 : i + 14])
        corrected_count += int(c1) + int(c2)
        out.append((high << 4) | low)
    return bytes(out), corrected_count


def tone_power(segment: np.ndarray, freq: float, sample_rate: int) -> float:
    n = np.arange(len(segment))
    return float(np.abs(np.dot(segment, np.exp(-2j * np.pi * freq * n / sample_rate))))


def find_frame_start(audio: np.ndarray, sample_rate: int, f0: float, f1: float, bit_duration: float) -> int | None:
    """
    Slide across the start of the recording at 1 ms resolution to find the exact
    sample index where PREAMBLE + SYNC_BITS achieves maximum correlation.
    """
    spb = int(round(sample_rate * bit_duration))
    step = max(1, int(sample_rate * 0.001))  # 1 ms search step
    margin = int(0.2 * spb)
    win = np.hanning(spb - 2 * margin)

    max_search = min(len(audio) - len(HEADER_BITS) * spb, int(sample_rate * 1.5))
    if max_search <= 0:
        return None

    best_score = -np.inf
    best_offset = None

    for start in range(0, max_search, step):
        score = 0.0
        for i, bit in enumerate(HEADER_BITS):
            a = start + i * spb + margin
            b = a + len(win)
            seg = audio[a:b] * win
            p0 = tone_power(seg, f0, sample_rate)
            p1 = tone_power(seg, f1, sample_rate)
            denom = p0 + p1 + 1e-12
            contrast = (p1 - p0) / denom
            score += contrast if bit == "1" else -contrast

        if score > best_score:
            best_score = score
            best_offset = start

    if best_offset is None or (best_score / len(HEADER_BITS)) < 0.35:
        return None
    return best_offset


def read_bits(audio: np.ndarray, start_sample: int, n_bits: int, sample_rate: int, f0: float, f1: float, bit_duration: float) -> str:
    spb = sample_rate * bit_duration
    margin = int(0.22 * spb)
    bits = []

    for k in range(n_bits):
        a = int(round(start_sample + k * spb)) + margin
        b = int(round(start_sample + (k + 1) * spb)) - margin
        if b > len(audio):
            break
        seg = audio[a:b] * np.hanning(b - a)
        p1 = tone_power(seg, f1, sample_rate)
        p0 = tone_power(seg, f0, sample_rate)
        bits.append("1" if p1 > p0 else "0")

    return "".join(bits)


def decode_frame(audio: np.ndarray, sample_rate: int, f0: float, f1: float, bit_duration: float) -> tuple[bytes | None, int]:
    header_start = find_frame_start(audio, sample_rate, f0, f1, bit_duration)
    if header_start is None:
        print("[decode] Could not lock onto preamble + sync word.")
        return None, 0

    spb = sample_rate * bit_duration
    len_start = int(round(header_start + len(HEADER_BITS) * spb))

    # Read 28 encoded bits -> 2-byte payload length (14 bits per byte)
    len_encoded_bits = read_bits(audio, len_start, 28, sample_rate, f0, f1, bit_duration)
    if len(len_encoded_bits) < 28:
        print("[decode] Truncated length header.")
        return None, 0

    len_bytes, len_fixes = hamming74_decode_bytes(len_encoded_bits)
    payload_len = struct.unpack(">H", len_bytes)[0]

    if payload_len == 0 or payload_len > 4096:
        print(f"[decode] Implausible payload length: {payload_len}")
        return None, 0

    # Read (payload_len + 4 bytes CRC) * 14 encoded bits/byte
    body_start = int(round(len_start + 28 * spb))
    total_body_bits = (payload_len + 4) * 14
    body_encoded_bits = read_bits(audio, body_start, total_body_bits, sample_rate, f0, f1, bit_duration)
    if len(body_encoded_bits) < total_body_bits:
        print(f"[decode] Incomplete frame: got {len(body_encoded_bits)}/{total_body_bits} bits.")
        return None, 0

    raw, body_fixes = hamming74_decode_bytes(body_encoded_bits)
    total_fixes = len_fixes + body_fixes

    payload, crc_bytes = raw[:-4], raw[-4:]
    expected_crc = struct.unpack(">I", crc_bytes)[0]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF

    if actual_crc != expected_crc:
        print(f"[decode] CRC mismatch after {total_fixes} FEC fix(es)! Expected {expected_crc:#010x}, got {actual_crc:#010x}")
        return None, total_fixes

    return payload, total_fixes


class BlockAnalyzer:
    def __init__(self, sample_rate, f0, f1, block_size, min_rms, min_tone_fraction):
        self.min_rms = min_rms
        self.min_tone_fraction = min_tone_fraction
        bins = []
        for f in (f0, f1):
            k = int(round(f * block_size / sample_rate))
            bins.extend([k - 1, k, k + 1])
        self.bins = np.array(bins)

    def is_active(self, block):
        if float(np.sqrt(np.mean(block**2))) < self.min_rms:
            return False
        spectrum = np.abs(np.fft.rfft(block)) ** 2
        total = spectrum[1:].sum()
        return total > 0 and (spectrum[self.bins].sum() / total >= self.min_tone_fraction)


def save_transmission(data: bytes, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, datetime.now().strftime("received_%Y%m%d_%H%M%S.bin"))
    with open(path, "wb") as f:
        f.write(data)
    return path


def listen(args):
    sr = args.sample_rate
    block_size = int(sr * 0.01)
    analyzer = BlockAnalyzer(sr, args.f0, args.f1, block_size, args.min_rms, args.min_tone_fraction)

    start_blocks = 3
    end_blocks = int(args.silence_timeout / 0.01)
    preroll = deque(maxlen=25)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    state = "idle"
    active_run = 0
    quiet_run = 0
    recording = []

    print(f"Listening at {sr} Hz (0={args.f0:.0f} Hz, 1={args.f1:.0f} Hz, {args.bit_duration*1000:.0f} ms/bit, Hamming(7,4) FEC)...")

    with sd.InputStream(samplerate=sr, channels=1, blocksize=block_size, dtype="float32", device=args.device, callback=callback):
        try:
            while True:
                block = q.get()
                if len(block) != block_size:
                    continue
                active = analyzer.is_active(block)

                if state == "idle":
                    preroll.append(block)
                    active_run = active_run + 1 if active else 0
                    if active_run >= start_blocks:
                        state = "receiving"
                        recording = list(preroll)
                        quiet_run = 0
                        print("Signal detected — capturing...")
                else:
                    recording.append(block)
                    quiet_run = 0 if active else quiet_run + 1
                    if quiet_run >= end_blocks:
                        audio = np.concatenate(recording).astype(np.float64)
                        recording = []
                        state, active_run = "idle", 0
                        preroll.clear()

                        data, fixes = decode_frame(audio, sr, args.f0, args.f1, args.bit_duration)
                        if data is not None:
                            path = save_transmission(data, args.out_dir)
                            fix_note = f", corrected {fixes} bit error(s)" if fixes else ", 0 bit errors"
                            print(f"Verified {len(data)} bytes (CRC OK{fix_note}) -> {path}")
                            try:
                                print(f"Text: {data.decode('utf-8')!r}")
                            except UnicodeDecodeError:
                                pass
                        print("Listening...")
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Framed FSK audio decoder with Hamming(7,4) FEC")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="received")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--f0", type=float, default=5000.0)
    parser.add_argument("--f1", type=float, default=6500.0)
    parser.add_argument("--bit-duration", type=float, default=0.06)
    parser.add_argument("--min-rms", type=float, default=0.004)
    parser.add_argument("--min-tone-fraction", type=float, default=0.35)
    parser.add_argument("--silence-timeout", type=float, default=0.4)
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    if isinstance(args.device, str) and args.device.isdigit():
        args.device = int(args.device)
    listen(args)


if __name__ == "__main__":
    main()