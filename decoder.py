"""
Continuous FSK decoder - the counterpart of the encoder script.

Encoder format (what this decodes):
  - Each bit is a 0.1 s sine tone: 12000 Hz = '0', 13000 Hz = '1'
  - Each tone has a 5 ms fade-in / fade-out (so a small dip at every bit boundary)
  - Bits are sent back to back, 8 bits per byte, MSB first, no header or footer

Behaviour:
  - Listens to the microphone forever
  - A transmission starts when energy appears at the 12 kHz / 13 kHz tones
  - A transmission ends after a stretch of silence (no tones)
  - The recording is then demodulated into bits -> bytes and saved to a file
  - Listening continues for the next transmission

Usage:
  python decoder.py                  # default input device, saves to ./received
  python decoder.py --list-devices
  python decoder.py --device 2 --out-dir captures
"""
import argparse
from html import parser
import os
import queue
import time
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd


# ----------------------------------------------------------------------------
# Demodulation (works on a complete recording of one transmission)
# ----------------------------------------------------------------------------
def envelope(x, sample_rate, ms=2.0):
    n = max(1, int(sample_rate * ms / 1000))
    return np.convolve(np.abs(x), np.ones(n) / n, mode="same")


def tone_power(segment, freq, sample_rate):
    n = np.arange(len(segment))
    return abs(np.dot(segment, np.exp(-2j * np.pi * freq * n / sample_rate)))


def decode_samples(audio, sample_rate, f0, f1, bit_duration):
    """Turn the audio of one transmission into a bitstring ('0'/'1' characters)."""
    spb = int(round(sample_rate * bit_duration))  # samples per bit
    env = envelope(audio, sample_rate)
    peak = np.percentile(env, 95)
    if peak <= 0:
        return ""

    above = np.flatnonzero(env > 0.2 * peak)
    if len(above) == 0:
        return ""
    onset, end = above[0], above[-1] + 1

    # The envelope crosses 20% a little after the true start of the first bit
    p = float(max(0, onset - int(0.001 * sample_rate)))
    n_bits = int(round((end - p) / spb))
    if n_bits < 1:
        return ""

    search = int(0.003 * sample_rate)  # +/- 3 ms boundary search window
    margin = int(0.2 * spb)            # ignore the edges of each bit when measuring
    bits = []

    for k in range(n_bits):
        nxt = p + spb
        # Track clock drift: every bit boundary has an amplitude dip, find it
        if k < n_bits - 1:
            lo, hi = int(nxt) - search, int(nxt) + search
            if lo >= 0 and hi <= len(env):
                found = lo + int(np.argmin(env[lo:hi]))
                nxt += 0.5 * (found - nxt)

        a, b = int(p) + margin, int(nxt) - margin
        seg = audio[max(a, 0):min(b, len(audio))]
        if len(seg) < 10:
            break
        seg = seg * np.hanning(len(seg))
        bits.append("1" if tone_power(seg, f1, sample_rate) > tone_power(seg, f0, sample_rate) else "0")
        p = nxt

    return "".join(bits)


def bits_to_bytes(bitstring):
    bitstring = bitstring[: len(bitstring) // 8 * 8]  # drop any incomplete trailing byte
    return bytes(int(bitstring[i:i + 8], 2) for i in range(0, len(bitstring), 8))


# ----------------------------------------------------------------------------
# Live listener
# ----------------------------------------------------------------------------
class BlockAnalyzer:
    """Decides, for each ~10 ms block, whether the FSK tones are present."""

    def __init__(self, sample_rate, f0, f1, block_size, min_rms, min_tone_fraction):
        self.block_size = block_size
        self.min_rms = min_rms
        self.min_tone_fraction = min_tone_fraction
        self.bins = []
        for f in (f0, f1):
            k = int(round(f * block_size / sample_rate))
            self.bins.extend([k - 1, k, k + 1])
        self.bins = np.array(self.bins)

    def is_active(self, block):
        rms = float(np.sqrt(np.mean(block ** 2)))
        if rms < self.min_rms:
            return False
        spectrum = np.abs(np.fft.rfft(block)) ** 2
        total = spectrum[1:].sum()
        if total <= 0:
            return False
        return spectrum[self.bins].sum() / total >= self.min_tone_fraction


def save_transmission(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    name = datetime.now().strftime("received_%Y%m%d_%H%M%S.bin")
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


def select_loudest_device(sample_rate=44100, duration=3.0):
    """
    Open every input device at once for `duration` seconds and return the ID of the
    one that picked up the highest peak amplitude (or None if none could be opened).

    Run this while the transmitter is playing (or while there is some sound) so the
    device that actually hears the signal wins.
    """
    peaks = {}      # device id -> highest absolute sample seen
    streams = {}    # device id -> open stream

    def make_callback(dev_id):
        def callback(indata, frames, time_info, status):
            peaks[dev_id] = max(peaks[dev_id], float(np.max(np.abs(indata))))
        return callback

    for dev_id, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        peaks[dev_id] = 0.0
        try:
            stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                    device=dev_id, callback=make_callback(dev_id))
            stream.start()
            streams[dev_id] = stream
        except Exception as e:
            print(f"  skipping device {dev_id} ({info['name']}): {e}")
            del peaks[dev_id]

    if not streams:
        return None

    print(f"Measuring {len(streams)} input device(s) for {duration:.0f} s...")
    time.sleep(duration)

    for stream in streams.values():
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    names = {i: d["name"] for i, d in enumerate(sd.query_devices())}
    for dev_id, peak in sorted(peaks.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  device {dev_id:>3}  peak {peak:.4f}  {names[dev_id]}")

    best = max(peaks, key=peaks.get)
    print(f"Selected device {best}: {names[best]}")
    return best


def listen(args):
    sr = args.sample_rate
    block_size = int(sr * 0.01)  # 10 ms blocks
    analyzer = BlockAnalyzer(sr, args.f0, args.f1, block_size, args.min_rms, args.min_tone_fraction)

    start_blocks = 3                                   # consecutive active blocks to trigger
    end_blocks = int(args.silence_timeout / 0.01)      # consecutive quiet blocks to finish
    preroll = deque(maxlen=start_blocks + 2)

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio warning] {status}")
        q.put(indata[:, 0].copy())

    state = "idle"
    active_run = 0
    quiet_run = 0
    recording = []

    print(f"Listening at {sr} Hz  (0 = {args.f0:.0f} Hz, 1 = {args.f1:.0f} Hz, "
          f"{args.bit_duration * 1000:.0f} ms/bit). Ctrl+C to stop.")

    with sd.InputStream(samplerate=sr, channels=1, blocksize=block_size,
                        dtype="float32", device=args.device, callback=callback):
        try:
            while True:
                block = q.get()
                if len(block) != block_size:
                    continue
                active = analyzer.is_active(block)

                rms = float(np.sqrt(np.mean(block ** 2)))
                spectrum = np.abs(np.fft.rfft(block)) ** 2
                total = spectrum[1:].sum()
                tone_frac = spectrum[analyzer.bins].sum() / total if total > 0 else 0
                if rms > 0.0005:  # print when there's any noticeable sound
                    print(f"DEBUG: RMS={rms:.5f}, ToneFrac={tone_frac:.2f}, Active={active}")

                if state == "idle":
                    preroll.append(block)
                    active_run = active_run + 1 if active else 0
                    if active_run >= start_blocks:
                        state = "receiving"
                        recording = list(preroll)
                        quiet_run = 0
                        print("Transmission detected - receiving...")
                else:
                    recording.append(block)
                    quiet_run = 0 if active else quiet_run + 1
                    if quiet_run >= end_blocks:
                        audio = np.concatenate(recording).astype(np.float64)
                        recording = []
                        state, active_run = "idle", 0
                        preroll.clear()

                        bitstring = decode_samples(audio, sr, args.f0, args.f1, args.bit_duration)
                        data = bits_to_bytes(bitstring)
                        if not data:
                            print("Transmission ended but nothing could be decoded; ignoring.")
                            continue
                        path = save_transmission(data, args.out_dir)
                        print(f"Transmission ended: {len(bitstring)} bits -> {len(data)} bytes, saved to {path}")
                        try:
                            text = data.decode("utf-8")
                            preview = text[:200] + ("..." if len(text) > 200 else "")
                            print(f"Content (text): {preview!r}")
                        except UnicodeDecodeError:
                            pass
                        print("Listening...")
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Continuous FSK audio decoder")
    parser.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--device", default="auto",
                        help="input device index or name, or 'auto' to pick the loudest device")
    parser.add_argument("--auto-seconds", type=float, default=3.0,
                        help="how long to measure each device when --device auto")
    parser.add_argument("--out-dir", default="received", help="folder for decoded files")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--f0", type=float, default=12000.0, help="frequency for bit 0")
    parser.add_argument("--f1", type=float, default=13000.0, help="frequency for bit 1")
    parser.add_argument("--bit-duration", type=float, default=0.1, help="seconds per bit")
    parser.add_argument("--min-rms", type=float, default=0.001,
                        help="minimum block loudness to count as signal")
    parser.add_argument("--min-tone-fraction", type=float, default=0.15,
                        help="fraction of block power that must sit at the two tones")
    parser.add_argument("--silence-timeout", type=float, default=0.5,
                        help="seconds without tones that ends a transmission")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    if isinstance(args.device, str) and args.device.lower() == "auto":
        args.device = select_loudest_device(args.sample_rate, args.auto_seconds)
        if args.device is None:
            print("No input device could be opened.")
            return
    elif isinstance(args.device, str) and args.device.isdigit():
        args.device = int(args.device)

    listen(args)


if __name__ == "__main__":
    main()