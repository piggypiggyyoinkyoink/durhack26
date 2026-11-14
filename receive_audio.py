FREQUENCY_0 = 3000.0
FREQUENCY_1 = 4000.0
PULSE_DURATION = 0.25

"""Record the tones produced by binary_audio_enc.py and reconstruct the file."""

import argparse
import sys
from threading import Lock

import numpy as np
import sounddevice as sd


def record_until_stopped(sample_rate: int, block_size: int) -> np.ndarray:
    """Record microphone input until the user presses Ctrl+C."""
    chunks: list[np.ndarray] = []
    lock = Lock()

    def callback(indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
        with lock:
            chunks.append(indata[:, 0].copy())

    print("Listening... start binary_audio_enc.py, then press Ctrl+C when it finishes.")
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=callback,
        ):
            while True:
                sd.sleep(250)
    except KeyboardInterrupt:
        pass

    with lock:
        if not chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(chunks)


def find_signal_start(signal: np.ndarray, sample_rate: int) -> int:
    """Return the first sample in a sustained signal, ignoring leading silence."""
    window = max(1, int(sample_rate * 0.01))
    energy = np.convolve(signal * signal, np.ones(window) / window, mode="valid")
    threshold = max(1e-7, float(energy.max()) * 0.05)
    active = np.flatnonzero(energy > threshold)
    return int(active[0]) if active.size else len(signal)


def classify_pulse(pulse: np.ndarray, sample_rate: int) -> str:
    """Classify a pulse by comparing its FFT energy at the two carrier frequencies."""
    window = np.hanning(len(pulse))
    spectrum = np.abs(np.fft.rfft(pulse * window))
    frequencies = np.fft.rfftfreq(len(pulse), 1 / sample_rate)

    def energy_at(frequency: float) -> float:
        bandwidth = max(20.0, sample_rate / len(pulse) * 2)
        mask = np.abs(frequencies - frequency) <= bandwidth
        return float(spectrum[mask].sum())

    return "1" if energy_at(FREQUENCY_1) > energy_at(FREQUENCY_0) else "0"


def decode(signal: np.ndarray, sample_rate: int, pulse_duration: float) -> bytes:
    """Decode complete pulses from a recorded mono signal."""
    samples_per_pulse = int(sample_rate * pulse_duration)
    start = find_signal_start(signal, sample_rate)
    usable = signal[start:]
    pulse_count = len(usable) // samples_per_pulse
    bits = "".join(
        classify_pulse(
            usable[index * samples_per_pulse : (index + 1) * samples_per_pulse],
            sample_rate,
        )
        for index in range(pulse_count)
    )
    complete_bits = bits[: len(bits) - len(bits) % 8]
    return bytes(
        int(complete_bits[index : index + 8], 2)
        for index in range(0, len(complete_bits), 8)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="path for the reconstructed file")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--pulse-duration", type=float, default=PULSE_DURATION)
    parser.add_argument("--block-size", type=int, default=2048)
    args = parser.parse_args()

    if args.sample_rate <= 0 or args.pulse_duration <= 0 or args.block_size <= 0:
        parser.error("sample rate, pulse duration, and block size must be positive")

    signal = record_until_stopped(args.sample_rate, args.block_size)
    if signal.size == 0:
        print("No audio was recorded.", file=sys.stderr)
        return 1

    data = decode(signal, args.sample_rate, args.pulse_duration)
    if not data:
        print("No complete encoded bytes were detected.", file=sys.stderr)
        return 1

    with open(args.output, "wb") as output_file:
        output_file.write(data)
    print(f"Wrote {len(data)} bytes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



