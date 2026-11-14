"""Continuously receive encoded audio and save each transmission separately."""

FREQUENCY_0 = 12000.0
FREQUENCY_1 = 13000.0
PULSE_DURATION = 0.1

import argparse
from pathlib import Path
from queue import Empty, Queue
import sys

import numpy as np
import sounddevice as sd


SILENCE_DURATION = 0.2
MIN_SIGNAL_RMS = 0.005


def save_session(
    chunks: list[np.ndarray],
    output_base: Path,
    session_number: int,
    sample_rate: int,
    pulse_duration: float,
) -> bool:
    """Decode and save one transmission, returning whether a file was written."""
    data = decode(np.concatenate(chunks), sample_rate, pulse_duration)
    if not data:
        return False

    if session_number == 0:
        output_path = output_base
    else:
        output_path = output_base.with_name(
            f"{output_base.stem}_{session_number}{output_base.suffix}"
        )
    output_path.write_bytes(data)
    print(f"Wrote {len(data)} bytes to {output_path}")
    return True


def record_sessions(
    output_base: Path,
    sample_rate: int,
    pulse_duration: float,
    block_size: int,
) -> None:
    """Listen continuously and save each transmission after sustained silence."""
    blocks: Queue[np.ndarray] = Queue()

    def callback(
        indata: np.ndarray,
        _frames: int,
        _time: object,
        status: object,
    ) -> None:
        if status:
            print(f"Audio input warning: {status}", file=sys.stderr)
        blocks.put(indata[:, 0].copy())

    block_duration = block_size / sample_rate
    silence_blocks_required = max(1, int(SILENCE_DURATION / block_duration))
    calibration_blocks_required = max(1, int(1 / block_duration))
    calibration_rms: list[float] = []
    noise_threshold = MIN_SIGNAL_RMS
    session: list[np.ndarray] = []
    silent_blocks = 0
    session_number = 0

    print("Listening continuously. Each transmission is saved after 0.75 seconds of silence.")
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=callback,
        ):
            while True:
                try:
                    block = blocks.get(timeout=0.25)
                except Empty:
                    continue

                rms = float(np.sqrt(np.mean(block * block)))
                if len(calibration_rms) < calibration_blocks_required:
                    calibration_rms.append(rms)
                    if len(calibration_rms) == calibration_blocks_required:
                        noise_threshold = max(
                            MIN_SIGNAL_RMS, float(np.median(calibration_rms)) * 4
                        )

                if rms >= noise_threshold:
                    if not session:
                        print("Transmission detected.")
                    session.append(block)
                    silent_blocks = 0
                elif session:
                    session.append(block)
                    silent_blocks += 1
                    if silent_blocks >= silence_blocks_required:
                        if save_session(
                            session[:-silent_blocks],
                            output_base,
                            session_number,
                            sample_rate,
                            pulse_duration,
                        ):
                            session_number += 1
                        session = []
                        silent_blocks = 0
    except KeyboardInterrupt:
        if session:
            save_session(
                session[:-silent_blocks] if silent_blocks else session,
                output_base,
                session_number,
                sample_rate,
                pulse_duration,
            )


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

    record_sessions(
        Path(args.output),
        args.sample_rate,
        args.pulse_duration,
        args.block_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



