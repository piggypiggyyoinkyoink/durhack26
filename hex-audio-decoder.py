
import numpy as np
import sounddevice as sd
import time

# =====================================
# CONFIGURATION
# =====================================

SAMPLE_RATE = 44100

TRANSMISSION_INTERVAL = 0.100  # 100 ms
WINDOW_DURATION = 0.060        # 60 ms

START_HZ = 8000
FREQUENCY_SPACING = 250

SILENCE_THRESHOLD = 0.01

FREQUENCIES = [
    START_HZ + i * FREQUENCY_SPACING
    for i in range(16)
]

# =====================================
# FFT SETUP
# =====================================

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)

FFT_SIZE = 8192

HANN = np.hanning(WINDOW_SIZE)

FFT_FREQUENCIES = np.fft.rfftfreq(
    FFT_SIZE,
    1 / SAMPLE_RATE
)

FREQUENCY_BINS = [
    np.argmin(np.abs(FFT_FREQUENCIES - f))
    for f in FREQUENCIES
]


# =====================================
# DECODING
# =====================================

def decode(samples):

    samples = samples - np.mean(samples)

    volume = np.sqrt(np.mean(samples ** 2))

    if volume < SILENCE_THRESHOLD:
        return None

    samples = samples * HANN

    spectrum = np.abs(
        np.fft.rfft(samples, n=FFT_SIZE)
    )

    strengths = spectrum[FREQUENCY_BINS]

    index = np.argmax(strengths)

    sorted_strengths = np.sort(strengths)

    strongest = sorted_strengths[-1]
    second = sorted_strengths[-2]

    if strongest < 1e-6:
        return None

    if strongest < second * 1.5:
        return None

    return format(index, "04b")


# =====================================
# RECEIVER
# =====================================

def main():

    print("Starting timing-based receiver...")
    print(f"Interval: {TRANSMISSION_INTERVAL * 1000} ms")
    print("Press CTRL+C to stop.\n")

    # Circular audio buffer
    buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)

    # Timing
    next_sample_time = time.perf_counter()

    decoded_count = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=WINDOW_SIZE
    ) as stream:

        while True:

            # Wait for the next transmission slot
            now = time.perf_counter()

            if now < next_sample_time:
                time.sleep(next_sample_time - now)

            # Capture audio
            audio, overflowed = stream.read(WINDOW_SIZE)

            samples = audio[:, 0]

            # Decode
            nibble = decode(samples)

            if nibble is not None:

                frequency = FREQUENCIES[
                    int(nibble, 2)
                ]

                print(
                    f"Slot {decoded_count:04d}: "
                    f"{nibble} ({frequency} Hz)"
                )

            else:
                print(
                    f"Slot {decoded_count:04d}: "
                    f"No signal / invalid signal"
                )

            decoded_count += 1

            # Schedule next slot
            next_sample_time += TRANSMISSION_INTERVAL

            # Recover if processing fell behind
            if time.perf_counter() > next_sample_time:
                next_sample_time = (
                    time.perf_counter() + TRANSMISSION_INTERVAL
                )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nReceiver stopped.")
