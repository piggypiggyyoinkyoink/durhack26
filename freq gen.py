import numpy as np
import sounddevice as sd

# Audio parameters
frequency = 6000.0    # Frequency in Hz (e.g., 440 Hz = A4)
duration = 6.0       # Duration in seconds
sample_rate = 44100  # Samples per second (CD quality)
amplitude = 0.5      # Volume (0.0 to 1.0)

# Generate time array and sine wave
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
wave = amplitude * np.sin(2 * np.pi * frequency * t)

# Play the sound and wait until it finishes
sd.play(wave, samplerate=sample_rate)
sd.wait()