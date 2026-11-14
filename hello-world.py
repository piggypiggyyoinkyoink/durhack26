print("Hello World")

import numpy as np
import sounddevice as sd

frequency = 440.0  # Frequency in Hz (440 Hz = Musical note A4)
duration = 2.0     # Duration in seconds
sample_rate = 44100  # Standard audio sample rate

t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # 0.5 scales volume to 50%
sd.play(wave, sample_rate)
sd.wait()
