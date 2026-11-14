import numpy as np
import sounddevice as sd

def read_file(file_path):
    file = open(file_path, 'rb')
    data = file.read()
    file.close()
    return data

def get_bitstring(data):
    return "".join(f"{byte:08b}" for byte in data)


def transmit_bitstring(bitstring, frequency_0, frequency_1, duration=0.1, sample_rate=44100, amplitude=0.5):
    samples_per_bit = int(sample_rate * duration)
    t = np.linspace(0, duration, samples_per_bit, endpoint=False)
    
    fade_len = int(sample_rate * 0.005)  # 5 ms fade in/out
    envolope = np.ones(samples_per_bit)
    envolope[:fade_len] = np.linspace(0, 1, fade_len)  # Fade in
    envolope[-fade_len:] = np.linspace(1, 0, fade_len)

    waves = []
    for bit in bitstring:
        frequency = frequency_1 if bit == '1' else frequency_0
        wave = amplitude * np.sin(2 * np.pi * frequency * t) * envolope
        waves.append(wave)

    full_signal = np.concatenate(waves)
    sd.play(full_signal, samplerate=sample_rate)
    sd.wait()

file_bytes = read_file("a.txt")
bitstring = get_bitstring(file_bytes)
print(bitstring)
transmit_bitstring(bitstring, frequency_0=3000.0, frequency_1=4000.0, duration=0.25, amplitude=0.1)