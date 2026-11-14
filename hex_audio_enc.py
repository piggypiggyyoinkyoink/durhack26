import numpy as np
import sounddevice as sd

def read_file(file_path):
    file = open(file_path, 'rb')
    data = file.read()
    file.close()
    return data

def get_bitstring(data):
    return "".join(f"{byte:08b}" for byte in data)


def transmit_bitstring(bitstring, frequencies, duration=0.1, sample_rate=44100, amplitude=0.5):
    samples_per_bit = int(sample_rate * duration)
    t = np.linspace(0, duration, samples_per_bit, endpoint=False)
    
    fade_len = int(sample_rate * 0.005)  # 5 ms fade in/out
    envolope = np.ones(samples_per_bit)
    envolope[:fade_len] = np.linspace(0, 1, fade_len)  # Fade in
    envolope[-fade_len:] = np.linspace(1, 0, fade_len)

    bitstring = "0"*(4 - (len(bitstring) % 4)) + bitstring  # Pad with leading zeros to make length a multiple of 4
    waves = []
    for i in range(0, len(bitstring), 4):  # Process 4 bits at a time
        frequency = frequencies[int(bitstring[i:i+4], 2)]  # turns 4 bits into integer to index into frequencies
        wave = amplitude * np.sin(2 * np.pi * frequency * t) * envolope
        waves.append(wave)

    full_signal = np.concatenate(waves)
    sd.play(full_signal, samplerate=sample_rate)
    sd.wait()

file_bytes = read_file("a.txt")
bitstring = get_bitstring(file_bytes)
print(bitstring)
frequencies = np.arange(11000, 7000, -500)  # Create an array of frequencies from 12000 to 8000 with a step of -250
transmit_bitstring(bitstring, frequencies=frequencies, duration=0.1, amplitude=1)