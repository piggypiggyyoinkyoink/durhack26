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
    for bit in bitstring:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)  
        if bit == '1':
            frequency = frequency_1
        else:
            frequency = frequency_0
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        sd.play(wave, samplerate=sample_rate)
        sd.wait()  # Wait for the sound to finish playing


file_bytes = read_file("a.txt")
bitstring = get_bitstring(file_bytes)
print(bitstring)
transmit_bitstring(bitstring, frequency_0=10000.0, frequency_1=12000.0, duration=0.25)