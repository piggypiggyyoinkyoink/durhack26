import numpy as np 
import soundfile as sf

x = np.array([10.0,-10.0,10.0,-10.0]) # x.dtype is 'float64'
sf.write("x.wav", x, 2) # a wav at sampling rate 1 Hz