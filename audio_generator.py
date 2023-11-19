import pyaudio
import numpy as np
from scipy import signal

p = pyaudio.PyAudio()

volume = 1  # range [0.0, 1.0]
fs = 44100  # sampling rate, Hz, must be integer
duration = 1  # in seconds, may be float
f = 25.0  # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
x_data = np.arange(fs * duration)
dx = np.full_like(x_data, 1)
multi = 2
freq_multi = np.arange(1, multi, 1 / len(x_data))
x_data = freq_multi.cumsum()
fade_in_out = 0.3 * np.sin(np.arange(0, np.pi, np.pi / len(x_data)))

dx = np.full_like(x_data, 0.1)
first_wave = np.sin(2 * np.pi * x_data * f / fs) * fade_in_out
wave = first_wave


samples = (wave * volume).astype(np.float32).tobytes()
channels = 1

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
                channels=channels,
                rate=fs,
                output=True)

samples = (wave * volume).astype(np.float32).tobytes()
stream.write(samples)

# play. May repeat with different volume values (if done interactively)
# stream.write(samples)
freq = ""

wave = np.zeros(duration * fs)
while True:
    freq = input("next freq")
    if freq == "q": break
    if freq.split(sep=" ")[0] == "c":
        for i, f in enumerate(freq.split(sep=" ")[1:]):
            f = float(f) / channels
            wave += np.sin(2 * np.pi * np.arange(fs * duration) * f / fs) * fade_in_out
    elif freq.split(sep=" ")[0] == "u":
        multi = float(freq.split(sep=" ")[1])
        for i, f in enumerate(freq.split(sep=" ")[2:]):
            f = float(f) / channels
            x_in = np.linspace(1, multi, duration * fs).cumsum()
            data = 2 * np.pi * x_in * int(f) / fs
            wave += np.sin(data) * fade_in_out
    elif freq.split(sep=" ")[0] == "d":
        for i, f in enumerate(freq.split(sep=" ")[2:]):
            f = float(f) / channels
            multi = float(freq.split(sep=" ")[1])
            x_in = np.linspace(1, 1 / multi, duration * fs).cumsum()
            data = 2 * np.pi * x_in * int(f) / fs
            wave += np.sin(data) * fade_in_out
    elif freq.split(sep=" ")[0] == "dur":
        duration = int(freq.split(sep=" ")[1])
        wave = np.zeros(duration * fs)
        fade_in_out = 1 * np.sin(np.arange(0, np.pi, np.pi / len(wave))) * 10
        fade_in_out[fade_in_out > volume] = volume

        continue
    else:
        wave = np.zeros(duration * fs)
        continue

    wave = wave / max(wave)
    samples = (wave * volume).astype(np.float32).tobytes()
    stream.write(samples)

stream.stop_stream()

stream.close()

p.terminate()
