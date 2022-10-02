import pyaudio
import numpy as np
from scipy import signal

p = pyaudio.PyAudio()
import threading

volume = 0.8  # range [0.0, 1.0]
fs = 44100  # sampling rate, Hz, must be integer
duration = 200  # in seconds, may be float
f = 10.0  # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
x_data = np.arange(fs * duration)
dx = np.full_like(x_data, 1)
multi = 2
freq_multi = np.arange(1, multi, 1 / len(x_data))
x_data = freq_multi.cumsum()
fade_in_out =1 * np.sin(np.arange(0, np.pi, np.pi / len(x_data)))*10
fade_in_out[fade_in_out>volume]=volume

dx = np.full_like(x_data, 0.1)
first_wave = np.sin(2 * np.pi * x_data * f / fs) * fade_in_out
wave = first_wave
wave += np.roll(first_wave, 5 * fs)

samples = (wave * volume).astype(np.float32).tobytes()

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
                channels=5,
                rate=fs,
                output=True)

# play. May repeat with different volume values (if done interactively)
#stream.write(samples)
freq = "z"


def test():
    wave = np.zeros(duration * fs)
    while True:

        if freq == "q": break
        stream.write(samples)


testThrd = threading.Thread(target=test)
testThrd.start()
ii = 0
while freq != "q":
    ii += 1
    freq = input("next freq")
    if freq == "q": break
    x_in = np.arange(fs * duration)
    f_l = freq.split(sep=" ")
    for i in range(len(f_l)):
        k = freq.split(sep=" ")[0]
        if i == 0:
            if k == "usweep":
                func = np.sin
                multi = int(f_l[1])
                x_in = np.linspace(1, multi, duration * fs).cumsum()
                i += 2
                continue
            elif k == "dsweep":
                func = np.sin
                multi = int(f_l[1])
                x_in = np.linspace(1, 1 / multi, duration * fs).cumsum()
                i += 2
                continue
            elif k == "sawd":
                func = signal.sawtooth
                saw_width = 0
                continue
            elif k == "saw":
                func = signal.sawtooth
                saw_width = 0.5
                continue
            elif k == "z":
                wave = np.zeros(duration * fs)
            else:
                func = np.sin

        if f_l[i].isnumeric():
            f = int(f_l[i])
            data = 2 * np.pi * x_in * int(f) / fs
            wave += func(data) * fade_in_out
        else:
            break
    if max(wave)>1:
        wave = wave / max(wave)
    samples = (wave * volume).astype(np.float32).tobytes()

stream.stop_stream()

stream.close()

p.terminate()
