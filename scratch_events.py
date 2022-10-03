import numpy as np
import librosa
import pandas as pd
RATE=44000
l=2048
sig1= np.sin(np.linspace(0, 20, l))
sig2= np.sin(np.linspace(0, 71.3, l))
sig3= np.sin(np.linspace(0, 95.3, l))

sig=np.hstack((sig1,sig2,sig3, sig1))
df=pd.DataFrame()
df["signal"]=sig



first_chunk=sig[:2*l]
first_fx= librosa.effects.pitch_shift(first_chunk, sr=RATE, n_steps=-12)

df["first_fx"]=0
df.loc[0:2*l-1, "first_fx"]=first_fx

second_chunk=sig[l:3*l]

second_fx= librosa.effects.pitch_shift(second_chunk, sr=RATE, n_steps=-12)

df["second_fx"]=0
df.loc[l:3*l-1, "second_fx"]=second_fx

res=np.correlate(first_fx, second_fx, mode="full")
idx=np.argmax(res)


res=np.correlate(first_fx, second_fx, mode="full")
idx=np.argmax(res)

df["oberlay"]=0
over=first_fx.copy()
over[max(0, idx-2*l+1): min(idx,2*l)] +=second_fx[max(-2*l,-idx):min(2*l, 2*l+(2*l-idx-1))]
df.loc[0:len(over)-1, "oberlay"]=over

import pyaudio
import numpy as np
from scipy import signal

p = pyaudio.PyAudio()
import threading
from pynput import keyboard

fs = 44100  # sampling rate, Hz, must be integer
duration = 1  # in seconds, may be float
f = 250.0  # sine frequency, Hz, may be float
frame_length = int(duration * fs)
volume = np.ones(frame_length) * 0.5

# generate samples, note conversion to float32 array
x_data = np.arange(frame_length)
dx = np.full_like(x_data, 1)
multi = 2
freq_multi = np.arange(1, multi, 1 / frame_length)
x_data = freq_multi.cumsum()
fade_in_out = 1  # * np.sin(np.arange(0, np.pi, np.pi / len(x_data)))*volume

dx = np.full_like(x_data, 0.1)
first_wave = np.sin(2 * np.pi * x_data * f / fs) * fade_in_out
wave = first_wave
wave += np.roll(first_wave, 5 * fs)

samples = (wave * volume).astype(np.float32).tobytes()

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# play. May repeat with different volume values (if done interactively)
stream.write(samples)
freq = "50"

wave = np.zeros(frame_length)
f = 117.2
x_in = [0]
def on_press(key):
    global wave
    global samples
    global x_in
    global f
    global volume
    if volume[-1] == 0:
        volume = np.ones(frame_length) * 0.5
        volume[:frame_length // 10] = volume[:frame_length // 10] * np.linspace(0, 0.5, frame_length // 10)
    volume = np.ones(frame_length) * 0.5

    try:
        f = 100 * 2 ** ((ord(key.char) - 96) / 12)
    except:
        return
    # x_in = np.arange(frame_length) + x_in[-1]+1
    # print(x_in[-1])
    # data = 2 * np.pi * x_in * float(f) / fs
    # wave += np.sin(data) * fade_in_out
    # if max(wave)>1:
    #     wave = wave / max(wave)
    # samples = (wave * volume).astype(np.float32).tobytes()


def on_release(key):
    global wave
    global samples
    global volume
    volume[-frame_length // 10:] = volume[-frame_length // 10:] * np.linspace(0.5, 0, frame_length // 10)
    wave = np.zeros(frame_length)
    print("release")
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

while True:
    x_delta = np.ones(frame_length) * float(f)
    x_in = x_delta.cumsum() + x_in[-1] + 1
    data = 2 * np.pi * x_in / fs

    wave += np.sin(data)
    if max(wave) > 1:
        wave = wave / max(wave)
    samples = (wave * volume).astype(np.float32).tobytes()
    stream.write(samples)





old_f = 0


def test():
    global wave
    wave = np.zeros(frame_length)
    while True:
        global x_in
        global f
        global old_f
        global volume

        switch = np.arange(old_f, f, frame_length // 10)
        x_delta = np.ones(frame_length) * float(f)
        x_delta[0:frame_length // 10] = switch
        x_in = x_delta.cumsum() + x_in[-1] + 1
        data = 2 * np.pi * x_in / fs
        wave = np.sin(data) * fade_in_out
        if max(wave) > 1:
            wave = wave / max(wave)
        samples = (wave * volume).astype(np.float32).tobytes()
        stream.write(samples)
        if volume[-1] == 0:
            volume = np.zeros(frame_length)


testThrd = threading.Thread(target=test)
testThrd.start()
ii = 0

#
# while freq != "q":
#     ii += 1
#     freq = input("next freq")
#     if freq == "q": break
#
#     f_l = freq.split(sep=" ")
#     for i in range(len(f_l)):
#         k = freq.split(sep=" ")[0]
#         if i == 0:
#             if k == "usweep":
#                 func = np.sin
#                 multi = int(f_l[1])
#                 x_in = np.linspace(1, multi, duration * fs).cumsum()
#                 i += 2
#                 continue
#             elif k == "dsweep":
#                 func = np.sin
#                 multi = int(f_l[1])
#                 x_in = np.linspace(1, 1 / multi, duration * fs).cumsum()
#                 i += 2
#                 continue
#             elif k == "sawd":
#                 func = signal.sawtooth
#                 saw_width = 0
#                 continue
#             elif k == "saw":
#                 func = signal.sawtooth
#                 saw_width = 0.5
#                 continue
#             elif k == "z":
#                 wave = np.zeros(duration * fs)
#             else:
#                 func = np.sin
#
#         if f_l[i].isnumeric():
#             f = int(f_l[i])
#             data = 2 * np.pi * x_in * int(f) / fs
#             wave += func(data) * fade_in_out * 3 / (i + 0.5)
#         else:
#             break
#     if max(wave)>1:
#         wave = wave / max(wave)
#     samples = (wave * volume).astype(np.float32).tobytes()

stream.stop_stream()

stream.close()

p.terminate()
