import pyaudio
import time
import librosa
import numpy as np
import pandas as pd
from multiprocessing import Process
from multiprocessing import Pipe
from pynput.keyboard import Key, Listener
import note_finder
import threading
import math

from note_finder import auto_pitch

FORMAT = pyaudio.paInt16
CHANNELS = 2
CHUNK = 1024*4
RATE = 22000
PITCH_SHIFT = -12
process_switch = True
signal = []
lowcut = 100.0
highcut = 2000.0
from scipy.signal import butter, sosfilt, sosfreqz


def on_press(key):
    global PITCH_SHIFT
    global i
    global overlay_mode
    global delete_mode
    global base
    if key == Key.up:
        PITCH_SHIFT += 1
        time.sleep(0.01)
        print("'\r{0}".format("Freq: "), base , 10 * " ",end='')
    if key == Key.down:
        PITCH_SHIFT -= 1
        time.sleep(0.01)
        print("'\r{0}".format("Freq: "), base , 10 * " ",end='')
    try:
        if key.char == "q":
            i = float("inf")
        if key.char == "o":
            overlay_mode = True
        if key.char == "d":
            delete_mode = True
    except AttributeError:
        pass


def on_release(key):
    if key == Key.esc:
        return False


# Collect events until released
def foo():
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()

print("start")


i = 1
counter = 0
last_gradient_up = True
df = pd.DataFrame(columns=["signal", "sig2"])
BYTE_CHUNK = CHUNK * CHANNELS

input_data = np.zeros(BYTE_CHUNK * 2)
res = 0
output_data = np.zeros(BYTE_CHUNK * 2)
ham = np.hamming(BYTE_CHUNK * 2)

overlay_mode = False
delete_mode = False
base = [0]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


OLD = np.zeros(BYTE_CHUNK * 2)

def pitch_shift(conn1):
    global OLD

    while True:
        data, shift = conn1.recv()
        # data = butter_bandpass_filter(data, lowcut, highcut, RATE, order=6)
        pitches, magnitudes = librosa.piptrack(y=data, sr=RATE)
        # pitch=max(1,min(sum([max(p) for p in pitches])/len(pitches), 9999))
        pitch=max(1,min(np.max((pitches)), 9999))
        notes = ["A","B","C","D","E","F","G"]
        notes = ["A","C","D","E","G"]
        print(pitch)
        auto_pitch= note_finder.auto_pitch(pitch, notes)
        #
        # pitch_delta=(pitch - auto_pitch) / abs(closest_pitch - other_pitch)
        half_steps= 12*(math.log(auto_pitch)-math.log(pitch))/(math.log(2))
        print(auto_pitch, pitch, half_steps)
        # print(pitch, pitch_delta)

        conn1.send(OLD)
        OLD = librosa.effects.pitch_shift(data, sr=RATE, n_steps=half_steps)

        # pitches, magnitudes = librosa.piptrack(y=OLD, sr=RATE)
        # pitch=np.max((pitches))
        # closest_note, closest_pitch, other_pitch= note_finder.find_closest_note(np.max((pitches)))
        # pitch_delta=(pitch - closest_pitch) / abs(closest_pitch - other_pitch)
        # print(pitch_delta)
        OLD = butter_bandpass_filter(OLD, 10, highcut, RATE, order=6)


def callback(in_data, frame_count, time_info, status):
    global conn2
    global conn4
    global mixed_last_frame
    global mixed_crossover
    global i
    global last_gradient_up
    global input_data
    global last_data
    global last_cross_over
    global res
    global output_data
    global process_switch
    global PITCH_SHIFT
    global signal
    global counter
    global base
    global overlay_mode
    global delete_mode
    repeats = 5
    i += 0
    # Append 2th input

    s = time.time()
    res = in_data
    data = (np.frombuffer(in_data, np.int16) / (2 ** 15)).astype(np.float32)
    input_data = np.roll(input_data, shift=BYTE_CHUNK)
    input_data[BYTE_CHUNK:] = data


    s2 = time.time()
    if process_switch:
        conn2.send((input_data,PITCH_SHIFT))
        this_frame = conn2.recv()
    else:
        conn4.send((input_data,PITCH_SHIFT))
        this_frame = conn4.recv()
    process_switch = not process_switch

    #
    # # Sine Geneartor
    # x_data = np.linspace(counter, counter + BYTE_CHUNK, BYTE_CHUNK * 2)
    # if overlay_mode:
    #     base.append(440 + PITCH_SHIFT * 2)
    #     overlay_mode = False
    # if delete_mode:
    #     base = [0]
    #     delete_mode = False
    # base[0] = 440 + PITCH_SHIFT * 2
    # this_frame = x_data * 0
    # for frq in base:
    #     this_frame += np.sin(2 * np.pi * x_data * (frq) / RATE)


    last_data = this_frame
    counter += BYTE_CHUNK / 2
    s3 = time.time()
    this_frame = ham * this_frame

    output_data += this_frame
    data = output_data[:BYTE_CHUNK].copy()
    output_data[:BYTE_CHUNK] = 0
    output_data = np.roll(output_data, shift=BYTE_CHUNK)
    signal.extend(data)
    data = data / max(max(data), 0.3) * 0.8
    data = ((data * 2 ** 15).astype(np.int16)).tobytes()
    t = (time.time() - s)

    # if t > CHUNK * 0.9 / RATE:
    #     print(t, s3 - s2, CHUNK / RATE)
    if i > repeats:
        return data, pyaudio.paComplete
    return data, pyaudio.paContinue


if __name__ == '__main__':
    # create the pipe
    conn1, conn2 = Pipe(duplex=True)
    conn3, conn4 = Pipe(duplex=True)
    audio = pyaudio.PyAudio()

    player1 = Process(target=pitch_shift, args=(conn1,))
    player1.start()

    player2 = Process(target=pitch_shift, args=(conn3,))
    player2.start()

    x = threading.Thread(target=foo, args=(), daemon=True)
    x.start()

    for k in [1]:
        print(f"recording in {k} s")
        time.sleep(1)
    print("now")

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        input_device_index=7,
                        frames_per_buffer=CHUNK,
                        stream_callback=callback)

    stream.start_stream()

    # wait for stream to finish (5)
    while stream.is_active():
        time.sleep(0.1)

    # stop stream (6)
    stream.stop_stream()
    stream.close()

    # close PyAudio (7)
    audio.terminate()
    d = pd.DataFrame()
    df = pd.DataFrame()
    df["signal:out"] = signal

    df.plot()
