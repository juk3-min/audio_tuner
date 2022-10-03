import pyaudio
import time
import librosa
import numpy as np
import pandas as pd
from multiprocessing import Process
from multiprocessing import Pipe
from pynput.keyboard import Key, Listener
import threading
FORMAT = pyaudio.paInt16
CHANNELS = 2
CHUNK = 1024
RATE = 44000
PITCH_SHIFT = -12
process_switch = True

lowcut = 400.0
highcut = 2000.0

def on_press(key):
    global PITCH_SHIFT
    # print('{0} pressed'.format(
    #     key))
    print(" ", flush=True)
    if key == Key.up:
        PITCH_SHIFT+=1
        time.sleep(0.01)
        print("Pitch Shift is ", PITCH_SHIFT)
    if key == Key.down:
        PITCH_SHIFT -= 1
        time.sleep(0.01)
        print("Pitch Shift is ", PITCH_SHIFT)


def on_release(key):
    # print('{0} release'.format(
    #     key))

    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
def foo():
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()



print("start")
#
# for i in range(audio.get_device_count()):
#     print(json.dumps(audio.get_device_info_by_index(i), indent=2))

i=1


last_gradient_up = True
df = pd.DataFrame(columns=["signal", "sig2"])


BYTE_CHUNK= CHUNK*CHANNELS
# last_data=[np.sin(np.linspace(0, 2* np.pi*50, BYTE_CHUNK))]

input_data=np.zeros(BYTE_CHUNK*2)
# last_cross_over=last_data.copy()

res = 0
# mixed_last_frame=0
# mixed_crossover=0

output_data = np.zeros(BYTE_CHUNK*2)
ham = np.hamming(BYTE_CHUNK*2)

from scipy.signal import butter, sosfilt, sosfreqz


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

OLD= np.zeros(BYTE_CHUNK*2)

def pitch_shift(conn1):
    global OLD

    while True:
        data, shift=conn1.recv()
        data=butter_bandpass_filter(data, lowcut, highcut, RATE, order=6)
        conn1.send(OLD)
        OLD=librosa.effects.pitch_shift(data, sr=RATE, n_steps=shift)


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

    repeats =20
    i +=0
    # Append 2th input

    s= time.time()
    res=in_data
    data=(np.frombuffer(in_data, np.int16)/(2**15)).astype(np.float32)
    input_data =np.roll(input_data,shift=BYTE_CHUNK)
    input_data[BYTE_CHUNK:]=data
    # input_data = np.hstack((last_data[0],last_data[1]))
    # l = len((np.frombuffer(in_data, np.int16) / 2 ** 15).astype(np.float32))
    # last_data.append(np.sin(np.linspace(0, 2 * np.pi *  (i % 50), l)))

    s2= time.time()
    if process_switch:
        conn2.send((input_data,PITCH_SHIFT))
        last_frame = conn2.recv()
    else:
        conn4.send((input_data,PITCH_SHIFT))
        last_frame = conn4.recv()
    process_switch = not process_switch

    s3 = time.time()
    # last_frame=input_data
    last_frame = ham*last_frame

    output_data +=last_frame
    data = output_data[:BYTE_CHUNK].copy()
    output_data[:BYTE_CHUNK]=0
    output_data=np.roll(output_data,shift=BYTE_CHUNK)
    # signal.extend(data)
    # data = data/max(max(data), 0.3)*0.8



    # Fade in out
    # k=200
    # ks=200
    # data[:ks]=data[:ks]*np.linspace(0,1,ks)
    # data[-k:]=data[-k:]*np.linspace(1,0,k)
    data = ((data*2**15).astype(np.int16)).tobytes()
    # data = res.tobytes()
    t =(time.time() - s)
    if t>CHUNK*0.9/RATE:
        print(t, s3-s2 , CHUNK/RATE)
    if i> repeats:
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

    df.signal.plot()
    df.sig2.plot()
    df.plot()