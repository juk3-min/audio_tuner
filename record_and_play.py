import pyaudio
import time
import librosa
import numpy as np
import pandas as pd

FORMAT = pyaudio.paInt16
CHANNELS = 2
CHUNK = 1024
RATE = 15000

print("start")
audio = pyaudio.PyAudio()
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
signal=[]
signal_p=[]
res = 0
# mixed_last_frame=0
# mixed_crossover=0

output_data = np.zeros(BYTE_CHUNK*2)
ham = np.hamming(BYTE_CHUNK*2)

def callback(in_data, frame_count, time_info, status):
    global mixed_last_frame
    global mixed_crossover
    global i
    global last_gradient_up
    global input_data
    global last_data
    global last_cross_over
    global res
    global output_data

    repeats =20
    i +=0
    pitch_shift=-12
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
    last_frame = librosa.effects.pitch_shift(input_data, sr=RATE, n_steps=pitch_shift)
    s3 = time.time()
    # last_frame=input_data
    last_frame = ham*last_frame

    output_data +=last_frame
    data = output_data[:BYTE_CHUNK].copy()
    output_data[:BYTE_CHUNK]=0
    output_data=np.roll(output_data,shift=BYTE_CHUNK)
    # signal.extend(data)
    data = data/max(max(data), 0.3)*0.8



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

for k in [1]:
    print(f"recording in {k} s")
    time.sleep(1)
print("now")


stream = audio.open(format              = FORMAT,
                    channels            = CHANNELS,
                    rate                = RATE,
                    input               = True,
                    output              = True,
                    input_device_index  = 7,
                    frames_per_buffer   = CHUNK,
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
df.signal=signal
df.sig2=signal_p
d=pd.DataFrame()

df.signal.plot()
df.sig2.plot()
df.plot()

