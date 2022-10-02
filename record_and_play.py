import pyaudio
import time
import librosa
import numpy as np
import pandas as pd
import multiprocessing

FORMAT = pyaudio.paInt16
CHANNELS = 2
CHUNK = 1024*8
RATE = 44100

audio = pyaudio.PyAudio()
#
# for i in range(audio.get_device_count()):
#     print(json.dumps(audio.get_device_info_by_index(i), indent=2))

i=0
last_gradient_up = True
df = pd.DataFrame(columns=["signal", "sig2"])
last_data=np.sin(np.linspace(0, 2* np.pi*50, CHUNK*CHANNELS))
input_data=last_data.copy()
last_cross_over=last_data.copy()
signal=[]
signal_p=[]
res = 0
mixed_last_frame=0
mixed_crossover=0
def callback(in_data, frame_count, time_info, status):
    global mixed_last_frame
    global mixed_crossover
    global i
    global last_gradient_up
    global input_data
    global last_data
    global last_cross_over
    global res
    rev_l=1000
    repeats =4
    i +=0
    pitch_shift=-12

    data = np.frombuffer(in_data, np.int16) / 2 ** 15


    data = data.astype(np.float32)
    # data=np.sin(np.linspace(0, 2* np.pi*10*i/100, CHUNK*CHANNELS))
    input_data = np.hstack((input_data,data))

    signal_p.extend(input_data[:len(data)])
                # # Fade
                # k=100
                # ks=100
                # data[:ks]=data[:ks]*np.linspace(0,1,ks)
                # data[-k:]=data[-k:]*np.linspace(1,0,k)

    # Sine signal instead?
    # data =np.sin(np.linspace(0, 2*np.pi*5.133*i, len(in_data)))

    fr= CHUNK*CHANNELS
    c2 = int(fr/2)
    last_frame = librosa.effects.pitch_shift(input_data[:fr], sr=RATE, n_steps=pitch_shift)
    crossover_frame =librosa.effects.pitch_shift(input_data[c2:int(fr*1.5)], sr=RATE, n_steps=pitch_shift)


    input_data=input_data[fr:]
                    # # Downwards grad
                    # this_gradient_up=(data[10]-data[0]>0)
                    # if this_gradient_up==last_gradient_up:
                    #     # Same Gradient --do nothing
                    #     pass
                    # else:
                    #     # Flip signal
                    #     data = data * -1



    f_in = np.linspace(0,1,c2)
    f_out =np.linspace(1,0,c2)

    mixed_crossover=np.hstack((last_cross_over[c2:]*f_out,crossover_frame[:c2]* f_in))
    last_cross_over=crossover_frame
    mixed_last_frame=np.hstack((last_frame[c2:]*f_in,last_frame[:c2]* f_out))
    # res = np.correlate(end2x, end, mode='full')
    # buffer = 50
    # idx=np.argmax(res[:len(end2x)-buffer])
    data = mixed_crossover + mixed_last_frame


    # last_gradient_up=(data[-1]-data[-11]>0)
    # # Use reverse last past
    # rev=last_data[-2:-(rev_l+1):-1]*-1
    # rev=rev- (rev[0]-(last_data[-1]+last_data[-1]-last_data[-2]))
    # rev = rev* np.linspace(1,0, len(rev))
    # # Use repeated shifted last part
    # end = last_data[-rev_l:]
    # end2x = last_data[-2*rev_l:]
    # res = np.correlate(end2x, end, mode='full')
    # buffer = 50
    # idx=np.argmax(res[:len(end2x)-buffer])
    # print(len(end2x)-idx , idx)
    # best_rep = end2x[idx:]
    # best_rep = best_rep* np.linspace(1,0, len(best_rep))
    # last_data=data



    # # Apply reverse
    # old = data[:len(rev)]* np.linspace(0,1, len(rev))
    # data[:len(rev)] =rev +old
    # last_data=data[-100:]


    # # or applay correlat
    # old = data[:len(best_rep)]* np.linspace(0,1, len(best_rep))
    # data[:len(best_rep)] =best_rep +old


    # normalize
    signal.extend(data)
    data = data/max(max(data), 0.3)*0.8



    # Fade in out
    # k=200
    # ks=200
    # data[:ks]=data[:ks]*np.linspace(0,1,ks)
    # data[-k:]=data[-k:]*np.linspace(1,0,k)

    data = (data * 2 ** 15).astype(np.int16).tobytes()

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

df.signal=signal
df.sig2=signal_p
d=pd.DataFrame()
d["this"]=mixed_last_frame
d["last"]=mixed_crossover

df.signal.plot()
df.sig2.plot()
df.plot()
# stop stream (6)
stream.stop_stream()
stream.close()

# close PyAudio (7)
audio.terminate()
