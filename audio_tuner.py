import wave
from pylab import *
from matplotlib import *
import numpy as np
import pyaudio
import matplotlib.pyplot as plt

class Tuner():
    def __init__(a,RATE, lowest_frequency, inputname):
        a.stream=None
        a.FORMAT = pyaudio.paInt16
        a.CHANNELS = 1
        a.RATE = RATE

        needed_time = 1/lowest_frequency
        needed_data_length=a.RATE*needed_time
        a.CHUNK = int(needed_data_length)

        a.index_of_chosen_input_device = 1
        a.data =None
        # # Get the right input device with the name of
        # input_name = "What U Hear"
        input_name=inputname
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
                if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i))
                    name=p.get_device_info_by_host_api_device_index(0, i)['name']
                    if name.rfind(input_name)>=0:
                        a.index_of_chosen_input_device=i
                        print("found")
                        break




    def open_stream(a):
        p = pyaudio.PyAudio()
        a.stream = p.open(format=a.FORMAT,
                        channels=a.CHANNELS,
                        rate=a.RATE,
                        input_device_index=a.index_of_chosen_input_device,
                        input=True,
                        frames_per_buffer=a.CHUNK)

    def open_stream_out(a):
        p = pyaudio.PyAudio()
        a.stream = p.open(format=a.FORMAT,
                        channels=a.CHANNELS,
                        rate=a.RATE,
                        input_device_index=a.index_of_chosen_input_device,
                        input=True,
                        output_device_index=1,
                        frames_per_buffer=a.CHUNK)



    def record_for_time(a,RECORD_SECONDS):
        a.open_stream()
        print("Recording")
        data=[]
        for i in range(0, int(a.RATE / a.CHUNK * RECORD_SECONDS)):
            a.data = a.stream.read(a.CHUNK)
            data.append[a.data]

        a.stream.stop_stream()
        print("Recording Stopped")
        return np.frombuffer(data, np.int16)

    def record_for_chunk(a):
        # a.open_stream()
        a.data = a.stream.read(a.CHUNK)
        # a.stream.stop_stream()
        return np.frombuffer(a.data, np.int16)

    def close_stream(a):
        a.stream.close()

    def plot_specgram(a, data=None,NFFT=None):
        if data==None:
            data=a.data
        if NFFT==None:
            NFFT=a.CHUNK
        else: NFFT=128
        fig, ax1 = plt.subplots(nrows=1, figsize=(14, 7))
        ax1.specgram(data, NFFT=NFFT, Fs=a.RATE, scale_by_freq=False)
        plt.show()


