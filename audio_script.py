import pyaudio
import numpy as np
from pynput.keyboard import Key, Listener
import threading
from scipy import signal


pressed =False
p = pyaudio.PyAudio()

VOLUME = 0.5  # range [0.0, 1.0]
SAMPLING_RATE = 44100  # sampling rate, Hz, must be integer
SAMPLE_DURATION = 0.05  # in seconds, may be float
frequencies = [100]  # sine frequency, Hz, may be float
LOOP = True
phases = []

def create_harmonic_wave(freq: float, phase: float = 0):
    x_data = np.arange(SAMPLING_RATE * SAMPLE_DURATION)# + phase
    offset = (SAMPLING_RATE+1) * SAMPLE_DURATION*freq / SAMPLING_RATE
    aim = (offset // 1)
    freq = aim * SAMPLING_RATE / ((SAMPLING_RATE+1)*SAMPLE_DURATION)
    base_phase = freq * 2 * np.pi / SAMPLING_RATE
    output = x_data.copy()*0
    for i, harmonics in enumerate(np.arange(1,8,2)):
        phase_helper = (harmonics*freq) * 2 * np.pi / SAMPLING_RATE
        output += np.sin(x_data * phase_helper)/(i+1)
    phase = (((x_data[-1]+1)*base_phase) % (2*np.pi)) / phase_helper
    # print(output[0],output[-1])
    return output,  phase

def create_wave(freq: float, phase: float = 0):
    x_data = np.arange(SAMPLING_RATE * SAMPLE_DURATION) + phase
    phase_helper = freq * 2 * np.pi / SAMPLING_RATE
    wave = np.sin(x_data * phase_helper)
    phase = (((x_data[-1]+1)*phase_helper) % (2*np.pi)) / phase_helper
    return wave,  phase


def main():
    # generate samples, note conversion to float32 array
    x_data = np.arange(SAMPLING_RATE * SAMPLE_DURATION)
    fade_in_out = np.sin(np.arange(0, np.pi, np.pi / len(x_data)))
    wave, phase = create_wave(frequencies[0])
    wave *= fade_in_out
    channels = 1
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=channels,
                    rate=SAMPLING_RATE,
                    output=True)

    x = threading.Thread(target=foo, args=(), daemon=True)
    x.start()

    sample_wave = wave
    global phases
    global pressed

    phases =[phase]
    # for _ in range(int(10/SAMPLE_DURATION)):
    while LOOP:
        sample_wave *= 0
        if pressed:
            for i, freq in enumerate(frequencies):
                try:
                    phases[i]
                except IndexError:
                    phases.append(phases[i-1])
                wave, phases[i] = create_wave(freq, phases[i])
                sample_wave += wave/len(frequencies)
        samples = (sample_wave * VOLUME).astype(np.float32).tobytes()
        stream.write(samples)

    stream.stop_stream()
    stream.close()
    p.terminate()


def on_press(key):
    global LOOP
    global frequencies
    global pressed
    pressed = True
    steps = 12
    if key == Key.up:
        frequencies[-1] = frequencies[-1] * (2 ** (1 / steps))
        print(frequencies[-1])
    if key == Key.down:
        frequencies[-1] = frequencies[-1] / (2 ** (1 / steps))
        print(frequencies[-1])
    try:
        if key.char == "q":
            LOOP = False
        if key.char == "i":
            print(frequencies)
            print(phases)
        if key.char == "o":
            frequencies.append(frequencies[-1])
        if key.char == "r":
            if len(frequencies) > 1:
                frequencies = frequencies[:-1]
        keys_for_c = ["a", "s", "d", "f", "g", "h", "j","k", "l", "ö", "ä"]
        if key.char in keys_for_c:
            pressed = True
            index = keys_for_c.index(key.char)
            c2 = 90
            frequencies[-1] = c2 * (2 ** (index/12))

    except AttributeError:
        pass


def on_release(key):
    keys_for_c = ["a", "s", "d", "f", "g", "h", "j", "k", "l", "ö", "ä"]
    global pressed
    if key.char in keys_for_c:
        pressed = False

    if key == Key.esc:
        global LOOP
        LOOP = False
        return False



# Collect events until released
def foo():
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
