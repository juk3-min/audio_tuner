import pyaudio
import numpy as np
from pynput.keyboard import Key, Listener
import threading
from scipy import signal
from typing import Protocol
import pandas as pd
from matplotlib import pyplot as plt

pressed = False
p = pyaudio.PyAudio()

VOLUME = 0.5  # range [0.0, 1.0]
SAMPLING_RATE = 44100  # sampling rate, Hz, must be integer
SAMPLE_DURATION = 0.01  # in seconds, may be float
frequencies = [100]  # sine frequency, Hz, may be float
LOOP = True
phases = []


class WaveGenerator(Protocol):
    def create_wave(self):
        ...


class AudioGenerator:
    def __init__(self, sample_rate: int, samples: int, channels: int = 1):
        self.wave_generators = dict()
        self.end_waves = list()
        self.sample_rate = sample_rate
        self.samples = samples
        self.channels = channels
        self.volume: float = 0.2
        self.stream = None
        self.open_stream()
        self.waves = np.zeros(samples)

    def open_stream(self):
        self.stream = p.open(format=pyaudio.paFloat32,
                             channels=self.channels,
                             rate=self.sample_rate,
                             output=True)

    def add_wave_generator(self, key, wave_generator: WaveGenerator):
        self.wave_generators[key] = wave_generator

    def remove_wave_generator(self, key):
        try:
            self.end_waves.append(self.wave_generators[key])
            del self.wave_generators[key]
        except KeyError:
            pass

    def play(self):
        output = np.zeros(self.samples)
        wave_generators = self.wave_generators.copy()
        for wave_generator in wave_generators.values():
            output += wave_generator.create_wave()
        fade_out = np.zeros(self.samples)
        fade_out[:self.samples//4] = np.linspace(1, 0, self.samples//4)
        for wave_generator in self.end_waves:
            output += wave_generator.create_wave() * fade_out
        self.end_waves = []
        output = output  # / (max(len(wave_generators), 1))
        self.waves = np.hstack((self.waves, output))
        output = (output * self.volume).astype(np.float32).tobytes()
        self.stream.write(output)

    def close_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        p.terminate()


class SingleWaveGenerator:
    def __init__(self, freq: float, samples: int, sample_rate: int):
        self.freq = freq
        self.samples = samples
        self.sample_rate = sample_rate
        self.signal_function = np.sin
        self.phase = 0

    def create_wave(self):
        x_data = np.arange(self.samples) + self.phase
        phase_helper = self.freq * 2 * np.pi / self.sample_rate
        self.phase = (((x_data[-1] + 1) * phase_helper) % (2 * np.pi)) / phase_helper
        output = self.signal_function(x_data * phase_helper)
        return output


class HarmonicsWaveGenerator(SingleWaveGenerator):
    def __init__(self, freq: float, samples: int, sample_rate: int, harmonics: list = None, volumes: list = None):
        super().__init__(freq, samples, sample_rate)

        if harmonics is None:
            self.harmonics = [1, 3, 5, 7]
        else:
            self.harmonics = harmonics

        if volumes is None:
            self.volumes = [1, 0.5, 0.33, 0.2]
        else:
            self.volumes = harmonics

        assert len(self.volumes) == len(self.harmonics)

        self.wave_generators = []
        for i, harmonic in enumerate(self.harmonics):
            wave_generator = SingleWaveGenerator(self.freq * harmonic, self.samples, self.sample_rate)
            self.wave_generators.append(wave_generator)

    def create_wave(self):
        output = np.zeros(self.samples)
        for i, wave_generator in enumerate(self.wave_generators):
            output += wave_generator.create_wave() * self.volumes[i]
        return output / np.max(output)


class AudioController:
    def __init__(self, audio_generator: AudioGenerator):
        self.audio_generator = audio_generator
        self.thread = None
        self.exit = False
        self.pressed = set()
        self.listener: Listener

    def stop_thread(self):
        self.thread.join()

    def get_samples(self):
        return self.audio_generator.samples

    def create_listener(self):
        self.listener = Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        with self.listener as listener:
            listener.join()

    def start_thread(self):
        self.thread = threading.Thread(target=self.create_listener, args=(), daemon=True)
        self.thread.start()

    def get_sample_rate(self):
        return self.audio_generator.sample_rate

    def on_press(self, key):
        keys_for_c = ["a", "s", "d", "f", "g", "h", "j", "k", "l", "ö", "ä"]
        try:
            if key.char in self.pressed:
                return

            if key.char in keys_for_c:
                self.pressed.add(key.char)
                index = keys_for_c.index(key.char)
                c2 = 90
                freq = c2 * (2 ** (index / 12))
                self.audio_generator.add_wave_generator(key=key.char,
                                                        wave_generator=SingleWaveGenerator(
                                                            freq=freq,
                                                            samples=self.get_samples(),
                                                            sample_rate=self.get_sample_rate()))
        except AttributeError:
            pass

    def loop(self):
        while self.exit is False:
            self.audio_generator.play()
        self.audio_generator.close_stream()

    def on_release(self, key):
        try:
            if key == Key.esc:
                self.exit = True
                self.listener.stop()
            self.pressed.remove(key.char)
            self.audio_generator.remove_wave_generator(key.char)
        except (AttributeError, KeyError):
            pass


#
# def create_harmonic_wave(freq: float, phase: float = 0):
#     x_data = np.arange(SAMPLING_RATE * SAMPLE_DURATION)# + phase
#     offset = (SAMPLING_RATE+1) * SAMPLE_DURATION*freq / SAMPLING_RATE
#     aim = (offset // 1)
#     freq = aim * SAMPLING_RATE / ((SAMPLING_RATE+1)*SAMPLE_DURATION)
#     base_phase = freq * 2 * np.pi / SAMPLING_RATE
#     output = x_data.copy()*0
#     for i, harmonics in enumerate(np.arange(1,8,2)):
#         phase_helper = (harmonics*freq) * 2 * np.pi / SAMPLING_RATE
#         output += np.sin(x_data * phase_helper)/(i+1)
#     phase = (((x_data[-1]+1)*base_phase) % (2*np.pi)) / phase_helper
#     # print(output[0],output[-1])
#     return output,  phase
#
# def create_wave(freq: float, phase: float = 0):
#     x_data = np.arange(SAMPLING_RATE * SAMPLE_DURATION) + phase
#     phase_helper = freq * 2 * np.pi / SAMPLING_RATE
#     wave = np.sin(x_data * phase_helper)
#     phase = (((x_data[-1]+1)*phase_helper) % (2*np.pi)) / phase_helper
#     return wave,  phase


def main():
    audio_generator = AudioGenerator(sample_rate=SAMPLING_RATE, samples=int(SAMPLE_DURATION * SAMPLING_RATE),
                                     channels=1)
    audio_controller = AudioController(audio_generator=audio_generator)
    audio_controller.start_thread()
    audio_controller.loop()
    pd.DataFrame(audio_generator.waves).plot()
    plt.show()

    # generate samples, note conversion to float32 array
    # x_data = np.arange(SAMPLING_RATE * SAMPLE_DURATION)
    # fade_in_out = np.sin(np.arange(0, np.pi, np.pi / len(x_data)))
    # wave, phase = create_wave(frequencies[0])
    # wave *= fade_in_out
    #
    #
    # x = threading.Thread(target=foo, args=(), daemon=True)
    # x.start()
    #
    # sample_wave = wave
    # global phases
    # global pressed
    #
    # phases =[phase]
    # # for _ in range(int(10/SAMPLE_DURATION)):
    # while LOOP:
    #     sample_wave *= 0
    #     if pressed:
    #         for i, freq in enumerate(frequencies):
    #             try:
    #                 phases[i]
    #             except IndexError:
    #                 phases.append(phases[i-1])
    #             wave, phases[i] = create_wave(freq, phases[i])
    #             sample_wave += wave/len(frequencies)
    #
    #
    #
    # p.terminate()


#
#
# def on_press(key):
#     global LOOP
#     global frequencies
#     global pressed
#     pressed = True
#     steps = 12
#     if key == Key.up:
#         frequencies[-1] = frequencies[-1] * (2 ** (1 / steps))
#         print(frequencies[-1])
#     if key == Key.down:
#         frequencies[-1] = frequencies[-1] / (2 ** (1 / steps))
#         print(frequencies[-1])
#     try:
#         if key.char == "q":
#             LOOP = False
#         if key.char == "i":
#             print(frequencies)
#             print(phases)
#         if key.char == "o":
#             frequencies.append(frequencies[-1])
#         if key.char == "r":
#             if len(frequencies) > 1:
#                 frequencies = frequencies[:-1]
#         keys_for_c = ["a", "s", "d", "f", "g", "h", "j","k", "l", "ö", "ä"]
#         if key.char in keys_for_c:
#             pressed = True
#             index = keys_for_c.index(key.char)
#             c2 = 90
#             frequencies[-1] = c2 * (2 ** (index/12))
#
#     except AttributeError:
#         pass
#
#
# def on_release(key):
#     keys_for_c = ["a", "s", "d", "f", "g", "h", "j", "k", "l", "ö", "ä"]
#     global pressed
#     if key.char in keys_for_c:
#         pressed = False
#
#     if key == Key.esc:
#         global LOOP
#         LOOP = False
#         return False
#
#
#
# # Collect events until released
# def create_listener():
#     with Listener(
#             on_press=on_press,
#             on_release=on_release) as listener:
#         listener.join()

if __name__ == "__main__":
    main()
