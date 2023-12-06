from __future__ import annotations

import random
import time
from typing import Protocol

from matplotlib import pyplot as plt
import pyaudio
import numpy as np
from pygame import Surface, SurfaceType
from pynput.keyboard import Key, Listener
import threading

import mido

from wave_generator import WaveGenerator

import pygame
import sys

VOLUME = 0.5  # range [0.0, 1.0]
SAMPLING_RATE = 44100  # sampling rate, Hz, must be integer
SAMPLE_DURATION = 0.01  # in seconds, may be float


def main():
    #########
    # Mapping of keys to note indicies
    keys_for_c = ["#"] * 80
    keys_for_c[60] = "3"
    keys_for_c[64] = "2"
    keys_for_c[65] = "1"
    keys_for_c[67] = "5"
    keys_for_c[69] = "j"
    keys_for_c[70] = "h"
    keys_for_c[72] = "g"
    keys_for_c[74] = "f"

    print("Starting Audio Generator with Listener Thread")

    def envelope(data, time):
        length = len(data)
        return (data * np.max((np.ones(length)*0.5,
                np.exp(-0.2 * time) *
                np.min(np.vstack((np.ones(length), np.exp((time+0.1)*5*(np.arange(length) / length)))),
                       axis=0))))

    audio_generator = AudioGenerator(sample_rate=SAMPLING_RATE,
                                     samples=int(SAMPLE_DURATION * SAMPLING_RATE), channels=1,
                                     play_time_function=envelope)

    audio_controller = AudioController(audio_generator=audio_generator, keys_for_c=keys_for_c)
    audio_controller.base_freq = 1
    import json

    # with open("piano_fourier_coff", "rb") as f:
    with open("violin_fourier_coff", "rb") as f:
        fourier = json.load(f)

    from wav_to_generator_converter import FourierWaveGenerator
    def wave_factory(**kwargs):
        f = FourierWaveGenerator(**kwargs, anbn=fourier, wave_generator=SingleWaveGenerator)
        return f

    audio_controller.wave_generator = wave_factory

    audio_controller.start_thread()

    print("Starting Controller Loop which plays the audio generator ")
    thread = threading.Thread(target=audio_controller.loop, args=(), daemon=True)
    thread.start()

    frame = AnimationFrame(800, 600)
    frame.show()

    midi_animator = MidiAnimator(midifilepath="oh tannenbaum.mid", screen=frame.screen,
                                 keys_for_c=keys_for_c)
    frame.controller = midi_animator
    midi_animator.mid.ticks_per_beat = 80

    # Controller thread
    def animate():
        try:
            midi_animator.animate()
        except pygame.error:
            return

    thread = threading.Thread(target=midi_animator.create_listener, args=(), daemon=True)
    thread.start()

    thread = threading.Thread(target=animate, args=(), daemon=True)
    thread.start()

    print("t1")
    def foo(data, time):
        return data * np.exp(-time)

    audio_generator = AudioGenerator(sample_rate=SAMPLING_RATE,
                                     samples=int(SAMPLE_DURATION * SAMPLING_RATE), channels=1,
                                     play_time_function=foo)


    audio_controller = AudioController(audio_generator=audio_generator, keys_for_c=keys_for_c)
    audio_controller.base_freq = 2
    audio_controller.start_thread()

    print("t2")

    thread = threading.Thread(target=audio_controller.loop, args=(), daemon=True)
    thread.start()

    print("t3")
    # Start he pygame loop
    frame.loop()

    ...


class AudioGenerator:
    def __init__(self,play_time_function, sample_rate: int, samples: int, channels: int = 1):
        self.wave_generators = dict()
        self.play_times = dict()
        self.play_time_function=play_time_function
        self.end_waves = dict()
        self.sample_rate = sample_rate
        self.samples = samples
        self.channels = channels
        self.volume: float = 0.2
        self.stream = None
        self.wave = [0, 0]
        self.open_stream()
        self.p: pyaudio.PyAudio

    def restart(self):
        self.close_stream()
        self.open_stream()

    def open_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paFloat32,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  output=True)

    def add_wave_generator(self, key, wave_generator: WaveGenerator):
        self.wave_generators[key] = wave_generator
        self.play_times[key] = 0

    def remove_wave_generator(self, key):
        try:
            self.end_waves[key] = self.wave_generators[key]
            del self.wave_generators[key]
        except KeyError:
            pass

    def play(self):
        output = np.zeros(self.samples)
        wave_generators = self.wave_generators.copy()
        play_times = self.play_times.copy()
        for key, wave_generator in wave_generators.items():
            add_output = wave_generator.create_wave()
            add_output = self.play_time_function(add_output, play_times[key])
            output += add_output
            self.play_times[key] += self.samples / self.sample_rate

        fade_out = np.zeros(self.samples)
        fade_out[:] = np.linspace(1, 0, self.samples)
        end_waves = self.end_waves.copy()
        for key, wave_generator in end_waves.items():
            output += self.play_time_function(wave_generator.create_wave() * fade_out, play_times[key])
            self.play_times[key] = 0
        self.end_waves = dict()
        output = output  # / (max(len(wave_generators), 1))
        output = (output * self.volume).astype(np.float32).tobytes()
        self.stream.write(output)

    def close_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        self.p.terminate()
        self.p = None


class SingleWaveGenerator:
    def __init__(self, freq: float, samples: int, sample_rate: int):
        self.freq = freq
        self.samples = samples
        self.sample_rate = sample_rate
        self.signal_function = np.sin
        self.phase = 0
        self.volume = 1

    def pitch(self, factor: float):
        self.freq *= factor

    def create_wave(self):
        x_data = np.arange(self.samples) + self.phase
        phase_helper = self.freq * 2 * np.pi / self.sample_rate
        self.phase = (((x_data[-1] + 1) * phase_helper) % (2 * np.pi)) / phase_helper
        if np.isnan(self.phase):
            self.phase = 0
        output = self.signal_function(x_data * phase_helper)
        return output * self.volume


class HarmonicsWaveGenerator(SingleWaveGenerator):
    def __init__(self, freq: float, samples: int, sample_rate: int, harmonics: list = None,
                 volumes: list = None):
        super().__init__(freq, samples, sample_rate)

        if harmonics is None:
            self.harmonics = [1, 3, 5, 7]  # , 9] , 11, 13]
        else:
            self.harmonics = harmonics

        if volumes is None:
            self.volumes = [1, 0.5, 0.33, 0.2]  # , 1 / 6] #, 1 / 7, 1 / 8]
        else:
            self.volumes = harmonics

        assert len(self.volumes) == len(self.harmonics)

        self.wave_generators = []
        for i, harmonic in enumerate(self.harmonics):
            wave_generator = SingleWaveGenerator(self.freq * harmonic, self.samples,
                                                 self.sample_rate)
            self.wave_generators.append(wave_generator)

    def pitch(self, factor: float):
        self.freq *= factor
        for wave_gen in self.wave_generators:
            wave_gen.freq *= factor

    def create_wave(self):
        output = np.zeros(self.samples)
        for i, wave_generator in enumerate(self.wave_generators):
            output += wave_generator.create_wave() * self.volumes[i]
        return output * self.volume


class MidiController:
    def __init__(self, audio_generator: AudioGenerator, midifilepath):
        self.mid = mido.MidiFile(midifilepath)
        self.audio_generator = audio_generator
        self.dur = self.audio_generator.samples / self.audio_generator.sample_rate
        self.base_freq = 8

    def start_thread(self):
        self.thread = threading.Thread(target=self.play, args=(), daemon=True)
        self.thread.start()

    def play(self):
        self.mid.ticks_per_beat = 40
        for msg in self.mid.play():
            if msg.type == "note_on":
                note = msg.note
                volume = msg.velocity / 100
                note = self.base_freq * (2 ** (note / 12))
                print(note, volume)
                try:
                    if volume == 0:
                        self.audio_generator.remove_wave_generator(note)
                    else:
                        self.audio_generator.wave_generators[note].volume = volume
                except KeyError:
                    self.audio_generator.add_wave_generator(note, SingleWaveGenerator(note,
                                                                                      self.audio_generator.samples,
                                                                                      self.audio_generator.sample_rate))
                    self.audio_generator.wave_generators[note].volume = volume

    def loop(self):
        for _ in range(50000):
            if self.audio_generator.stream is not None:
                print("playing")
                self.audio_generator.play()
        self.audio_generator.close_stream()


class AudioController:
    def __init__(self, audio_generator: AudioGenerator, keys_for_c=None):
        self.audio_generator = audio_generator
        self.thread = None
        self.exit = False
        self.pressed = set()
        self.listener: Listener
        self.restart = False
        self.keys_for_c = keys_for_c
        self.base_freq = 50
        self.wave_generator: WaveGenerator
        if keys_for_c is None:
            self.keys_for_c = ["z", "u", "i", "o", "p", "ü", "y", "s", "x", "d", "c", "v", "g", "b",
                               "h", "n", "j", "m",
                               ",",
                               "l", "."]

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

    def on_press(self, key):  # F  #E     #D       #C
        keys_for_c = self.keys_for_c
        try:
            if key.vk in range(96, 106):
                key.char = str(key.vk - 96)
        except:
            pass

        try:
            if key.name == "up":
                self.base_freq *= 1.05
                for wave_gen in self.audio_generator.wave_generators.values():
                    wave_gen.pitch(1.05)

            if key.name == "down":
                self.base_freq *= 0.95
                for wave_gen in self.audio_generator.wave_generators.values():
                    wave_gen.pitch(0.95)
        except:
            pass
        try:
            if key.char in self.pressed:
                return

            if key.char == "W":
                if self.wave_generator == SingleWaveGenerator:
                    self.wave_generator = HarmonicsWaveGenerator
                    print("switched to harmonics wave generator")
                else:
                    self.wave_generator = SingleWaveGenerator
                    print("switched to single wave generator")
            if key.char == "P":
                plt.plot(self.audio_generator.wave)
                plt.show()


            if key.char in [char for sub_list in keys_for_c for char in sub_list]:
                self.pressed.add(key.char)
                index = 0
                for i, keys in enumerate(keys_for_c):
                    if key.char in keys:
                        index = i
                        break
                # index = keys_for_c.index(key.char)
                freq = self.base_freq * (2 ** (index / 12))
                self.audio_generator.add_wave_generator(key=key.char,
                                                        wave_generator=self.wave_generator(
                                                            freq=freq,
                                                            samples=self.get_samples(),
                                                            sample_rate=self.get_sample_rate()))
        except AttributeError:
            pass

    def loop(self):
        while self.exit is False:
            if self.audio_generator.stream is not None:
                self.audio_generator.play()
            if self.restart:
                print("restarting")
                self.audio_generator.restart()
                self.restart = False
        self.audio_generator.close_stream()

    def on_release(self, key):
        try:
            if key == Key.esc:
                self.exit = True
                self.listener.stop()
            if key.char is None:
                try:
                    key.char = str(key.vk - 96)
                except:
                    pass

            self.pressed.remove(key.char)
            self.audio_generator.remove_wave_generator(key.char)
        except (AttributeError, KeyError):
            pass


def play_with_midi_controller():
    audio_generator = AudioGenerator(sample_rate=SAMPLING_RATE,
                                     samples=int(SAMPLE_DURATION * SAMPLING_RATE), channels=1,
                                     play_time_function=lambda x, y: x)

    midi_controller = MidiController(audio_generator=audio_generator,
                                     midifilepath="oh tannenbaum.mid")
    midi_controller.start_thread()
    midi_controller.loop()


def play_with_audio_controller():
    audio_generator = AudioGenerator(sample_rate=SAMPLING_RATE,
                                     samples=int(SAMPLE_DURATION * SAMPLING_RATE), channels=1,
                                     play_time_function=lambda x, y: x)
    audio_controller = AudioController(audio_generator=audio_generator)
    audio_controller.start_thread()
    audio_controller.loop()


class AnimationObject(Protocol):
    def draw(self):
        ...

    def update_pos(self, x, y):
        ...

    def move_pos(self, dx, dy):
        ...

    def get_pos(self) -> tuple[int, int]:
        ...


class PyGameRect(AnimationObject):
    def __init__(self, screen, width, height, x_pos: int, y_pos: int, color):
        self.screen = screen
        self.width = width
        self.height = height
        self.x: int = x_pos
        self.y: int = y_pos
        self.color = color

    def update_pos(self, x: int, y: int):
        self.x = x
        self.y = y

    def move_pos(self, dx: int, dy: int):
        self.x += dx
        self.y += dy

    def get_pos(self) -> (int, int):
        return (self.x, self.y)

    def draw(self):
        pygame.draw.rect(self.screen, self.color,
                         (self.x,
                          self.y,
                          self.width,
                          self.height))


class PyGameText(AnimationObject):
    def __init__(self, screen, text, x_pos, y_pos, color, font, size):
        self.screen = screen
        self.text = text
        self.color = color
        pygame.init()
        self.font = pygame.font.Font(font, size)
        self.x = x_pos
        self.y = y_pos

    def update_pos(self, x, y):
        self.x = x
        self.y = y

    def move_pos(self, dx, dy):
        self.x += dx
        self.y += dy

    def get_pos(self) -> (int, int):
        return self.x, self.y

    def draw(self):
        text = self.font.render(self.text, True, self.color)
        text_rect = text.get_rect()
        text_rect.topleft = (self.x, self.y)
        self.screen.blit(text, text_rect)


def out_of_bounds(obj: AnimationObject, screen: Surface | SurfaceType) -> bool:
    x, y = obj.get_pos()
    x_max = screen.get_width()
    y_max = screen.get_height()
    if x < 0 or y < 0:
        return True
    if x > x_max or y > y_max:
        return True
    return False


class MidiAnimator:
    def __init__(self, midifilepath, screen, keys_for_c=[]):
        self.mid = mido.MidiFile(midifilepath)
        self.screen = screen
        self.objects: list[AnimationObject] = []
        self.tickrate_ms = 20
        self.dx = -1
        self.dy = 0
        self.keys_for_c = keys_for_c
        self.tick = 0
    def create_listener(self):
        self.listener = Listener(
            on_release=self.on_release)
        with self.listener as listener:
            listener.join()
    def on_release(self, key):
        try:
            if key.char == "R":
                self.tick = 0
        except:
            pass


    def move_objects(self):
        objects = self.objects.copy()
        for obj in objects:
            obj.move_pos(self.dx, self.dy)
            if out_of_bounds(obj, self.screen):
                self.objects.remove(obj)

    def animate(self):
        events = []
        t = 0
        events.append({"time": t})
        for msg in self.mid:
            if not msg.type == "note_on":
                continue
            note = int(msg.note)

            if msg.time > 0:
                t += msg.time
                events.append({"time": t})
                if msg.velocity > 0:
                    events[-1][note] = None
                else:
                    self.write_duration(events, note, t)
                continue

            if msg.velocity > 0:
                # Track duration of Note
                events[-1][note] = None
            if msg.velocity == 0:
                self.write_duration(events, note, t)

        while True:
            if self.tick >= len(events):
                continue
            i = self.tick
            self.tick += 1
            event = events[i]
            event = event.copy()
            del event["time"]

            for note in event:
                if note != max(event.keys()):
                    continue
                duration = event[note]
                rect_height = 5
                y_pos = self.screen.get_height() // 2 - (note - 60) * rect_height * 2
                rect = PyGameRect(self.screen,
                                  abs(duration / (self.tickrate_ms / 1000) * self.dx) - 2,
                                  rect_height,
                                  self.screen.get_width() - 10, y_pos,
                                  (255, 255, 255))
                try:
                    note_text = str(self.keys_for_c[note])  # + "  + " + str(note))
                except IndexError:
                    note_text = f"Not found {note}"
                text = PyGameText(self.screen, note_text, self.screen.get_width() - 10,
                                  y_pos + 2 * rect_height,
                                  (255, 255, 255), None,
                                  20)
                self.objects.append(rect)
                self.objects.append(text)

            now = time.time()
            try:
                next_event = now + events[i + 1]["time"] - events[i]["time"]
            except IndexError:
                while len(self.objects) > 0:
                    time.sleep(self.tickrate_ms / 1000)
                    self.move_objects()
                    self.draw()
                self.tickrate_ms = int(self.tickrate_ms * 1)
                self.mid.ticks_per_beat = int(self.mid.ticks_per_beat / 1)
                return self.animate()
            while next_event > time.time():
                time.sleep(self.tickrate_ms / 1000)
                self.move_objects()
                self.draw()

    def write_duration(self, events, note, t):
        for event in events[-2::-1]:
            if note in event:
                event[note] = t - event["time"]
                break
        else:
            raise Exception("Note to turn of not found")

    def draw(self):
        objects = self.objects.copy()
        for obj in objects:
            obj.draw()


class AnimationFrame:
    # Initialize Pygame
    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.objects = []
        self.screen: Surface | SurfaceType | None = None
        self.Controller = None
        self.tickrate = 60

    def add_element(self, obj: AnimationObject):
        self.objects.append(obj)

    def remove_element(self, obj: AnimationObject):
        self.objects.remove(obj)

    def show(self):
        pygame.init()
        # Set up display
        pygame.display.set_caption("Basic Pygame Frame")
        self.screen = pygame.display.set_mode((self.width, self.height))

    def loop(self):
        # Set up colors
        white = (255, 255, 255)
        black = (0, 0, 0)

        # Game loop
        clock = pygame.time.Clock()
        continue_loop = True

        while continue_loop:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    continue_loop = False
                    pygame.quit()
                    sys.exit()

            # Update game logic here

            # Draw background
            self.screen.fill(black)
            pygame.draw.line(self.screen, white, start_pos=(self.screen.get_width() // 2, 0),
                             end_pos=(self.screen.get_width() // 2, self.screen.get_height()))

            # Draw game elements here
            self.controller.draw()

            # Update display
            pygame.display.flip()

            # Cap the frame rate
            clock.tick(self.tickrate)


def create_alphabetic_shuffle():
    shuffled_alphabet = []
    base = ord("a")
    for i in range(26):
        shuffled_alphabet.append(chr(i + base))

    random.shuffle(shuffled_alphabet)
    return shuffled_alphabet


def is_pentatonic(num: int):
    pentatonic_numbers = [0, 2, 4, 7, 9]
    rest = num % 12
    if rest in pentatonic_numbers:
        return True
    return False


class DrawAudioController:
    def __init__(self, audio_generator: AudioGenerator, screen: Surface):
        self.audio_generator = audio_generator
        self.screen = screen

    def draw(self):
        y_pos = (np.array(self.audio_generator.wave) * 100 + 300).tolist()
        x_pos = np.arange(len(y_pos)).tolist()
        points = list(zip(x_pos, y_pos))

        pygame.draw.aalines(self.screen, (100, 200, 200), False, points)


if __name__ == "__main__":
    main()
