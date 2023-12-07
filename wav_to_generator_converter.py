from pathlib import Path
import numpy as np
from wave_generator import WaveGenerator
import sounddevice as sd
import soundfile as sf
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
import json

def main():
    path = Path("Violin_open_string.ogg")
    data, sr = read_wav(path)
    play_wav(path, start=0, end=5*sr)

    start = int(sr*2)
    end = int(sr*2.3)

    play_wav(path, start=start, end=end)
    plt.plot(data[start:end,0])
    plt.show()

    N = end-start
    # sample spacing
    T = 1.0 / sr
    base_freq,xf, freq_data = get_base_freq(N, T, data[start:end, 0])
    print(f"{base_freq=}")
    plt.plot(xf, freq_data)
    plt.grid()
    plt.show()

    sample_length = int(sr/base_freq)
    data_small = data[start:end, 0]
    idx_of_zero_pass = get_first_zero_pass(data_small)
    sample = data_small[idx_of_zero_pass:idx_of_zero_pass+sample_length]
    fourier_coff = fourierSeries(sample, 30)

    f_data = reconstruct(sr / base_freq, fourier_coff)
    plt.plot(sample)
    plt.plot(f_data)
    plt.show()
    d = np.tile(f_data, 144*2)
    play_data(d, sample_rate=sr)
    d = np.tile(sample, 144*2)
    play_data(d, sample_rate=sr)
    with open("violin_fourier_coff", "w") as f:
        fourier = [ (a,b) for a,b in fourier_coff]
        json.dump(list(fourier), f)




def get_first_zero_pass(data):
    for i in range(len(data) - 2):
        if data[i] < 0 and data[i + 1] > 0:
            return i



def get_base_freq(N, T, data):
    x = np.linspace(0.0, N * T, N, endpoint=False)
    y = data
    yf = fft(y)
    xf = fftfreq(N, T)[:N // 2]
    freq_data = 2.0 / N * np.abs(yf[0:N // 2])
    max_freq_idx = np.argmax(freq_data)
    base_max_power = freq_data[max_freq_idx]
    max_freq = xf[max_freq_idx]
    base_freq = max_freq
    min_factor = 0.2
    loop = True
    counter = 0
    while loop and counter < 10:
        counter += 1
        sub_max_freq_idx = np.argmax(freq_data[:max_freq_idx])
        sub_max_power = freq_data[sub_max_freq_idx]
        assert sub_max_power != base_max_power
        if sub_max_power / base_max_power > min_factor:
            max_freq_idx = sub_max_freq_idx
            base_freq = xf[max_freq_idx]
        else:
            break
    return base_freq, xf, freq_data


def read_wav(path:Path):
    # Load the sound file
    data, sample_rate = sf.read(path)
    return data, sample_rate

def play_wav(path:Path, start:int=0, end:int=-1):
    # Load the sound file
    data, sample_rate = sf.read(path)
    # Play the sound
    play_data(data[start:end], sample_rate)


def play_data(data, sample_rate):
    sd.play(data, sample_rate)
    sd.wait()

#TODO Song n mal wiederholen
# Einstellungen nach oben ziehen
# Lustig einleiten
# Wie klingt der harmonischeste Ton -- > Simple Generator
# Harmonic Generator
# Zeigen wie perfekt fourier ist
# Bandmember introduction
# Performance

class FourierWaveGenerator(WaveGenerator):
    def __init__(self, freq: float, samples: int, sample_rate: int, anbn, wave_generator, volume=1):
        self.freq = freq
        self.samples = samples
        self.sample_rate = sample_rate
        self.wave_generator = wave_generator
        self.volume = volume
        self.wave_generators = []
        self.anbn = anbn
        for n, (a, b) in enumerate(anbn):
            if n == 0:
                continue
                a = a / 2
            wga = self.wave_generator(n*self.freq, self.samples, self.sample_rate)
            wga.volume = b
            wgb = self.wave_generator(n*self.freq, self.samples, self.sample_rate)
            wgb.signal_function=np.cos
            wgb.volume = a
            self.wave_generators.extend([wga,wgb])

    def create_wave(self):
        output = np.zeros(self.samples)
        for wg in self.wave_generators:
            output += wg.create_wave() * self.volume
        return output

    def pitch(self, factor: float):
        self.freq *= factor
        for wg in self.wave_generators:
            wg.pitch(factor)


def fourierSeries(period, N):
    """Calculate the Fourier series coefficients up to the Nth harmonic"""
    result = []
    T = len(period)
    t = np.arange(T)
    for n in range(N+1):
        an = 2/T*(period * np.cos(2*np.pi*n*t/T)).sum()
        bn = 2/T*(period * np.sin(2*np.pi*n*t/T)).sum()
        result.append((an, bn))
    return np.array(result)

def reconstruct(P, anbn):
    """Sum up sines and cosines according to the coefficients to
    produce a reconstruction of the original waveform"""
    result = 0
    t = np.arange(P)
    for n, (a, b) in enumerate(anbn):
        if n == 0:
            a = a/2
        result = result + a*np.cos(2*np.pi*n*t/P) + b * np.sin(2*np.pi*n*t/P)
    return result

def extract_base_wave(data:np.array):
    ...

def get_fourier_data(data:np.array):
    ...


def generator_to_json(fourier_data):
    ...

if __name__ == "__main__":
    main()