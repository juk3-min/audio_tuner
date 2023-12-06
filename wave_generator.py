from typing import Protocol
class WaveGenerator(Protocol):
    def create_wave(self):
        ...

    def pitch(self, factor: float):
        ...
