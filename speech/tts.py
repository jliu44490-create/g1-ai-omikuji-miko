"""Piper text-to-speech and playback helpers."""

from __future__ import annotations

import tempfile
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Optional, Union

import sounddevice as sd
import soundfile as sf
from piper import PiperVoice

Device = Optional[Union[int, str]]


class JapaneseTTS:
    """Load a Piper voice once, then synthesize and play Japanese text."""

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        output_device: Device = None,
        use_cuda: bool = False,
        audio_backend: str = "sounddevice",
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Piper model not found: {model_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"Piper config not found: {config_path}")

        self.voice = PiperVoice.load(
            model_path=model_path,
            config_path=config_path,
            use_cuda=use_cuda,
        )
        self.output_device = output_device
        self.audio_backend = audio_backend

    def synthesize(self, text: str, output_path: Path) -> None:
        """Write synthesized speech to a WAV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav_file:
            self.voice.synthesize(
                text=text,
                wav_file=wav_file,
                length_scale=1.5,
                noise_scale=0.667,
                sentence_silence=0.15,
            )

    def speak(self, text: str) -> None:
        """Synthesize text and block until speaker playback completes."""
        with tempfile.TemporaryDirectory(prefix="miko-tts-") as directory:
            path = Path(directory) / "speech.wav"
            self.synthesize(text, path)
            if self.audio_backend == "pulse":
                paplay = shutil.which("paplay")
                if paplay is None:
                    raise RuntimeError(
                        "paplay is not installed (install pulseaudio-utils)"
                    )
                command = [paplay]
                if self.output_device is not None:
                    command.append(f"--device={self.output_device}")
                command.append(str(path))
                subprocess.run(command, check=True)
                return

            audio, sample_rate = sf.read(path, dtype="float32")
            sd.play(audio, sample_rate, device=self.output_device, blocking=True)
