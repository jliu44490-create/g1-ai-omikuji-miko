"""Automatic speech recognition used by the speech demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "kotoba-whisper-v2.0-faster"


class JapaneseASR:
    """Load Whisper once and transcribe mono 16 kHz audio."""

    def __init__(
        self,
        model_name: str | Path = DEFAULT_MODEL_PATH,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        model_path = Path(model_name).expanduser()
        if model_path.exists() and not (model_path / "model.bin").is_file():
            raise FileNotFoundError(f"ASR model.bin not found in: {model_path}")

        self.model = WhisperModel(
            str(model_path) if model_path.exists() else str(model_name),
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio: np.ndarray, language: str = "ja") -> str:
        """Return recognized text from float32 waveform samples."""
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        segments, _ = self.model.transcribe(
            samples,
            language=language,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
