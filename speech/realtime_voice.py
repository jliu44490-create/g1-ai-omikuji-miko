#!/usr/bin/env python3
"""Continuous microphone -> ASR -> response -> TTS demonstration."""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import signal
import subprocess
import sys
import sysconfig
from collections import deque
from pathlib import Path
from typing import Optional, Union

import numpy as np
import sounddevice as sd

try:
    from .asr import JapaneseASR
    from .tts import JapaneseTTS
except ImportError:  # Allow `python speech/realtime_voice.py`.
    from asr import JapaneseASR
    from tts import JapaneseTTS

SPEECH_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SPEECH_DIR / "models"
DEFAULT_ASR_MODEL_DIR = DEFAULT_MODEL_DIR / "kotoba-whisper-v2.0-faster"
DEFAULT_TTS_MODEL_DIR = DEFAULT_MODEL_DIR / "tsukuyomi"


def enable_venv_cuda_libraries(asr_device: str) -> None:
    """Restart once with pip-installed cuBLAS/cuDNN on the loader path."""
    if asr_device not in ("cuda", "auto"):
        return
    if os.environ.get("MIKO_CUDA_PATH_READY") == "1":
        return

    site_packages = Path(sysconfig.get_paths()["purelib"])
    candidates = [
        site_packages / "nvidia" / "cublas" / "lib",
        site_packages / "nvidia" / "cudnn" / "lib",
    ]
    cuda_paths = [str(path) for path in candidates if path.is_dir()]
    if not cuda_paths:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        cuda_paths + ([existing] if existing else [])
    )
    environment["MIKO_CUDA_PATH_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)


def parse_device(value: Optional[str]) -> Optional[Union[int, str]]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def listen_for_utterance(
    device: Optional[Union[int, str]],
    sample_rate: int,
    threshold: float,
    silence_seconds: float,
    min_speech_seconds: float,
    max_speech_seconds: float,
    audio_backend: str = "sounddevice",
) -> np.ndarray:
    """Capture one utterance using a simple RMS energy/silence detector."""
    block_seconds = 0.03
    block_size = round(sample_rate * block_seconds)
    pre_roll = deque(maxlen=max(1, round(0.3 / block_seconds)))
    blocks: list[np.ndarray] = []
    speaking = False
    silent_blocks = 0

    needed_silence = max(1, round(silence_seconds / block_seconds))
    min_blocks = max(1, round(min_speech_seconds / block_seconds))
    max_blocks = max(1, round(max_speech_seconds / block_seconds))

    def process_blocks(block_iterator) -> np.ndarray:
        nonlocal speaking, silent_blocks
        for block in block_iterator:
            rms = float(np.sqrt(np.mean(np.square(block))))

            if not speaking:
                pre_roll.append(block)
                if rms >= threshold:
                    speaking = True
                    blocks.extend(pre_roll)
                    pre_roll.clear()
                    print("Speech detected.", flush=True)
                continue

            blocks.append(block)
            silent_blocks = silent_blocks + 1 if rms < threshold else 0
            speech_blocks = len(blocks)
            if speech_blocks >= max_blocks:
                break
            if speech_blocks >= min_blocks and silent_blocks >= needed_silence:
                break
        return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)

    print("Listening...", flush=True)
    if audio_backend == "pulse":
        parec = shutil.which("parec")
        if parec is None:
            raise RuntimeError("parec is not installed (install pulseaudio-utils)")
        pulse_device = str(device) if device is not None else "@DEFAULT_SOURCE@"
        command = [
            parec,
            "--record",
            f"--device={pulse_device}",
            f"--rate={sample_rate}",
            "--format=float32le",
            "--channels=1",
            "--raw",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )

        def pulse_blocks():
            assert process.stdout is not None
            byte_count = block_size * np.dtype(np.float32).itemsize
            pending = bytearray()
            while True:
                data = process.stdout.read(byte_count - len(pending))
                if not data:
                    break
                pending.extend(data)
                if len(pending) == byte_count:
                    yield np.frombuffer(bytes(pending), dtype="<f4")
                    pending.clear()

        try:
            return process_blocks(pulse_blocks())
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=2)

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"Audio warning: {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    def sounddevice_blocks():
        while True:
            yield audio_queue.get()

    with sd.InputStream(
        device=device,
        channels=1,
        samplerate=sample_rate,
        dtype="float32",
        blocksize=block_size,
        callback=callback,
    ):
        return process_blocks(sounddevice_blocks())


def choose_audio_backend(requested: str, input_device, sample_rate: int = 16000) -> str:
    """Select PulseAudio on WSLg when PortAudio cannot see an input."""
    if requested != "auto":
        return requested
    try:
        sd.check_input_settings(
            device=input_device, channels=1, samplerate=sample_rate, dtype="float32"
        )
        return "sounddevice"
    except Exception:
        if os.environ.get("PULSE_SERVER") and shutil.which("parec"):
            return "pulse"
        return "sounddevice"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously recognize microphone speech and speak a response."
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device", help="Input device index or name")
    parser.add_argument("--output-device", help="Output device index or name")
    parser.add_argument(
        "--audio-backend",
        choices=["auto", "sounddevice", "pulse"],
        default="auto",
        help="Auto selects WSLg PulseAudio when PortAudio has no input device",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.015,
        help="Speech RMS threshold (lower this if speech is not detected)",
    )
    parser.add_argument("--silence-seconds", type=float, default=0.8)
    parser.add_argument("--min-speech-seconds", type=float, default=0.35)
    parser.add_argument("--max-speech-seconds", type=float, default=15.0)
    parser.add_argument(
        "--asr-model",
        type=Path,
        default=DEFAULT_ASR_MODEL_DIR,
        help="Local CTranslate2 ASR model directory",
    )
    parser.add_argument(
        "--asr-device",
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="ASR inference device. CPU is the portable default.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="CTranslate2 compute type; int8 is recommended for CPU",
    )
    parser.add_argument(
        "--tts-model",
        type=Path,
        default=DEFAULT_TTS_MODEL_DIR / "tsukuyomi-chan-6lang-fp16.onnx",
    )
    parser.add_argument(
        "--tts-config", type=Path, default=DEFAULT_TTS_MODEL_DIR / "config.json"
    )
    parser.add_argument(
        "--response-template",
        default="{text}",
        help="Text spoken after recognition; {text} is replaced with the transcript",
    )
    parser.add_argument("--no-speak", action="store_true", help="ASR-only debugging")
    parser.add_argument("--once", action="store_true", help="Process one utterance and exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_devices:
        print("PortAudio devices:")
        print(sd.query_devices())
        if shutil.which("pactl") and os.environ.get("PULSE_SERVER"):
            print("\nPulseAudio sources and sinks:", flush=True)
            subprocess.run(["pactl", "list", "short", "sources"], check=False)
            subprocess.run(["pactl", "list", "short", "sinks"], check=False)
        return

    enable_venv_cuda_libraries(args.asr_device)

    input_device = parse_device(args.input_device)
    output_device = parse_device(args.output_device)
    audio_backend = choose_audio_backend(
        args.audio_backend, input_device, args.sample_rate
    )
    if audio_backend == "sounddevice":
        sd.check_input_settings(
            device=input_device,
            channels=1,
            samplerate=args.sample_rate,
            dtype="float32",
        )
    print(f"Audio backend: {audio_backend}")

    print("Loading ASR model...", flush=True)
    asr = JapaneseASR(args.asr_model, args.asr_device, args.compute_type)
    tts = None
    if not args.no_speak:
        print("Loading TTS model...", flush=True)
        tts = JapaneseTTS(
            args.tts_model,
            args.tts_config,
            output_device,
            audio_backend=audio_backend,
        )

    print("Ready. Press Ctrl+C to stop.")
    try:
        while True:
            audio = listen_for_utterance(
                input_device,
                args.sample_rate,
                args.threshold,
                args.silence_seconds,
                args.min_speech_seconds,
                args.max_speech_seconds,
                audio_backend,
            )
            print("Recognizing...", flush=True)
            try:
                text = asr.transcribe(audio)
            except RuntimeError as error:
                if "libcublas" in str(error).lower():
                    raise RuntimeError(
                        "CUDA ASR was selected, but the CUDA 12 cuBLAS runtime is "
                        "unavailable. Run with '--asr-device cpu "
                        "--compute-type int8', or install the matching CUDA 12 "
                        "runtime in WSL."
                    ) from error
                raise
            if text:
                print(f"Recognized: {text}")
                if tts is not None:
                    response = args.response_template.format(text=text)
                    print(f"Speaking: {response}")
                    tts.speak(response)
            else:
                print("No speech recognized.")

            if args.once:
                break
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
