#!/usr/bin/env python3
"""Real-time Japanese voice chat through the Unitree G1 speaker."""

from __future__ import annotations

import argparse
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests

# Support both `python -m speech.unitree_sample_files.g1_voice_chat` and direct use.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech.asr import DEFAULT_MODEL_PATH, JapaneseASR
from speech.realtime_voice import (
    choose_audio_backend,
    enable_venv_cuda_libraries,
    listen_for_utterance,
    parse_device,
)
from speech.tts import JapaneseTTS
from speech.unitree_sample_files.wav import read_wav
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

SPEECH_DIR = Path(__file__).resolve().parents[1]
TTS_MODEL_DIR = SPEECH_DIR / "models" / "tsukuyomi"
TTS_MODEL_PATH = TTS_MODEL_DIR / "tsukuyomi-chan-6lang-fp16.onnx"
TTS_CONFIG_PATH = TTS_MODEL_DIR / "config.json"
STREAM_NAME = "tts"


# Keep the existing Qwen/Ollama LLM unchanged.
def chat(text: str) -> str:
    prompt = (
        "あなたはUnitree G1の音声アシスタントです。"
        "常に日本語の丁寧語で、短く自然に答えてください。"
        "余計な説明はせず、一文だけ答えてください。"
        "英語、中国語、ローマ字、括弧、話者ラベルは絶対に使わないでください。\n"
        f"ユーザー: {text}\nアシスタント:"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen:0.5b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.6, "num_predict": 48},
        },
        timeout=8,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def sanitize_reply_for_tts(text: str) -> str:
    """Remove small-model artifacts that can break Japanese phonemization."""
    # Do not speak a second generated turn or echoed prompt labels.
    text = re.split(r"(?:^|\n)\s*(?:アシスタント|ユーザー)\s*[:：]", text)[0]
    text = re.sub(r"[（(][^()（）]*[A-Za-z][^()（）]*[）)]", "", text)
    text = re.sub(r"[（(]\s*回答\s*[）)]", "", text)
    # Latin text invokes Piper's optional English/NLTK phonemizer. This robot
    # assistant is Japanese-only, so discard it rather than crashing a turn.
    text = re.sub(r"[A-Za-z]+(?:[ '\-][A-Za-z]+)*", "", text)
    text = re.sub(r"[\r\n\t]+", "。", text)
    text = re.sub(r"。{2,}", "。", text)
    text = re.sub(r"\s+", "", text).strip("。 、,")
    return text or "すみません、もう一度お願いします。"


def check_voice_service(client: AudioClient, interface: str) -> None:
    code, volume = client.GetVolume()
    if code == 3102:
        raise RuntimeError(
            f"Unitree voice service was not found on interface '{interface}'"
        )
    if code != 0:
        raise RuntimeError(f"Unitree voice-service check failed: SDK code {code}")
    print(f"Connected to G1 voice service (volume: {volume})")


def make_g1_wav(tts: JapaneseTTS, text: str, output_path: Path) -> None:
    """Synthesize locally and normalize to G1's 16-kHz mono PCM format."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to prepare Piper output for G1")

    with tempfile.TemporaryDirectory(prefix="g1-piper-") as directory:
        piper_wav = Path(directory) / "piper.wav"
        tts.synthesize(text, piper_wav)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(piper_wav),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
        )


def play_g1_interruptibly(
    client: AudioClient,
    pcm: list[int],
    interrupted: threading.Event,
    playback_tail_seconds: float,
) -> bool:
    """Stream short PCM chunks and stop immediately when barge-in occurs."""
    pcm_data = bytes(pcm)
    stream_id = str(int(time.time() * 1000))
    bytes_per_second = 16000 * 1 * 2
    # Match Unitree's working sample: larger chunks avoid choppy DDS playback.
    # Barge-in remains immediate because the listener calls PlayStop directly.
    chunk_size = 96000  # Three seconds of 16-kHz mono PCM.
    started = time.monotonic()

    try:
        for offset in range(0, len(pcm_data), chunk_size):
            if interrupted.is_set():
                return False
            chunk = pcm_data[offset : offset + chunk_size]
            code, _ = client.PlayStream(STREAM_NAME, stream_id, chunk)
            if code != 0:
                raise RuntimeError(f"G1 PlayStream failed with SDK code {code}")
            if interrupted.wait(min(1.0, len(chunk) / bytes_per_second)):
                return False

        remaining = len(pcm_data) / bytes_per_second - (time.monotonic() - started)
        if interrupted.wait(max(0.0, remaining) + playback_tail_seconds):
            return False
        return True
    finally:
        client.PlayStop(STREAM_NAME)


def start_listener(
    client: AudioClient,
    args: argparse.Namespace,
    audio_backend: str,
    threshold: float,
    interrupt_playback: bool = False,
) -> tuple[threading.Thread, queue.Queue]:
    """Start capture now; on speech onset, interrupt any G1 TTS stream."""
    result: queue.Queue = queue.Queue(maxsize=1)

    def on_speech_start() -> None:
        if not interrupt_playback:
            return
        args.interrupted.set()
        try:
            client.PlayStop(STREAM_NAME)
        except Exception as error:
            # Keep capturing the user's utterance even if a duplicate stop call
            # races with the playback thread's own cleanup.
            print(f"G1 playback stop warning: {error}", file=sys.stderr)

    def run() -> None:
        try:
            audio = listen_for_utterance(
                args.input_device_value,
                args.sample_rate,
                threshold,
                args.silence_seconds,
                args.min_speech_seconds,
                args.max_speech_seconds,
                audio_backend,
                on_speech_start=on_speech_start,
            )
            result.put((audio, None))
        except BaseException as error:
            result.put((None, error))

    thread = threading.Thread(target=run, name="g1-microphone", daemon=True)
    thread.start()
    return thread, result


def wait_for_audio(result: queue.Queue) -> np.ndarray:
    while True:
        try:
            audio, error = result.get(timeout=0.2)
            if error is not None:
                raise error
            return audio
        except queue.Empty:
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time local ASR/TTS chat through a Unitree G1"
    )
    parser.add_argument("net", help="Network interface connected to G1")
    parser.add_argument("--input-device", help="Microphone index or name")
    parser.add_argument(
        "--audio-backend",
        choices=["auto", "sounddevice", "pulse"],
        default="auto",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--threshold", type=float, default=0.015)
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="Listen during G1 speech (requires directional mic or echo cancellation)",
    )
    parser.add_argument(
        "--barge-in-threshold",
        type=float,
        default=0.04,
        help="Higher playback-time threshold to reduce G1 speaker echo triggers",
    )
    parser.add_argument(
        "--playback-tail-seconds",
        type=float,
        default=0.8,
        help="Wait after the expected audio duration before PlayStop",
    )
    parser.add_argument("--silence-seconds", type=float, default=0.8)
    parser.add_argument("--min-speech-seconds", type=float, default=0.35)
    parser.add_argument("--max-speech-seconds", type=float, default=15.0)
    parser.add_argument("--asr-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--asr-device", choices=["cpu", "cuda", "auto"], default="cpu"
    )
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--tts-model", type=Path, default=TTS_MODEL_PATH)
    parser.add_argument("--tts-config", type=Path, default=TTS_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_venv_cuda_libraries(args.asr_device)
    args.input_device_value = parse_device(args.input_device)
    args.interrupted = threading.Event()
    audio_backend = choose_audio_backend(
        args.audio_backend, args.input_device_value, args.sample_rate
    )

    print(f"Audio backend: {audio_backend}")
    print("Loading local Kotoba Whisper ASR...", flush=True)
    asr = JapaneseASR(args.asr_model, args.asr_device, args.compute_type)
    print("Loading local Piper Tsukuyomi TTS...", flush=True)
    tts = JapaneseTTS(args.tts_model, args.tts_config)

    ChannelFactoryInitialize(0, args.net)
    client = AudioClient()
    client.SetTimeout(10.0)
    client.Init()
    check_voice_service(client, args.net)

    print("G1 Voice Chat Ready (Ctrl+C to exit)")
    _, listener_result = start_listener(
        client, args, audio_backend, args.threshold
    )

    try:
        while True:
            audio = wait_for_audio(listener_result)
            print("Recognizing...", flush=True)
            user_text = asr.transcribe(audio)
            if not user_text:
                print("No speech recognized.")
                args.interrupted.clear()
                _, listener_result = start_listener(
                    client, args, audio_backend, args.threshold
                )
                continue

            print(f"User: {user_text}")
            try:
                raw_reply = chat(user_text)
            except (requests.RequestException, KeyError, ValueError) as error:
                print(f"LLM request failed; continuing: {error}", file=sys.stderr)
                args.interrupted.clear()
                _, listener_result = start_listener(
                    client, args, audio_backend, args.threshold
                )
                continue

            reply = sanitize_reply_for_tts(raw_reply)
            print(f"Robot: {reply}")
            if reply != raw_reply:
                print(f"Raw LLM output was sanitized: {raw_reply!r}")

            args.interrupted.clear()
            next_result = None
            if args.barge_in:
                _, next_result = start_listener(
                    client,
                    args,
                    audio_backend,
                    args.barge_in_threshold,
                    interrupt_playback=True,
                )

            with tempfile.TemporaryDirectory(prefix="g1-reply-") as directory:
                wav_path = Path(directory) / "reply.wav"
                try:
                    make_g1_wav(tts, reply, wav_path)
                except Exception as error:
                    # A malformed phoneme sequence should skip only this reply,
                    # never terminate the long-running robot interaction.
                    print(f"TTS failed; continuing: {error}", file=sys.stderr)
                    if next_result is None:
                        _, next_result = start_listener(
                            client, args, audio_backend, args.threshold
                        )
                    listener_result = next_result
                    continue
                if not args.interrupted.is_set():
                    pcm, sample_rate, channels, ok = read_wav(str(wav_path))
                    if not ok or sample_rate != 16000 or channels != 1:
                        raise RuntimeError("Invalid G1 WAV generated by local TTS")
                    completed = play_g1_interruptibly(
                        client,
                        pcm,
                        args.interrupted,
                        args.playback_tail_seconds,
                    )
                    if not completed:
                        print("Robot speech interrupted by user.")

            if next_result is None:
                _, next_result = start_listener(
                    client, args, audio_backend, args.threshold
                )
            listener_result = next_result

    except KeyboardInterrupt:
        client.PlayStop(STREAM_NAME)
        print("\nExit.")


if __name__ == "__main__":
    main()
