#!/usr/bin/env python3
"""Play a WAV file (or generated Japanese speech) through a Unitree G1."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

try:
    from .wav import play_pcm_stream, read_wav
except ImportError:  # Allow direct `python speak_japanese.py` execution.
    from wav import play_pcm_stream, read_wav


VOICE = "ja-JP-NanamiNeural"
DEFAULT_WAV = Path(__file__).resolve().parents[1] / "audio" / "output.wav"


def generate_wav(text: str, wav_path: str) -> None:
    """Generate a 16-kHz mono WAV using the optional edge-tts CLI."""
    if shutil.which("edge-tts") is None:
        raise RuntimeError(
            "edge-tts is not installed. Install it with 'pip install edge-tts', "
            "or play an existing recording with --wav."
        )
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert edge-tts output to WAV")

    mp3_path = wav_path.replace(".wav", ".mp3")
    try:
        subprocess.run(
            ["edge-tts", "--voice", VOICE, "--text", text,
             "--write-media", mp3_path],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path,
             "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", wav_path],
            check=True,
        )
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


def normalize_wav(source: Path, destination: str) -> None:
    """Convert input into the 16-kHz, mono, signed-16-bit format used by G1."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"{source} is not 16-kHz mono PCM; install ffmpeg to convert it"
        )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
         "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", destination],
        check=True,
    )


def speak_wav(audio_client: AudioClient, wav_path: Path) -> None:
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    temporary_path = None
    pcm, sample_rate, channels, ok = read_wav(str(wav_path))
    if not ok:
        raise RuntimeError(f"Could not read PCM WAV: {wav_path}")

    try:
        if sample_rate != 16000 or channels != 1:
            temporary_path = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ).name
            print(
                f"Converting {sample_rate} Hz/{channels} channel(s) "
                "to 16000 Hz/mono..."
            )
            normalize_wav(wav_path, temporary_path)
            pcm, sample_rate, channels, ok = read_wav(temporary_path)
            if not ok:
                raise RuntimeError("Could not read the converted WAV")

        print(f"Playing {wav_path} ({sample_rate} Hz, {channels} channel)")
        play_pcm_stream(audio_client, pcm, "tts")
        audio_client.PlayStop("tts")
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def speak_text(audio_client: AudioClient, text: str) -> None:
    temporary_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        generate_wav(text, temporary_path)
        speak_wav(audio_client, Path(temporary_path))
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def check_voice_service(audio_client: AudioClient, interface: str) -> None:
    """Fail early when DDS cannot discover the G1 voice service."""
    code, volume = audio_client.GetVolume()
    if code == 3102:
        raise RuntimeError(
            "Unitree voice service was not discovered on interface "
            f"'{interface}' (SDK error 3102). Check that this is the wired "
            "robot-facing interface, that it has the robot-subnet IPv4 address, "
            "and that multicast/firewall settings allow CycloneDDS traffic."
        )
    if code != 0:
        raise RuntimeError(f"Unitree voice-service check failed with SDK code {code}")
    print(f"Connected to Unitree voice service (volume: {volume})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Japanese speech or an existing WAV through a Unitree G1."
    )
    parser.add_argument("network_interface", help="Interface connected to G1, e.g. enp7s0f1")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--wav", type=Path, help=f"WAV to play (default: {DEFAULT_WAV})")
    source.add_argument("--text", help="Japanese text to synthesize with edge-tts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ChannelFactoryInitialize(0, args.network_interface)

    client = AudioClient()
    client.SetTimeout(10.0)
    client.Init()
    check_voice_service(client, args.network_interface)

    if args.text is not None:
        speak_text(client, args.text)
    else:
        speak_wav(client, args.wav or DEFAULT_WAV)


if __name__ == "__main__":
    main()
