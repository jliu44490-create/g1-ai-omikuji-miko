#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

# 使用官方 wav.py
from wav import read_wav, play_pcm_stream


VOICE = "ja-JP-NanamiNeural"      # 女声
# VOICE = "ja-JP-KeitaNeural"     # 男声


def generate_wav(text, wav_path):
    mp3_path = wav_path.replace(".wav", ".mp3")

    # 1. edge-tts
    subprocess.check_call([
        "edge-tts",
        "--voice", VOICE,
        "--text", text,
        "--write-media", mp3_path
    ])

    # 2. ffmpeg 转 WAV
    subprocess.check_call([
        "ffmpeg",
        "-y",
        "-i", mp3_path,
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        wav_path
    ])

    os.remove(mp3_path)


def speak(audio_client, text):
    tmp_wav = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ).name

    generate_wav(text, tmp_wav)

    pcm, sample_rate, channels, ok = read_wav(tmp_wav)

    if not ok:
        print("读取 WAV 失败")
        return

    print(f"SampleRate={sample_rate}")
    print(f"Channels={channels}")

    play_pcm_stream(audio_client, pcm, "tts")

    audio_client.PlayStop("tts")

    os.remove(tmp_wav)


def main():

    if len(sys.argv) < 3:
        print("Usage:")
        print(f"python3 {sys.argv[0]} enp7s0f1 'こんにちは'")
        return

    net = sys.argv[1]
    text = sys.argv[2]

    ChannelFactoryInitialize(0, net)

    client = AudioClient()
    client.SetTimeout(10.0)
    client.Init()

    speak(client, text)


if __name__ == "__main__":
    main()
