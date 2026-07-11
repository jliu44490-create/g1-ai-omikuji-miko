#!/usr/bin/env python3
import argparse
import asyncio
import os
import subprocess
import time
import requests
import tempfile
import edge_tts

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from wav import read_wav, play_pcm_stream

# ---------- TTS（复用你现有的 speak 逻辑） ----------
VOICE = "ja-JP-NanamiNeural"

def generate_wav(text: str, wav_path: str):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
        mp3_path = mp3_tmp.name

    asyncio.run(
        edge_tts.Communicate(text=text, voice=VOICE).save(mp3_path)
    )

    subprocess.run([
        "ffmpeg", "-y",
        "-i", mp3_path,
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        wav_path
    ], check=True)

    os.remove(mp3_path)


def speak(client: AudioClient, text: str):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
        wav_path = wav_tmp.name

    try:
        generate_wav(text, wav_path)

        pcm, sample_rate, channels, ok = read_wav(wav_path)
        if not ok:
            raise RuntimeError("WAV read failed")

        play_pcm_stream(client, pcm, "tts")

        duration = len(pcm) / (sample_rate * channels * 2)
        time.sleep(duration + 0.5)
        client.PlayStop("tts")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ---------- LLM（Qwen 0.5B） ----------
def chat(text: str) -> str:
    prompt = (
        "あなたはUnitree G1の音声アシスタントです。"
        "常に日本語の丁寧語で、短く自然に答えてください。"
        "余計な説明はせず、一言だけ答えてください。\n"
        f"ユーザー: {text}\nアシスタント:"
    )

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen:0.5b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.6,
                "num_predict": 48
            }
        },
        timeout=8
    )
    return r.json()["response"].strip()


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net")
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.net)
    client = AudioClient()
    client.SetTimeout(10.0)
    client.Init()

    print("G1 Voice Chat Ready（Ctrl+C to exit）")

    while True:
        try:
            user_text = input(">> ").strip()
            if not user_text:
                continue

            reply = chat(user_text)
            print("Robot:", reply)
            speak(client, reply)

        except KeyboardInterrupt:
            print("\nExit.")
            break


if __name__ == "__main__":
    main()