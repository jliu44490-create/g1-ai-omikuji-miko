"""
Integration test: Microphone → ASR → LLM
Record voice → Whisper transcription → Miko reading generation

Usage:
    python3 test_asr_llm.py
    python3 test_asr_llm.py --color red
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from llm.miko import generate_reading

SAMPLE_RATE = 16000
CHANNELS = 1
RECORDING_PATH = Path("speech/audio/test_recording.wav")

audio_queue: queue.Queue = queue.Queue()
stop_event = threading.Event()


def mic_callback(indata, frames, time_info, status):
    if status:
        print(f"  audio warning: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


def write_audio(path, sample_rate, channels):
    with sf.SoundFile(str(path), mode="w", samplerate=sample_rate,
                      channels=channels, subtype="PCM_16", format="WAV") as f:
        while not stop_event.is_set() or not audio_queue.empty():
            try:
                block = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            f.write(block)
            audio_queue.task_done()


def record():
    RECORDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    stop_event.clear()
    while not audio_queue.empty():
        audio_queue.get_nowait()

    writer = threading.Thread(target=write_audio,
                              args=(RECORDING_PATH, SAMPLE_RATE, CHANNELS))
    writer.start()

    input("Enter を押すと録音開始...")
    print("録音中... もう一度 Enter で停止")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", callback=mic_callback):
        input()

    stop_event.set()
    writer.join()
    print(f"録音保存: {RECORDING_PATH}")


def transcribe():
    print("\n音声認識中...")
    from faster_whisper import WhisperModel
    t0 = time.time()
    model = WhisperModel("kotoba-tech/kotoba-whisper-v2.0-faster")
    segments, info = model.transcribe(
        str(RECORDING_PATH), language="ja",
        chunk_length=15, condition_on_previous_text=False,
    )
    text = "".join(seg.text for seg in segments).strip()
    elapsed = time.time() - t0
    print(f"認識結果 ({elapsed:.1f}s): {text}")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", default=None,
                        choices=["gold", "red", "blue"])
    args = parser.parse_args()

    print("=== Integration Test: Mic → ASR → LLM ===\n")

    # Step 1: Record
    record()

    # Step 2: ASR
    text = transcribe()
    if not text:
        print("認識できませんでした。もう一度お試しください。")
        return

    # Step 3: Color
    color = args.color
    if not color:
        color = input("\n色を選んでください (gold / red / blue): ").strip().lower()
        if color not in ("gold", "red", "blue"):
            color = "gold"

    # Step 4: LLM
    print(f"\n巫女が言葉を紡いでいます... (color={color})")
    t0 = time.time()
    reading = generate_reading(text, color)
    elapsed = time.time() - t0

    print(f"\n--- 巫女の言葉 ({elapsed:.1f}s) ---\n")
    print(reading)
    print()


if __name__ == "__main__":
    main()
