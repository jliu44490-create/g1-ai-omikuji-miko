#!/usr/bin/env python3
import sys
import time
import cv2
import numpy as np
import asyncio
import subprocess
import tempfile
import os

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
try:
    from .unitree_sample_files.wav import read_wav, play_pcm_stream
except ImportError:  # Allow direct `python speech/g1_color_speak.py` execution.
    from unitree_sample_files.wav import read_wav, play_pcm_stream

VOICE = "ja-JP-NanamiNeural"
COOLDOWN = 3.0  # 同一颜色最少间隔 3 秒

# ============ 颜色 → 日文 ============
COLOR_TEXT = {
    "red":   "これは赤です",
    "gold":  "これは金色です",
    "blue":  "これは青です",
}

# ============ HSV（收紧版）============
HSV_PARAMS = {
    "red": (
        (np.array([0, 150, 150]), np.array([8, 255, 255])),
        (np.array([172, 150, 150]), np.array([180, 255, 255])),
    ),
    "gold": (
        (np.array([16, 200, 180]), np.array([22, 255, 240])),
    ),
    "blue": (
        (np.array([95, 140, 120]), np.array([115, 255, 240])),
    ),
}


# ============ 你自己的 speak 逻辑 ============
async def _generate_mp3(text, mp3_path):
    from edge_tts import Communicate
    communicate = Communicate(text=text, voice=VOICE)
    await communicate.save(mp3_path)


def generate_wav(text, wav_path, debug=False):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
        mp3_path = mp3_tmp.name

    try:
        asyncio.run(_generate_mp3(text, mp3_path))
        subprocess.run([
            "ffmpeg", "-y",
            "-i", mp3_path,
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            wav_path
        ], check=True)
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


def speak(audio_client, text, debug=False):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
        wav_path = wav_tmp.name

    try:
        generate_wav(text, wav_path, debug)
        pcm, sample_rate, channels, ok = read_wav(wav_path)
        if not ok:
            raise RuntimeError("WAV 读取失败")

        duration = len(pcm) / (sample_rate * channels * 2)
        play_pcm_stream(audio_client, pcm, "tts")
        time.sleep(duration + 0.5)
        audio_client.PlayStop("tts")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ============ 颜色识别 ============
def make_mask(hsv, name):
    masks = HSV_PARAMS[name]
    m = cv2.inRange(hsv, masks[0][0], masks[0][1])
    if len(masks) == 2:
        m2 = cv2.inRange(hsv, masks[1][0], masks[1][1])
        m = cv2.bitwise_or(m, m2)
    kernel = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m


def detect_colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result = {}
    for name in COLOR_TEXT:
        mask = make_mask(hsv, name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 1200:
                result[name] = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, name, (x, max(y-5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                continue
        result[name] = False
    return result


def recognize_omikuji_color(
    camera_index=0,
    timeout_seconds=15.0,
    confirmation_frames=5,
    show_preview=False,
):
    """Return one stable `red`, `gold`, or `blue` camera detection.

    A color must be the only detected candidate for several consecutive frames.
    Returns None on timeout and always releases the camera.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"摄像头打开失败: index={camera_index}")

    confirmation_frames = max(1, int(confirmation_frames))
    candidate = None
    consecutive = 0
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))

    try:
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            detected = detect_colors(frame)
            colors = [name for name, found in detected.items() if found]
            current = colors[0] if len(colors) == 1 else None

            if current is not None and current == candidate:
                consecutive += 1
            elif current is not None:
                candidate = current
                consecutive = 1
            else:
                candidate = None
                consecutive = 0

            if show_preview:
                cv2.imshow("G1 Omikuji Color", frame)
                if cv2.waitKey(1) == 27:
                    return None

            if consecutive >= confirmation_frames:
                return candidate
        return None
    finally:
        cap.release()
        if show_preview:
            cv2.destroyWindow("G1 Omikuji Color")


# ============ main ============
def main():
    if len(sys.argv) < 2:
        print("用法: python3 color_speak_g1.py enp7s0f1")
        return

    ChannelFactoryInitialize(0, sys.argv[1])
    audio = AudioClient()
    audio.SetTimeout(10.0)
    audio.Init()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 摄像头打开失败")
        return

    last_spoken = {k: 0.0 for k in COLOR_TEXT}

    print("🎥 颜色识别 + G1 日语播报（ESC 退出）")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        detected = detect_colors(frame)
        now = time.time()

        for cname, ok in detected.items():
            if ok and now - last_spoken[cname] > COOLDOWN:
                print(f"🔊 {COLOR_TEXT[cname]}")
                speak(audio, COLOR_TEXT[cname])
                last_spoken[cname] = now

        cv2.imshow("G1 Color Speak", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
