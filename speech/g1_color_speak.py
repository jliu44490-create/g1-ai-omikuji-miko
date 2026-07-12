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
if __package__:
    from .unitree_sample_files.wav import read_wav, play_pcm_stream
else:  # Allow direct `python speech/g1_color_speak.py` execution.
    from unitree_sample_files.wav import read_wav, play_pcm_stream

if __package__:
    from .g1_camera_receiver import G1CameraReceiver
else:
    from g1_camera_receiver import G1CameraReceiver

# ================== 配置区 ==================
VOICE = "ja-JP-NanamiNeural"
CAMERA_IP = "192.168.123.164"
CAMERA_PORT = 55555
PREVIEW_WINDOW = "G1 Color Speak (Center ROI)"

# 中心区域占比（0.6 = 画面中间 60%×60%）
ROI_RATIO = 0.6

# 颜色 → 日语
COLOR_TEXT = {
    "red":   "これは赤です",
    "gold":  "これは金色です",
    "blue":  "これは青です",
}

# HSV（现场级保守参数）
HSV_PARAMS = {
    "red": (
        (np.array([0, 160, 140]), np.array([6, 255, 255])),
        (np.array([174, 160, 140]), np.array([180, 255, 255])),
    ),
    "gold": (
        (np.array([0, 0, 0]), np.array([180, 255, 50])),
    ),
    "blue": (
        (np.array([95, 140, 120]), np.array([115, 255, 240])),
    ),
}
# ==========================================


class ColorRecognitionCancelled(Exception):
    """Raised when the operator presses Escape in the preview window."""


# ================== TTS ==================
async def _generate_mp3(text, mp3_path):
    from edge_tts import Communicate
    communicate = Communicate(text=text, voice=VOICE)
    await communicate.save(mp3_path)


def generate_wav(text, wav_path):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
        mp3_path = mp3_tmp.name
    asyncio.run(_generate_mp3(text, mp3_path))
    subprocess.run([
        "ffmpeg", "-y",
        "-i", mp3_path,
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        wav_path
    ], check=True)
    os.remove(mp3_path)


def speak(audio_client, text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
        wav_path = wav_tmp.name
    generate_wav(text, wav_path)
    pcm, sr, ch, ok = read_wav(wav_path)
    if ok:
        play_pcm_stream(audio_client, pcm, "tts")
        time.sleep(len(pcm) / (sr * ch * 2) + 0.3)
        audio_client.PlayStop("tts")
    os.remove(wav_path)


# ================== 中心区域判断 ==================
def is_in_center_roi(x, y, w, h, img_shape):
    cx = x + w // 2
    cy = y + h // 2

    h_img, w_img = img_shape[:2]
    left   = int(w_img * (1 - ROI_RATIO) / 2)
    right  = int(w_img * (1 + ROI_RATIO) / 2)
    top    = int(h_img * (1 - ROI_RATIO) / 2)
    bottom = int(h_img * (1 + ROI_RATIO) / 2)

    return left < cx < right and top < cy < bottom


# ================== 颜色识别 ==================
def make_mask(hsv, name):
    masks = HSV_PARAMS[name]
    m = cv2.inRange(hsv, masks[0][0], masks[0][1])
    if len(masks) == 2:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, masks[1][0], masks[1][1]))

    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def detect_colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result = {}

    for name in COLOR_TEXT:
        mask = make_mask(hsv, name)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)

            # ===== 通用过滤 =====
            if area < 3000:
                result[name] = False
                continue

            fill_ratio = area / (w * h)
            if fill_ratio < 0.6:
                result[name] = False
                continue

            if not is_in_center_roi(x, y, w, h, img.shape):
                result[name] = False
                continue
            # =====================

            result[name] = True
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, name, (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            continue

        result[name] = False

    # 画中心区域（绿色框）
    h_img, w_img = img.shape[:2]
    left   = int(w_img * (1 - ROI_RATIO) / 2)
    right  = int(w_img * (1 + ROI_RATIO) / 2)
    top    = int(h_img * (1 - ROI_RATIO) / 2)
    bottom = int(h_img * (1 + ROI_RATIO) / 2)
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    return result


def recognize_omikuji_color(
    camera_ip=CAMERA_IP,
    camera_port=CAMERA_PORT,
    timeout_seconds=15.0,
    confirmation_frames=5,
    show_preview=True,
):
    """Return one stable color from the G1 RealSense ZMQ image stream.

    G1CameraReceiver.recv performs the timed socket wait, so no polling sleep is
    needed. When preview is enabled, Escape raises ColorRecognitionCancelled.
    The newest HSV, center-ROI, contour-area, and fill-ratio rules above are
    reused through detect_colors(). Returns None on timeout.
    """
    receiver = G1CameraReceiver(camera_ip, port=int(camera_port))
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    required = max(1, int(confirmation_frames))
    candidate = None
    consecutive = 0

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            image, _meta = receiver.recv(
                timeout_ms=max(1, min(1000, int(remaining * 1000)))
            )
            if image is None:
                continue

            detected = detect_colors(image)
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
                cv2.imshow(PREVIEW_WINDOW, image)
                if cv2.waitKey(1) & 0xFF == 27:
                    raise ColorRecognitionCancelled(
                        "operator pressed Escape in the color preview"
                    )

            if consecutive >= required:
                return candidate
    finally:
        receiver.close()


# ================== main ==================
def main():
    if len(sys.argv) < 2:
        print("用法: python3 g1_color_speak_center.py <network_interface>")
        print("示例: python3 g1_color_speak_center.py enp7s0f1")
        return

    ChannelFactoryInitialize(0, sys.argv[1])

    audio = AudioClient()
    audio.SetTimeout(10.0)
    audio.Init()

    receiver = G1CameraReceiver(CAMERA_IP, port=CAMERA_PORT)

    print("🎥 G1 颜色识别（仅中心区域，识别一次即退出）")
    print("👉 绿色框内为目标识别区")
    print("👉 按 Esc 可手动退出")

    try:
        while True:
            img, meta = receiver.recv()
            if img is None:
                continue

            detected = detect_colors(img)

            spoken = False
            for cname, ok in detected.items():
                if ok:
                    print(f"🔊 {COLOR_TEXT[cname]}")
                    speak(audio, COLOR_TEXT[cname])
                    spoken = True
                    break

            cv2.imshow("G1 Color Speak (Center ROI)", img)

            if spoken:
                print("✅ 已稳定识别一次，程序退出")
                break

            if cv2.waitKey(1) & 0xFF == 27:
                print("👋 手动退出")
                break

    finally:
        receiver.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
