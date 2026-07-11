#!/usr/bin/env python3
"""Real-time Japanese voice chat through the Unitree G1 speaker."""

from __future__ import annotations

import argparse
import queue
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# Support both `python -m speech.unitree_sample_files.g1_voice_chat` and direct use.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import generate_reading
from g1_control.g1_move_to_symmetric_审判硬腰_real import (
    WAIST_KD_HARD,
    WAIST_KP_HARD,
    RealG1ArmController,
)
from speech.asr import DEFAULT_MODEL_PATH, JapaneseASR
from speech.realtime_voice import (
    choose_audio_backend,
    enable_venv_cuda_libraries,
    listen_for_utterance,
    parse_device,
)
from speech.unitree_sample_files.speak_japanese import (
    VOICE as EDGE_TTS_VOICE,
    generate_wav as generate_edge_wav,
)
from speech.unitree_sample_files.wav import read_wav
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

STREAM_NAME = "tts"
FIXED_OMIKUJI_COLOR = "blue"


class PoseSession:
    """Run one pose/hold/return lifecycle without blocking LLM or TTS."""

    def __init__(self, controller: RealG1ArmController) -> None:
        self.controller = controller
        self.release_event = threading.Event()
        self.result: queue.Queue = queue.Queue(maxsize=1)
        self._result_collected = False
        self._error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="g1-judgment-pose",
            daemon=True,
        )

    def _run(self) -> None:
        try:
            self.controller.run_until_released(self.release_event)
            self.result.put(None)
        except BaseException as error:
            self.result.put(error)

    def start(self) -> None:
        self.thread.start()

    def release_and_wait(self) -> None:
        """Request standing and wait until Arm SDK control is released."""
        self.release_event.set()
        self.thread.join()
        if not self._result_collected:
            self._error = self.result.get()
            self._result_collected = True
        if self._error is not None:
            raise RuntimeError(
                f"G1 pose lifecycle failed: {self._error}"
            ) from self._error


def chat(text: str) -> str:
    """Generate a miko reading using the shared LLM module and fixed blue color."""
    return generate_reading(text, FIXED_OMIKUJI_COLOR)


def check_voice_service(client: AudioClient, interface: str) -> None:
    code, volume = client.GetVolume()
    if code == 3102:
        raise RuntimeError(
            f"Unitree voice service was not found on interface '{interface}'"
        )
    if code != 0:
        raise RuntimeError(f"Unitree voice-service check failed: SDK code {code}")
    print(f"Connected to G1 voice service (volume: {volume})")


def make_g1_wav(text: str, output_path: Path) -> None:
    """Generate G1-ready audio with Edge TTS ja-JP-NanamiNeural."""
    generate_edge_wav(text, str(output_path))


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
    parser.add_argument(
        "--waist-kp",
        type=float,
        default=WAIST_KP_HARD,
        help="Judgment-pose waist stiffness; lower it if the robot shakes",
    )
    parser.add_argument(
        "--waist-kd",
        type=float,
        default=WAIST_KD_HARD,
        help="Judgment-pose waist damping",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_venv_cuda_libraries(args.asr_device)
    print("WARNING: judgment pose control is only for a 29-DOF G1.")
    print(
        "WARNING: clear the arm/waist area, use physical protection, and keep "
        "the E-stop ready. Hard waist holding can fight balance."
    )
    if input("Type YES to enable real-robot pose control: ").strip() != "YES":
        print("Cancelled before robot control was initialized.")
        return

    args.input_device_value = parse_device(args.input_device)
    args.interrupted = threading.Event()
    audio_backend = choose_audio_backend(
        args.audio_backend, args.input_device_value, args.sample_rate
    )

    print(f"Audio backend: {audio_backend}")
    print("Loading local Kotoba Whisper ASR...", flush=True)
    asr = JapaneseASR(args.asr_model, args.asr_device, args.compute_type)
    if shutil.which("edge-tts") is None:
        raise RuntimeError("edge-tts is required; install it with 'pip install edge-tts'")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert Edge TTS audio for G1")
    print(
        f"LLM: miko reading (fixed omikuji color: {FIXED_OMIKUJI_COLOR})"
    )
    print(f"TTS: Microsoft Edge {EDGE_TTS_VOICE}")

    ChannelFactoryInitialize(0, args.net)
    client = AudioClient()
    client.SetTimeout(10.0)
    client.Init()
    check_voice_service(client, args.net)
    pose_controller = RealG1ArmController(args.waist_kp, args.waist_kd)

    print("G1 Voice Chat Ready (Ctrl+C to exit)")
    _, listener_result = start_listener(
        client, args, audio_backend, args.threshold
    )

    active_pose: PoseSession | None = None
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
            # Start moving at the same moment this recognized text is sent to
            # Claude. The background session holds TARGET through TTS playback.
            active_pose = PoseSession(pose_controller)
            active_pose.start()
            try:
                raw_reply = chat(user_text)
            except Exception as error:
                print(f"Miko LLM failed; continuing: {error}", file=sys.stderr)
                active_pose.release_and_wait()
                active_pose = None
                args.interrupted.clear()
                _, listener_result = start_listener(
                    client, args, audio_backend, args.threshold
                )
                continue

            reply = raw_reply.strip()
            if not reply:
                print("Miko LLM returned an empty response; continuing.", file=sys.stderr)
                active_pose.release_and_wait()
                active_pose = None
                args.interrupted.clear()
                _, listener_result = start_listener(
                    client, args, audio_backend, args.threshold
                )
                continue
            print(f"Robot: {reply}")

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
                    make_g1_wav(reply, wav_path)
                except Exception as error:
                    print(f"Edge TTS failed; continuing: {error}", file=sys.stderr)
                    active_pose.release_and_wait()
                    active_pose = None
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

            # Playback (including its tail buffer) is complete. Only now return
            # from the judgment pose to the measured standing arm position.
            active_pose.release_and_wait()
            active_pose = None

            if next_result is None:
                _, next_result = start_listener(
                    client, args, audio_backend, args.threshold
                )
            listener_result = next_result

    except KeyboardInterrupt:
        client.PlayStop(STREAM_NAME)
        print("\nExit.")
    finally:
        if active_pose is not None:
            active_pose.release_and_wait()


if __name__ == "__main__":
    main()
