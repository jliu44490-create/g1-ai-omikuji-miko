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
    TARGET as JUDGMENT_TARGET,
    WAIST_KD_HARD,
    WAIST_KP_HARD,
    RealG1ArmController,
)
from g1_control.g1_move_to_symmetric_pose_硬腰 import TARGET as LISTENING_TARGET
from speech.asr import DEFAULT_MODEL_PATH, JapaneseASR
from speech.g1_color_speak import (
    CAMERA_IP,
    CAMERA_PORT,
    COLOR_TEXT,
    ColorRecognitionCancelled,
    recognize_omikuji_color,
)
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
from unitree_sdk2py.g1.arm.g1_arm_action_client import (
    G1ArmActionClient,
    action_map,
)

STREAM_NAME = "tts"
COLOR_REQUEST_TEXT = "おみくじをカメラの前に見せてください。"
DEFAULT_OMIKUJI_COLOR = "gold"


class PoseSession:
    """Run one pose/hold/return lifecycle without blocking LLM or TTS."""

    def __init__(
        self,
        controller: RealG1ArmController,
        target: np.ndarray,
        pose_name: str,
    ) -> None:
        self.controller = controller
        self.target = np.asarray(target, dtype=float)
        self.pose_name = pose_name
        self.release_event = threading.Event()
        self.result: queue.Queue = queue.Queue(maxsize=1)
        self._result_collected = False
        self._error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"g1-{pose_name}-pose",
            daemon=True,
        )

    def _run(self) -> None:
        try:
            self.controller.run_until_released(
                self.release_event,
                target=self.target,
                pose_name=self.pose_name,
            )
            self.result.put(None)
        except BaseException as error:
            self.result.put(error)

    def start(self) -> None:
        self.thread.start()

    def release(self) -> None:
        """Begin the controlled return without blocking the caller."""
        self.release_event.set()

    def wait(self) -> None:
        """Wait until standing is restored and Arm SDK control is released."""
        self.thread.join()
        if not self._result_collected:
            self._error = self.result.get()
            self._result_collected = True
        if self._error is not None:
            raise RuntimeError(
                f"G1 pose lifecycle failed: {self._error}"
            ) from self._error

    def release_and_wait(self) -> None:
        self.release()
        self.wait()


def execute_face_wave(
    client: G1ArmActionClient,
    action_seconds: float,
    settle_seconds: float,
) -> None:
    """Run Unitree's official face-wave action, then release to normal arms."""
    print("Action: face wave...")
    code = client.ExecuteAction(action_map["face wave"])
    if code != 0:
        raise RuntimeError(f"G1 face-wave action failed with SDK code {code}")
    try:
        time.sleep(action_seconds)
    finally:
        code = client.ExecuteAction(action_map["release arm"])
        if code != 0:
            raise RuntimeError(f"G1 arm release failed with SDK code {code}")
    time.sleep(settle_seconds)
    print("Action: face wave complete; normal arms restored.")


def chat(text: str, color: str) -> str:
    """Generate a miko reading from recognized speech and omikuji color."""
    return generate_reading(text, color)


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


def request_omikuji_color(
    client: AudioClient,
    args: argparse.Namespace,
) -> str | None:
    """Play the Japanese request while recognizing color concurrently."""
    with tempfile.TemporaryDirectory(prefix="g1-color-prompt-") as directory:
        wav_path = Path(directory) / "prompt.wav"
        make_g1_wav(COLOR_REQUEST_TEXT, wav_path)
        pcm, sample_rate, channels, ok = read_wav(str(wav_path))
        if not ok or sample_rate != 16000 or channels != 1:
            raise RuntimeError("Invalid color-request WAV generated by Edge TTS")

        playback_result: queue.Queue = queue.Queue(maxsize=1)

        def play_prompt() -> None:
            try:
                play_g1_interruptibly(
                    client,
                    pcm,
                    threading.Event(),
                    args.playback_tail_seconds,
                )
                playback_result.put(None)
            except BaseException as error:
                playback_result.put(error)

        playback_thread = threading.Thread(
            target=play_prompt,
            name="g1-color-prompt",
            daemon=True,
        )
        playback_thread.start()
        try:
            color = recognize_omikuji_color(
                camera_ip=args.camera_ip,
                camera_port=args.camera_port,
                timeout_seconds=args.color_timeout_seconds,
                confirmation_frames=args.color_confirmation_frames,
                show_preview=args.show_color_preview,
            )
        finally:
            playback_thread.join()

        playback_error = playback_result.get()
        if playback_error is not None:
            raise RuntimeError(
                f"G1 color-request playback failed: {playback_error}"
            ) from playback_error
        return color


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


def start_listening_cycle(
    pose_controller: RealG1ArmController,
    audio_client: AudioClient,
    args: argparse.Namespace,
    audio_backend: str,
) -> tuple[PoseSession, queue.Queue]:
    """Enter the listening pose and capture the user's next utterance."""
    session = PoseSession(pose_controller, LISTENING_TARGET, "listening")
    session.start()
    _, result = start_listener(audio_client, args, audio_backend, args.threshold)
    return session, result


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
    parser.add_argument("--asr-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument(
        "--waist-kp",
        type=float,
        default=WAIST_KP_HARD,
        help="Hard-waist pose stiffness; lower it if the robot shakes",
    )
    parser.add_argument(
        "--waist-kd",
        type=float,
        default=WAIST_KD_HARD,
        help="Hard-waist pose damping",
    )
    parser.add_argument(
        "--face-wave-seconds",
        type=float,
        default=4.0,
        help="Time allowed for Unitree's official face-wave action",
    )
    parser.add_argument(
        "--action-settle-seconds",
        type=float,
        default=1.0,
        help="Settling time after releasing the official arm action",
    )
    parser.add_argument(
        "--camera-ip",
        default=CAMERA_IP,
        help="G1 RealSense ZMQ stream IP",
    )
    parser.add_argument(
        "--camera-port",
        type=int,
        default=CAMERA_PORT,
        help="G1 RealSense ZMQ stream port",
    )
    parser.add_argument(
        "--color-timeout-seconds",
        type=float,
        default=15.0,
        help="Maximum time to wait for an omikuji color",
    )
    parser.add_argument(
        "--color-confirmation-frames",
        type=int,
        default=3,
        help="Consecutive frames required to accept one color",
    )
    parser.add_argument(
        "--no-color-preview",
        action="store_false",
        dest="show_color_preview",
        help="Disable the annotated color preview window (enabled by default)",
    )
    parser.set_defaults(show_color_preview=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_venv_cuda_libraries(args.asr_device)
    print("WARNING: custom pose control is only for a 29-DOF G1.")
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
        raise RuntimeError(
            "edge-tts is required; install it with 'pip install edge-tts'"
        )
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert Edge TTS audio for G1")
    print("LLM: miko reading (camera-detected omikuji color)")
    print(f"TTS: Microsoft Edge {EDGE_TTS_VOICE}")

    ChannelFactoryInitialize(0, args.net)
    client = AudioClient()
    client.SetTimeout(10.0)
    client.Init()
    check_voice_service(client, args.net)
    pose_controller = RealG1ArmController(args.waist_kp, args.waist_kd)
    arm_action_client = G1ArmActionClient()
    arm_action_client.SetTimeout(10.0)
    arm_action_client.Init()

    print("G1 Voice Chat Ready (Ctrl+C to exit)")
    execute_face_wave(
        arm_action_client,
        args.face_wave_seconds,
        args.action_settle_seconds,
    )
    active_pose, listener_result = start_listening_cycle(
        pose_controller, client, args, audio_backend
    )

    try:
        while True:
            audio = wait_for_audio(listener_result)
            # Keep holding the listening pose until Whisper has produced a
            # usable sentence. Only then restore standing and transfer arm
            # ownership to the judgment/reading pose.
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
            if active_pose is not None:
                active_pose.release_and_wait()
                active_pose = None

            # Ask for the omikuji only after the listening pose has returned to
            # normal, then feed the stable camera result into the miko LLM.
            try:
                print(f"Robot prompt: {COLOR_REQUEST_TEXT}")
                print("Color recognition: show the omikuji to the camera...")
                omikuji_color = request_omikuji_color(client, args)
            except ColorRecognitionCancelled as error:
                print(f"Color recognition cancelled: {error}")
                raise KeyboardInterrupt from error
            except Exception as error:
                print(f"Color recognition failed: {error}", file=sys.stderr)
                print(
                    f"Using default omikuji color: {DEFAULT_OMIKUJI_COLOR}",
                    file=sys.stderr,
                )
                omikuji_color = DEFAULT_OMIKUJI_COLOR

            if omikuji_color is None:
                print(
                    "Color recognition timed out; using default omikuji "
                    f"color: {DEFAULT_OMIKUJI_COLOR}",
                    file=sys.stderr,
                )
                omikuji_color = DEFAULT_OMIKUJI_COLOR

            print(f"Omikuji color: {omikuji_color} " f"({COLOR_TEXT[omikuji_color]})")

            # Color is known and Arm SDK is released; transition immediately
            # to the reading pose and send text + color to Claude.
            active_pose = PoseSession(
                pose_controller,
                JUDGMENT_TARGET,
                "judgment",
            )
            active_pose.start()
            try:
                raw_reply = chat(user_text, omikuji_color)
            except Exception as error:
                print(f"Miko LLM failed; continuing: {error}", file=sys.stderr)
                active_pose.release_and_wait()
                active_pose = None
                args.interrupted.clear()
                active_pose, listener_result = start_listening_cycle(
                    pose_controller, client, args, audio_backend
                )
                continue

            reply = raw_reply.strip()
            if not reply:
                print(
                    "Miko LLM returned an empty response; continuing.", file=sys.stderr
                )
                active_pose.release_and_wait()
                active_pose = None
                args.interrupted.clear()
                active_pose, listener_result = start_listening_cycle(
                    pose_controller, client, args, audio_backend
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
                        active_pose, next_result = start_listening_cycle(
                            pose_controller, client, args, audio_backend
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
                execute_face_wave(
                    arm_action_client,
                    args.face_wave_seconds,
                    args.action_settle_seconds,
                )
                active_pose, next_result = start_listening_cycle(
                    pose_controller, client, args, audio_backend
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
