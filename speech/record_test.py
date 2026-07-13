#!/usr/bin/env python3

import argparse
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import sounddevice as sd
import soundfile as sf

# Audio blocks produced by the microphone callback are placed here.
audio_queue: queue.Queue = queue.Queue()

# Tells the writer thread that recording has stopped.
stop_event = threading.Event()


def microphone_callback(indata, frames, time_info, status) -> None:
    """
    Called automatically by sounddevice whenever a new audio block arrives.

    Keep this function lightweight. Writing to disk is handled by another
    thread so that microphone capture is not blocked.
    """
    if status:
        print(f"Audio warning: {status}", file=sys.stderr)

    audio_queue.put(indata.copy())


def write_audio(
    output_path: Path,
    sample_rate: int,
    channels: int,
) -> None:
    """Continuously take audio blocks from the queue and write them to WAV."""
    try:
        with sf.SoundFile(
            file=str(output_path),
            mode="w",
            samplerate=sample_rate,
            channels=channels,
            subtype="PCM_16",
            format="WAV",
        ) as wav_file:

            while not stop_event.is_set() or not audio_queue.empty():
                try:
                    audio_block = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                wav_file.write(audio_block)
                audio_queue.task_done()

    except Exception as error:
        print(f"Failed to write the audio file: {error}", file=sys.stderr)
        stop_event.set()


def parse_device(device_argument: Optional[str]) -> Optional[Union[int, str]]:
    """
    Convert a numeric device argument into an integer device index.

    Text is preserved so sounddevice can also search by device name.
    """
    if device_argument is None:
        return None

    try:
        return int(device_argument)
    except ValueError:
        return device_argument


def clear_audio_queue() -> None:
    """Remove any leftover audio blocks before starting a new recording."""
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
            audio_queue.task_done()
        except queue.Empty:
            break


def sounddevice_has_input(device: Optional[Union[int, str]]) -> bool:
    """Return whether PortAudio can see the requested input device."""
    try:
        sd.check_input_settings(device=device, channels=1, samplerate=16000)
        return True
    except Exception:
        return False


def record_with_pulse(
    output_path: Path,
    sample_rate: int,
    channels: int,
    device: Optional[Union[int, str]],
) -> None:
    """Record through WSLg/PulseAudio when PortAudio has no Pulse backend."""
    parec = shutil.which("parec")
    if parec is None:
        raise RuntimeError("parec is not installed (install pulseaudio-utils)")

    pulse_device = str(device) if device is not None else "@DEFAULT_SOURCE@"
    command = [
        parec,
        "--record",
        f"--device={pulse_device}",
        f"--rate={sample_rate}",
        "--format=s16le",
        f"--channels={channels}",
        "--file-format=wav",
        str(output_path),
    ]

    print(f"Recording from PulseAudio source {pulse_device}...")
    print("Press Enter to stop.")
    process = subprocess.Popen(command)
    try:
        input()
    finally:
        # SIGINT lets parec finalize the WAV header before exiting.
        process.send_signal(signal.SIGINT)
        try:
            return_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            return_code = process.wait(timeout=2)
        if return_code not in (0, -signal.SIGINT):
            raise RuntimeError(f"parec exited with status {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record microphone audio and save it as a WAV file."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output WAV filename. A timestamped name is used by default.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Recording sample rate. Default: 16000 Hz.",
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of channels. Default: 1 (mono).",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Input device index or part of its name.",
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Show available audio devices and exit.",
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "sounddevice", "pulse"],
        default="auto",
        help="Audio backend. Auto uses PulseAudio when PortAudio has no input device.",
    )

    args = parser.parse_args()

    if args.list_devices:
        print("PortAudio devices:")
        print(sd.query_devices())
        if shutil.which("pactl") and os.environ.get("PULSE_SERVER"):
            print("\nPulseAudio sources:")
            subprocess.run(["pactl", "list", "short", "sources"], check=False)
        return

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"audio/recording_{timestamp}.wav")
    else:
        output_path = args.output

    if output_path.suffix.lower() != ".wav":
        output_path = output_path.with_suffix(".wav")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = parse_device(args.device)

    use_pulse = args.backend == "pulse" or (
        args.backend == "auto"
        and not sounddevice_has_input(device)
        and bool(os.environ.get("PULSE_SERVER"))
        and shutil.which("parec") is not None
    )

    if use_pulse:
        input("Press Enter to start recording...")
        try:
            record_with_pulse(
                output_path, args.sample_rate, args.channels, device
            )
        except (OSError, RuntimeError) as error:
            print(f"PulseAudio recording failed: {error}", file=sys.stderr)
            sys.exit(1)

        if output_path.exists() and output_path.stat().st_size > 44:
            print(f"\nRecording saved successfully:\n{output_path.resolve()}")
            return
        print("\nNo usable recording was saved.", file=sys.stderr)
        sys.exit(1)

    try:
        # Check that the selected microphone supports these settings.
        sd.check_input_settings(
            device=device,
            channels=args.channels,
            samplerate=args.sample_rate,
            dtype="float32",
        )
    except Exception as error:
        print(f"Cannot use the selected microphone settings: {error}")
        print("\nRun this command to see available devices:")
        print("  python record_audio.py --list-devices")
        sys.exit(1)

    input("Press Enter to start recording...")

    clear_audio_queue()
    stop_event.clear()

    writer_thread = threading.Thread(
        target=write_audio,
        args=(output_path, args.sample_rate, args.channels),
        daemon=False,
    )
    writer_thread.start()

    try:
        with sd.InputStream(
            device=device,
            samplerate=args.sample_rate,
            channels=args.channels,
            dtype="float32",
            callback=microphone_callback,
            blocksize=0,
        ):
            print("Recording...")
            print("Press Enter to stop.")
            input()

    except KeyboardInterrupt:
        print("\nRecording interrupted.")

    except Exception as error:
        print(f"\nRecording failed: {error}", file=sys.stderr)

    finally:
        stop_event.set()
        writer_thread.join()

    if output_path.exists() and output_path.stat().st_size > 44:
        print(f"\nRecording saved successfully:")
        print(output_path.resolve())
    else:
        print("\nNo usable recording was saved.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
