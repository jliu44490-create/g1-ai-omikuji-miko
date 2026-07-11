
## Prerequisite

Create venv:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
(or by uv:)
```bash
uv venv
uv pip install -r requirements.txt
```

Download tsukuyomi-chan:

```bash
python -m piper \
  --download-model tsukuyomi \
  --download-dir speech/models/tsukuyomi
```

Place the converted Kotoba Whisper model at:

```text
speech/models/kotoba-whisper-v2.0-faster/
```

It must contain at least `model.bin`, `config.json`, `tokenizer.json`,
`preprocessor_config.json`, and `vocabulary.json`. This local directory is the
default, so no Hugging Face request is made during normal startup.

## Real-time voice loop

Run from the repository root:

```bash
python -m speech.realtime_voice
```

The program listens until it detects the end of an utterance, transcribes it,
and speaks the recognized text. Microphone capture is stopped during playback,
so the synthesized voice is not immediately recognized again. Press `Ctrl+C`
to quit.

Useful options:

```bash
# Find microphone and speaker indices.
python -m speech.realtime_voice --list-devices

# Select devices and process only one utterance.
python -m speech.realtime_voice --input-device 1 --output-device 3 --once

# Test recognition without loading Piper or playing audio.
python -m speech.realtime_voice --no-speak

# Add a simple spoken acknowledgement.
python -m speech.realtime_voice \
  --response-template '「{text}」と聞こえました。'
```

The default speech detector threshold is `0.015`. If it triggers on room noise,
raise it with `--threshold`; if it misses speech, lower it. Use
`python -m speech.realtime_voice --help` for all timing, model, and device
options.

For application integration, import `JapaneseASR` from `speech.asr` and
`JapaneseTTS` from `speech.tts`. Replace the response-template line in
`realtime_voice.py` with an LLM response function when connecting the full demo.

## WSL2 microphone recording

WSLg exposes its microphone as the PulseAudio source `RDPSource`. Ubuntu's
standard PortAudio package may expose only ALSA/OSS, in which case
`sounddevice` reports default device `-1`. `record_test.py` detects this and
automatically uses `parec` instead:

```bash
python speech/record_test.py --output speech/audio/output.wav
```

To inspect both kinds of audio devices or force the WSLg backend:

```bash
python speech/record_test.py --list-devices
python speech/record_test.py --backend pulse --device RDPSource
```

If `parec` is unavailable, install the Ubuntu package `pulseaudio-utils`.
Windows must also grant microphone access to desktop applications under
**Settings > Privacy & security > Microphone**.

The integrated loop uses the same automatic selection. On WSLg it captures
through `parec` and plays synthesized speech through `paplay`:

```bash
python -m speech.realtime_voice --once
```

You can override detection with `--audio-backend pulse`. For PulseAudio,
`--input-device` and `--output-device` take source/sink names such as
`RDPSource` and `RDPSink`, rather than PortAudio numeric indices.

ASR uses CPU `int8` inference by default, which avoids requiring CUDA libraries
inside WSL. To opt into a configured NVIDIA CUDA 12 environment, pass
`--asr-device cuda --compute-type float16`.

When `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` are installed in the active
virtual environment, `realtime_voice.py` automatically adds their library
directories before starting CUDA. No manual `LD_LIBRARY_PATH` export is needed.

## Unitree G1 voice chat

The original G1 sample used Microsoft Edge TTS voice
`ja-JP-NanamiNeural`. The integrated version uses the local Kotoba Whisper ASR
and Piper Tsukuyomi TTS while retaining the existing Ollama `qwen:0.5b` LLM:

```bash
python -m speech.unitree_sample_files.g1_voice_chat ROBOT_INTERFACE \
  --asr-device cuda --compute-type float16
```

Replace `ROBOT_INTERFACE` with the wired interface used to reach G1. The default
audio input is the system microphone. Add `--input-device DEVICE` or
`--audio-backend pulse` when necessary.

The reliable default is half-duplex: microphone capture starts after the G1 has
finished its complete reply. This prevents the G1 speaker from being mistaken
for new user speech. A `0.8` second playback tail also prevents `PlayStop` from
cutting buffered audio; tune it with `--playback-tail-seconds` if necessary.

Optional barge-in can be enabled with `--barge-in`. Speech above
`--barge-in-threshold` then calls `PlayStop("tts")`, and the interrupting utterance
becomes the next turn. Energy detection alone cannot distinguish the user from
speaker echo, so barge-in should only be enabled with a directional microphone
or acoustic echo cancellation. Raise the threshold if speaker echo triggers it.
