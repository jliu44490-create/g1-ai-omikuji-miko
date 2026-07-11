import wave
from pathlib import Path

from piper import PiperVoice

MODEL_DIR = Path("models/tsukuyomi")
MODEL_PATH = MODEL_DIR / "tsukuyomi-chan-6lang-fp16.onnx"
CONFIG_PATH = MODEL_DIR / "config.json"

# Load once at program startup.
voice = PiperVoice.load(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    use_cuda=False,  # CPU is fast enough for this small model.
)


def generate_japanese_audio(
    text: str,
    output_path: str = "output.wav",
) -> None:
    with wave.open(output_path, "wb") as wav_file:
        voice.synthesize(
            text=text,
            wav_file=wav_file,
            length_scale=1.5,  # Higher = slower speech
            noise_scale=0.667,  # Voice variation
            sentence_silence=0.15,
        )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    generate_japanese_audio(
        "こんにちは、これはテストです。",
        "audio/output.wav",
    )
