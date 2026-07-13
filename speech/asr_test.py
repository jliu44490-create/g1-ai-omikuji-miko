from faster_whisper import WhisperModel
import time
from pathlib import Path

SPEECH_DIR = Path(__file__).resolve().parent
MODEL_PATH = SPEECH_DIR / "models" / "kotoba-whisper-v2.0-faster"
model = WhisperModel(str(MODEL_PATH), device="cpu", compute_type="int8")

start_time = time.time()

segments, info = model.transcribe(
    str(SPEECH_DIR / "audio" / "output.wav"),
    language="ja",
    chunk_length=15,
    condition_on_previous_text=False,
)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

recognized_text = " ".join([segment.text for segment in segments])
print("Recognized text: %s" % recognized_text)
end_time = time.time()
print("Execution time: %.2f seconds" % (end_time - start_time))
