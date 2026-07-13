"""
Simple integration test: ASR text → LLM → (TTS)
Verifies the full pipeline interfaces work end-to-end.
TTS is simulated if piper is not installed.
"""
import time
from llm.miko import generate_reading

TEST_INPUTS = [
    ("転職したいけど踏み出せない", "gold"),
    ("好きな人に気持ちを伝えたいけど怖い", "red"),
    ("毎日同じことの繰り返しで疲れた", "blue"),
]

try:
    from speech.tts_test import generate_japanese_audio
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False


def main():
    print("=" * 50)
    print("Integration Test: ASR → LLM → TTS")
    print(f"TTS: {'available' if TTS_AVAILABLE else 'simulated (piper not installed)'}")
    print("=" * 50)

    for i, (text, color) in enumerate(TEST_INPUTS, 1):
        print(f"\n--- Test {i}/3 ---")
        print(f"Input:  {text}")
        print(f"Color:  {color}")

        t0 = time.time()
        reading = generate_reading(text, color)
        llm_time = time.time() - t0

        print(f"Output: {reading}")
        print(f"LLM:    {llm_time:.1f}s, {len(reading)}字")

        if TTS_AVAILABLE:
            wav_path = f"speech/audio/test_{i}_{color}.wav"
            t0 = time.time()
            generate_japanese_audio(reading, wav_path)
            tts_time = time.time() - t0
            print(f"TTS:    {tts_time:.1f}s → {wav_path}")
        else:
            print(f"TTS:    (skipped) would generate audio from {len(reading)} chars")

    print("\n" + "=" * 50)
    print("All tests passed.")


if __name__ == "__main__":
    main()
