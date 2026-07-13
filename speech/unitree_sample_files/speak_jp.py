import asyncio
import edge_tts
import subprocess

VOICE = "ja-JP-NanamiNeural"

TEXT = """
皆さん、こんにちは。
私はUnitree G1です。
研究室へようこそ。
今日はよろしくお願いいたします。
"""

OUTPUT = "/tmp/g1.mp3"


async def tts():

    communicate = edge_tts.Communicate(
        text=TEXT,
        voice=VOICE
    )

    await communicate.save(OUTPUT)


asyncio.run(tts())

subprocess.run([
    "ffplay",
    "-nodisp",
    "-autoexit",
    OUTPUT
])