"""
AIおみくじ巫女 — LLM モジュール
generate_reading(text, color) -> str
"""

import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LLM_DIR = Path(__file__).parent
FALLBACK_PATH = LLM_DIR / "fallback_readings.json"
PRESETS_PATH = LLM_DIR / "demo_presets.json"

# --- System Prompt ---

def _load_system_prompt() -> str:
    md_path = LLM_DIR / "miko_system_prompt.md"
    text = md_path.read_text(encoding="utf-8")
    marker = "## 【PROMPT】"
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError("miko_system_prompt.md 中找不到【PROMPT】标记")
    after = text[idx:]
    start = after.find("```")
    if start == -1:
        raise RuntimeError("找不到 prompt 代码块")
    start = after.find("\n", start) + 1
    end = after.find("```", start)
    return after[start:end].strip()


SYSTEM_PROMPT = _load_system_prompt()

# --- Fallback ---

def _load_fallbacks() -> dict[str, list[str]]:
    with open(FALLBACK_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_presets() -> list[dict]:
    if not PRESETS_PATH.exists():
        return []
    with open(PRESETS_PATH, encoding="utf-8") as f:
        return json.load(f).get("presets", [])


def _match_preset(text: str, color: str):
    best_hits = 0
    best_preset = None
    for preset in _load_presets():
        if color not in preset:
            continue
        keywords = preset.get("keywords", [])
        hits = sum(1 for kw in keywords if kw in text)
        if hits > best_hits:
            best_hits = hits
            best_preset = preset
    if best_hits >= 3 and best_preset is not None:
        return best_preset[color]
    return None


def _fallback_reading(text: str, color: str) -> str:
    matched = _match_preset(text, color)
    if matched:
        return matched
    data = _load_fallbacks()
    key = color.lower()
    if key not in data:
        key = list(data.keys())[0]
    return random.choice(data[key])

# --- Claude LLM ---

COLOR_LABEL = {
    "gold": "金（祝福）",
    "red": "赤（行動）",
    "blue": "青（冷静）",
}


def _build_user_message(text: str, color: str) -> str:
    label = COLOR_LABEL.get(color.lower(), color)
    return f"【相手の言葉】{text}\n【色】{label}"


def _call_claude(user_msg: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in resp.content:
        if block.type == "text":
            return block.text.strip()
    return resp.content[-1].text.strip()


def _stream_claude(user_msg: str):
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    with client.messages.stream(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# --- Public API ---

def generate_reading(text: str, color: str) -> str:
    """
    接口 A → B：接收用户文字 + 颜色，返回巫女解读文。

    Parameters:
        text:  用户说的话（文字）
        color: おみくじ颜色 "gold" | "red" | "blue"

    Returns:
        巫女の解読文（str）
    """
    color = color.lower()
    if color not in COLOR_LABEL:
        raise ValueError(f"color must be one of {list(COLOR_LABEL.keys())}, got {color!r}")

    preset = _match_preset(text, color)
    if preset:
        return preset

    user_msg = _build_user_message(text, color)
    try:
        return _call_claude(user_msg)
    except Exception as e:
        print(f"[miko] LLM 呼び出し失敗 ({type(e).__name__}: {e})、フォールバックへ切替")
        return _fallback_reading(text, color)


def generate_reading_stream(text: str, color: str):
    """
    Streaming 版：逐块 yield 解读文，TTS 可以边收边读。
    失败时 yield 整段 fallback。
    """
    color = color.lower()
    if color not in COLOR_LABEL:
        raise ValueError(f"color must be one of {list(COLOR_LABEL.keys())}, got {color!r}")

    user_msg = _build_user_message(text, color)
    try:
        yield from _stream_claude(user_msg)
    except Exception as e:
        print(f"[miko] LLM stream 失敗 ({type(e).__name__}: {e})、フォールバックへ切替")
        yield _fallback_reading(text, color)


# --- CLI ---

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        txt = sys.argv[1]
        clr = sys.argv[2]
    else:
        txt = input("あなたの言葉を聞かせてください: ")
        clr = input("色を選んでください (gold / red / blue): ").strip()

    print("\n--- 巫女の言葉 ---\n")
    t0 = time.time()
    result = generate_reading(txt, clr)
    elapsed = time.time() - t0
    print(result)
    print(f"\n[{elapsed:.1f}s | provider=claude]")
