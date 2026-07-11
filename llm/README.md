# LLM Module — AIおみくじ巫女 解読文生成

> [日本語](#日本語) | [中文](#中文) | [English](#english)

---

## 日本語

### このモジュールの役割

デモ全体の流れ：

```
参加者が話す → ASR（音声認識）→ 【LLMモジュール（ここ）】→ TTS（音声合成）→ G1ロボットが語る
                                         ↑
                             カメラ → 色の識別 → 色情報
```

**このモジュールは「テキスト＋色」を受け取り、巫女の解読文を返します。**

入力：`(ユーザーの言葉: str, おみくじの色: "gold" | "red" | "blue")`
出力：`巫女の解読文: str`

### 使い方

```python
from llm import generate_reading

result = generate_reading("最近仕事が辛くて、毎日が苦しいです", "blue")
print(result)
```

CLIで直接テスト：
```bash
python3 llm/miko.py
```

### セットアップ

```bash
pip install -r requirements.txt
cp .env.example .env
# .env を編集して API キーを入力
```

`.env` の設定：
```
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-sonnet-5
```

### 三色の意味

| 色 | 姿勢 | キーワード |
|---|---|---|
| 金 | 祝福 | 肯定・祝福・積極 |
| 赤 | 行動 | 行動・楽観・前向き |
| 青 | 受容 | 冷静・平和・包容・受容 |

### 現在の到達点

- ✅ Claude API による高品質な日本語解読文の生成
- ✅ オフラインフォールバック（API障害時に自動切替）
- ✅ デモ用プリセット（推奨入力5つ × 3色 = 15パターンの定製フォールバック）
- ✅ System prompt v3（温かく芯の強い巫女の人格設定）

### 今後の予定

- [ ] 語りの文体・語感のさらなる調整
- [ ] 書籍の理念や哲学の人格への組み込み
- [ ] TTS・G1動作との連携テスト
- [ ] 多言語対応（中国語・英語）

### ファイル構成

```
llm/
├── __init__.py            # generate_reading のエクスポート
├── miko.py                # コアモジュール
├── miko_system_prompt.md  # 巫女の人格プロンプト (v3)
├── fallback_readings.json # 汎用フォールバック（3色×3段）
├── demo_presets.json      # デモ用プリセットフォールバック（5入力×3色）
└── README.md              # この文書
```

---

## 中文

### 这个模块做什么

Demo 整体流程：

```
参加者说话 → ASR（语音识别）→ 【LLM模块（这里）】→ TTS（语音合成）→ G1机器人宣读
                                        ↑
                            摄像头 → 颜色识别 → 颜色信息
```

**本模块接收"文字+颜色"，返回巫女的解读文。**

输入：`(用户说的话: str, 签的颜色: "gold" | "red" | "blue")`
输出：`巫女解读文: str`

### 怎么用

```python
from llm import generate_reading

result = generate_reading("最近仕事が辛くて、毎日が苦しいです", "blue")
print(result)
```

命令行直接测试：
```bash
python3 llm/miko.py
```

### 环境搭建

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 API key
```

`.env` 配置：
```
ANTHROPIC_API_KEY=你的key
CLAUDE_MODEL=claude-sonnet-5
```

### 三色含义

| 颜色 | 姿态 | 关键词 |
|---|---|---|
| 金 | 祝福 | 肯定、祝福、积极 |
| 赤 | 行动 | 行动、乐观、向前 |
| 青 | 受容 | 冷静、平和、包容、接纳 |

### 目前达到的效果

- ✅ 使用 Claude API 生成高质量日语解读文
- ✅ 断网兜底（API 调用失败时自动切换到预生成文案）
- ✅ Demo 预设兜底（5 个推荐输入 × 3 色 = 15 段量身定制的文案）
- ✅ System prompt v3（温柔坚强、善于倾听的巫女人格）

### 接下来要做的

- [ ] 语言风格的进一步微调
- [ ] 把书籍理念和哲学融入巫女人格
- [ ] 与 TTS、G1 动作联调
- [ ] 多语言支持（中文、英语）

### 文件结构

```
llm/
├── __init__.py            # 导出 generate_reading
├── miko.py                # 核心模块
├── miko_system_prompt.md  # 巫女人格 prompt (v3)
├── fallback_readings.json # 通用兜底文案（3色×3段）
├── demo_presets.json      # Demo 预设兜底（5输入×3色）
└── README.md              # 本文档
```

---

## English

### What this module does

Full demo pipeline:

```
Participant speaks → ASR → 【LLM Module (here)】→ TTS → G1 Robot speaks
                                    ↑
                        Camera → Color detection → Color info
```

**This module takes "text + color" as input and returns a shrine maiden's reading.**

Input: `(user_text: str, omikuji_color: "gold" | "red" | "blue")`
Output: `reading_text: str`

### Usage

```python
from llm import generate_reading

result = generate_reading("最近仕事が辛くて、毎日が苦しいです", "blue")
print(result)
```

CLI test:
```bash
python3 llm/miko.py
```

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API key
```

`.env` config:
```
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-sonnet-5
```

### Three colors

| Color | Posture | Keywords |
|---|---|---|
| Gold | Blessing | Affirmation, celebration, positivity |
| Red | Action | Action, optimism, moving forward |
| Blue | Acceptance | Calm, peace, embracing, acceptance |

### Current status

- ✅ High-quality Japanese readings via the Claude API
- ✅ Offline fallback (auto-switches to pre-generated readings on API failure)
- ✅ Demo presets (5 recommended inputs × 3 colors = 15 tailored fallback readings)
- ✅ System prompt v3 (warm, strong, empathetic shrine maiden persona)

### Next steps

- [ ] Further refinement of language style and tone
- [ ] Integrate philosophical concepts from books into the persona
- [ ] Integration testing with TTS and G1 motion control
- [ ] Multi-language support (Chinese, English)

### File structure

```
llm/
├── __init__.py            # Exports generate_reading
├── miko.py                # Core module
├── miko_system_prompt.md  # Shrine maiden persona prompt (v3)
├── fallback_readings.json # Generic fallback readings (3 colors × 3 each)
├── demo_presets.json      # Demo preset fallbacks (5 inputs × 3 colors)
└── README.md              # This document
```
