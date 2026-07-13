# G1 AI Omikuji Miko

<p align="center">
  <a href="docs/index.html">
    <img src="https://img.shields.io/badge/Open%20Project%20Webpage-1A1917?style=for-the-badge" alt="Open Project Webpage" />
  </a>
</p>

## English

G1 AI Omikuji Miko is a hackathon demo that combines the Unitree G1 robot, LLM-generated responses, speech interaction, and omikuji-inspired fortune colors.

The goal is not to predict the future. It is to create a warm, shrine-like robot experience where a person speaks about a concern, draws a fortune, and receives a gentle response with matching robot motion.

Current fallback demo:

- manual input for the user’s concern
- manual selection of the fortune color: `gold`, `red`, or `blue`
- generated response text and motion label output

Run it with:

```bash

python3 -m speech.g1_voice_chat ETHERNET_DRIVE --asr-device cuda
```

## 日本語

<p align="center">
  <a href="docs/index.html">
    <img src="https://img.shields.io/badge/Open%20Webpage-8A6A2F?style=for-the-badge" alt="Open project webpage" />
  </a>
</p>

G1 AI おみくじ巫女は、Unitree G1、LLM の応答生成、音声対話、おみくじの色分けを組み合わせたハッカソン用デモです。

未来を占うことが目的ではありません。人が悩みを話し、おみくじを引き、巫女のようなやさしい言葉と動きで返してもらう、神社のような雰囲気の体験をつくることが目的です。

現在のフォールバック版:

- 悩みを手入力
- 签色を手動で選択: `gold`, `red`, `blue`
- 応答テキストと動作ラベルを生成

実行方法:

```bash

python3 -m speech.g1_voice_chat ETHERNET_DRIVE --asr-device cuda
```

