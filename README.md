# G1 AI おみくじ巫女机器人项目

这是一个基于 **Unitree G1 + LLM + 语音交互 + 抽签识别 + 机器人动作** 的黑客松 Demo 项目。

项目目标不是做真正的“占卜”，而是做一个有日本神社文化氛围的人形机器人交互体验：

用户说出自己的烦恼，抽取一张签，机器人识别签的颜色，然后用巫女风格的语言进行回应，并配合 G1 的身体动作，让用户产生一种“被倾听、被回应、被祝福”的感觉。

---

## 1. 项目核心概念

本项目不是预测未来，也不是心理咨询。

我们想做的是：

```text
人说出烦恼
↓
机器人倾听
↓
用户抽签
↓
机器人根据签色和用户内容生成回应
↓
通过语音和动作完成一个仪式感互动
```

关键词：

- 人形机器人
- LLM 生成回应
- 日本おみくじ文化
- 巫女风格交互
- 语音输入输出
- G1 上半身动作
- 安全稳定的现场 Demo

---

## 2. Demo 流程

最终目标流程：

```text
G1 一礼
↓
用户说出烦恼
↓
ASR 语音识别
↓
用户抽签
↓
摄像头识别签色
↓
LLM 生成巫女风格回应
↓
TTS 朗读
↓
G1 做对应动作
↓
机器人回到安全姿态
```

当前保底版流程：

```text
手动输入烦恼
↓
手动选择签色 gold / red / blue
↓
程序生成回应
↓
输出动作标签
↓
模拟 G1 动作
```

---

## 3. 当前版本状态

当前版本是 **manual fallback demo**，也就是手动保底版本。

现在已经可以运行：

```bash
python3 demo/main_manual_mode.py
```

当前版本暂时没有接入：

- 真机 G1 控制
- 摄像头颜色识别
- ASR 语音识别
- TTS 真实朗读
- OpenAI / Gemini API

现在主要目的是先保证：

```text
整个交互流程可以跑通
```

后面再逐步替换成真实模块。

---

## 4. 快速运行

进入项目目录：

```bash
cd ~/Desktop/g1-ai-omikuji-miko
```

运行手动 Demo：

```bash
python3 demo/main_manual_mode.py
```

示例输入：

```text
请输入用户烦恼：我担心黑客松 demo 失败
请选择签色 gold / red / blue：gold
```

示例输出：

```text
Action: pray

你抽到了金色签。它代表祝福、希望和被支持。
你刚才说的是：我担心黑客松 demo 失败
这不是对未来的断言，而是一个帮助你整理心情的小仪式。
现在最重要的不是立刻得到答案，而是先看清楚自己真正担心的是什么。
```

---

## 5. 签色设计

目前只使用三种签色，降低工程复杂度。

| 签色 | 含义 | 对应动作 |
|---|---|---|
| gold | 祝福、鼓励、积极确认 | pray |
| red | 行动、改变、向前迈出一步 | bow |
| blue | 冷静、整理、慢慢来 | nod |

---

## 6. 动作标签设计

机器人动作先统一成几个简单标签：

```text
bow      一礼
pray     合掌
nod      点头
listen   倾听
reset    回正
```

LLM 或规则模块最后只需要输出动作标签，不要直接控制机器人关节。

这样比较安全，也方便调试。

---

## 7. 仓库结构

计划中的项目结构：

```text
g1-ai-omikuji-miko/
├── demo/
│   └── main_manual_mode.py
│
├── robot/
│   ├── g1_controller.py
│   ├── actions.py
│   ├── safety.py
│   └── poses/
│
├── vision/
│   └── color_detector.py
│
├── speech/
│   ├── asr.py
│   └── tts.py
│
├── llm/
│   └── omikuji_generator.py
│
├── config/
│   ├── demo_config.yaml
│   ├── robot_config.yaml
│   └── prompt_config.yaml
│
├── docs/
│   ├── demo_flow.md
│   ├── robot_safety.md
│   └── troubleshooting.md
│
└── README.md
```

当前版本可能还没有完全拆分成这个结构，后面逐步整理。

---

## 8. 模块分工建议

### robot/

负责人：G1 动作控制

内容：

- G1 连接
- 上半身动作
- 一礼
- 合掌
- 点头
- 倾听姿态
- 回正姿态
- 真机安全限制

注意：

G1 真机动作一定要保守。优先稳定，不要追求大幅度。

---

### vision/

负责人：视觉识别

内容：

- 摄像头读取
- 签色识别
- 输出 gold / red / blue / unknown

接口目标：

```python
def detect_omikuji_color() -> str:
    return "gold"
```

---

### speech/

负责人：语音输入输出

内容：

- ASR：用户语音转文字
- TTS：机器人朗读回应

接口目标：

```python
def listen_user() -> str:
    return "用户说的话"

def speak_text(text: str) -> None:
    pass
```

---

### llm/

负责人：LLM 生成回应

内容：

- 根据用户烦恼和签色生成回应
- 输出机器人动作标签

接口目标：

```python
def generate_omikuji_response(user_text: str, color: str) -> dict:
    return {
        "text": "巫女风格回应",
        "action": "pray"
    }
```

---

### demo/

负责人：集成

内容：

- 把 robot / vision / speech / llm 串起来
- 提供现场运行入口
- 提供手动 fallback 版本

最重要文件：

```text
demo/main_manual_mode.py
```

这是现场保底版本。

---

## 9. 现场安全原则

真机 G1 测试必须注意：

1. 不要一上来控制腰部大幅前倾。
2. 动作幅度先小后大。
3. 每个动作都要能回到 reset pose。
4. 真机运行前先在仿真里测试。
5. 保持急停准备。
6. 黑客松现场优先使用稳定动作，不追求复杂动作。
7. 如果机器人下肢开始明显找平衡或乱动，立刻停止动作。

当前建议：

```text
优先控制双臂
少量使用腰部
避免大幅前倾
```

---

## 10. Git 分支规则

推荐分支：

```text
main
dev
feature/robot-motion
feature/vision-color
feature/asr-tts
feature/llm-response
feature/demo-integration
```

规则：

- `main`：稳定版本
- `dev`：集成测试版本
- `feature/*`：个人开发分支
- 不要直接把不稳定代码 push 到 main
- 不要上传 API Key
- 不要上传 `.env`

---

## 11. 不要上传的内容

不要上传：

```text
.env
API key
大视频文件
临时录音
机器人运行日志
虚拟环境 venv/
__pycache__/
```

API Key 应该放在本地 `.env` 里。

---

## 12. 当前优先级

短期目标：

```text
先保证 demo 能稳定跑
```

优先级从高到低：

1. 手动版 demo 跑通
2. LLM 生成回应接入
3. TTS 接入
4. G1 安全动作接入
5. 颜色识别接入
6. ASR 接入

现场最重要的是：

```text
即使 ASR / 摄像头 / 网络失败，也能用手动模式继续演示
```

---

## 13. 项目一句话说明

这是一个用 Unitree G1 作为身体载体，结合 LLM 语言生成和おみくじ文化，完成“倾听—抽签—回应—动作”的人形机器人交互 Demo。
EOF
