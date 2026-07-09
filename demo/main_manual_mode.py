class G1Controller:
    def __init__(self, use_real_robot=False):
        self.use_real_robot = use_real_robot

    def connect(self):
        print("[G1] Mock mode: no real robot connected.")

    def disconnect(self):
        print("[G1] Disconnect.")

    def reset_pose(self):
        print("[G1] Action: reset_pose")

    def play_action(self, action_name):
        print(f"[G1] Play action: {action_name}")


def generate_omikuji_response(user_text, color):
    color_map = {
        "gold": {
            "action": "pray",
            "message": "你抽到了金色签。它代表祝福、希望和被支持。"
        },
        "red": {
            "action": "bow",
            "message": "你抽到了赤色签。它代表行动、勇气和向前迈出一步。"
        },
        "blue": {
            "action": "nod",
            "message": "你抽到了青色签。它代表冷静、整理和慢慢看清自己的心。"
        },
    }

    info = color_map.get(color, {
        "action": "listen",
        "message": "签色还没有被清楚识别。不过没关系，请先慢慢整理自己的心情。"
    })

    text = (
        info["message"] + "\n"
        + "你刚才说的是：" + user_text + "\n"
        + "这不是对未来的断言，而是一个帮助你整理心情的小仪式。\n"
        + "现在最重要的不是立刻得到答案，而是先看清楚自己真正担心的是什么。"
    )

    return {
        "text": text,
        "action": info["action"]
    }


def main():
    robot = G1Controller(use_real_robot=False)

    try:
        robot.connect()
        robot.reset_pose()

        print("=== G1 AI Omikuji Miko Manual Demo ===")

        robot.play_action("bow")

        user_text = input("请输入用户烦恼：")
        color = input("请选择签色 gold / red / blue：").strip().lower()

        result = generate_omikuji_response(user_text, color)

        print("\n生成结果：")
        print("Action:", result["action"])
        print("Text:", result["text"])

        robot.play_action(result["action"])

        print("\n[Miko TTS]")
        print(result["text"])

        robot.reset_pose()

    except KeyboardInterrupt:
        print("\n手动中断，回到安全姿态。")
        robot.reset_pose()

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
