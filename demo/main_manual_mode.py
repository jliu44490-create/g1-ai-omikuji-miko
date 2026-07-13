import time

from llm.miko import generate_reading


COLOR_ACTION = {
    "gold": "pray",
    "red": "bow",
    "blue": "nod",
}


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


def main():
    robot = G1Controller(use_real_robot=False)

    try:
        robot.connect()
        robot.reset_pose()

        print("=== G1 AI Omikuji Miko Manual Demo ===")
        robot.play_action("bow")

        user_text = input("あなたの言葉を聞かせてください: ")
        color = input("色を選んでください (gold / red / blue): ").strip().lower()

        if color not in COLOR_ACTION:
            print(f"Unknown color: {color}, defaulting to gold")
            color = "gold"

        print("\n巫女が言葉を紡いでいます...")
        t0 = time.time()
        reading = generate_reading(user_text, color)
        elapsed = time.time() - t0

        action = COLOR_ACTION[color]
        robot.play_action(action)

        print(f"\n--- 巫女の言葉 ({elapsed:.1f}s) ---\n")
        print(reading)

        robot.reset_pose()

    except KeyboardInterrupt:
        print("\n手動中断。")
        robot.reset_pose()

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
