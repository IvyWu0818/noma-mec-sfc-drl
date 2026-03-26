import numpy as np
from envs.iiot_env import IIoTEnv


def main():
    env = IIoTEnv(num_tasks=5, beta=10.0)

    print("=== Environment Info ===")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print()

    obs, info = env.reset()

    print("=== Reset ===")
    print("Initial obs:", obs)
    print()

    step_count = 0
    done = False

    while not done:
        step_count += 1

        # 隨機 action（測試用）
        action = env.action_space.sample()

        print(f"=== Step {step_count} ===")
        print("Action:", action)

        obs, reward, terminated, truncated, info = env.step(action)

        print("Obs:", obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Info:", info)
        print("-" * 40)

        done = terminated or truncated

    print("Episode finished.")


if __name__ == "__main__":
    main()