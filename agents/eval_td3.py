import numpy as np
from stable_baselines3 import TD3

from envs.iiot_env import IIoTEnv


def evaluate(model, num_episodes=20, beta=10.0):
    env = IIoTEnv(num_tasks=10, beta=beta)

    total_rewards = []
    total_delays = []
    total_slacks = []
    total_timeouts = 0
    total_tasks = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False

        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward

            # 收集 metrics
            total_delays.append(info["delay"])
            total_slacks.append(info["slack"])

            if info["delay"] > info["deadline"]:
                total_timeouts += 1

            total_tasks += 1

            done = terminated or truncated

        total_rewards.append(ep_reward)

    print("\n=== TD3 Evaluation Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Avg reward: {np.mean(total_rewards):.2f}")
    print(f"Avg delay: {np.mean(total_delays):.2f}")
    print(f"Avg slack: {np.mean(total_slacks):.2f}")
    print(f"Timeout ratio: {total_timeouts / total_tasks:.2%}")
    print(f"Max delay: {np.max(total_delays):.2f}")
    print(f"Max slack: {np.max(total_slacks):.2f}")
    print("==============================\n")


def evaluate_random(num_episodes=20, beta=10.0):
    env = IIoTEnv(num_tasks=10, beta=beta)

    total_delays = []
    total_slacks = []
    total_timeouts = 0
    total_tasks = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            total_delays.append(info["delay"])
            total_slacks.append(info["slack"])

            if info["delay"] > info["deadline"]:
                total_timeouts += 1

            total_tasks += 1

            done = terminated or truncated

    print("\n=== RANDOM Baseline ===")
    print(f"Avg delay: {np.mean(total_delays):.2f}")
    print(f"Avg slack: {np.mean(total_slacks):.2f}")
    print(f"Timeout ratio: {total_timeouts / total_tasks:.2%}")
    print("=======================\n")


def main():
    # 載入模型
    model = TD3.load("models/td3_iiot_env_final")

    # TD3 評估
    evaluate(model, num_episodes=30, beta=10.0)

    # Random baseline
    evaluate_random(num_episodes=30, beta=10.0)


if __name__ == "__main__":
    main()