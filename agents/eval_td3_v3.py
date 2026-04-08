import numpy as np
from stable_baselines3 import TD3

from envs.iiot_env_v3 import IIoTEnvV3


def evaluate(model, num_episodes=20, beta=10.0):
    env = IIoTEnvV3(num_tasks=10, beta=beta)

    total_rewards = []
    total_delays = []
    total_slacks = []
    total_timeouts = 0
    total_tasks = 0

    node_counts = {"mec0": 0, "mec1": 0, "mec2": 0}

    patterns = {}
    actions_list = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            actions_list.append(action)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            total_delays.append(info["delay"])
            total_slacks.append(info["slack"])

            if info["delay"] > info["deadline"]:
                total_timeouts += 1

            for node in info["selected_nodes"]:
                node_counts[node] += 1

            # ⭐ pattern 分析
            pattern = tuple(info["selected_nodes"])
            patterns[pattern] = patterns.get(pattern, 0) + 1

            total_tasks += 1
            done = terminated or truncated

        total_rewards.append(ep_reward)

    print("\n=== TD3 V3 Evaluation Results ===")
    print(f"Avg reward: {np.mean(total_rewards):.2f}")
    print(f"Avg delay: {np.mean(total_delays):.2f}")
    print(f"Avg slack: {np.mean(total_slacks):.2f}")
    print(f"Timeout ratio: {total_timeouts / total_tasks:.2%}")
    print(f"Node selection counts: {node_counts}")

    # ⭐ pattern
    print("\nTop selection patterns:")
    for k, v in sorted(patterns.items(), key=lambda x: -x[1])[:10]:
        print(k, v)

    # ⭐ action 分布
    actions_array = np.array(actions_list)
    print("\nAction stats:")
    print("mean:", np.mean(actions_array, axis=0))
    print("std:", np.std(actions_array, axis=0))

    # ⭐ delay 分布
    print("\nDelay percentiles:")
    print("p50:", np.percentile(total_delays, 50))
    print("p90:", np.percentile(total_delays, 90))
    print("p99:", np.percentile(total_delays, 99))

    print("=================================\n")
    

def evaluate_random(num_episodes=20, beta=10.0):
    env = IIoTEnvV3(num_tasks=10, beta=beta)

    total_delays = []
    total_slacks = []
    total_timeouts = 0
    total_tasks = 0

    node_counts = {"mec0": 0, "mec1": 0, "mec2": 0}

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

            for node in info["selected_nodes"]:
                node_counts[node] += 1

            total_tasks += 1
            done = terminated or truncated

    print("\n=== RANDOM V3 Baseline ===")
    print(f"Avg delay: {np.mean(total_delays):.2f}")
    print(f"Avg slack: {np.mean(total_slacks):.2f}")
    print(f"Timeout ratio: {total_timeouts / total_tasks:.2%}")
    print(f"Node selection counts: {node_counts}")
    print("==========================\n")


def main():
    model = TD3.load("models/td3_iiot_env_v3_final")

    evaluate(model, num_episodes=30, beta=10.0)
    evaluate_random(num_episodes=30, beta=10.0)


if __name__ == "__main__":
    main()