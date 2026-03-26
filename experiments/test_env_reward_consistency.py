import random
import numpy as np

from envs.iiot_env_v3 import IIoTEnvV3


EPS = 1e-9


def test_env_reward_consistency(
    num_steps: int = 10,
    seed: int = 42,
    beta: float = 10.0,
    use_fixed_action: bool = False,
):
    """
    驗證 IIoTEnvV3.step() 中 reward 是否滿足：

        reward = -(delay / 20 + beta * slack / 20)

    這代表公式 (1) 已正確嵌入 RL 環境（經過 scaling）。
    """

    print("=" * 90)
    print("Test E: Environment reward consistency check")
    print("=" * 90)

    np.random.seed(seed)
    random.seed(seed)

    env = IIoTEnvV3(num_tasks=num_steps, beta=beta, seed=seed)
    obs, info = env.reset(seed=seed)

    print(
        f"{'Step':<6}"
        f"{'Delay':<12}"
        f"{'Deadline':<12}"
        f"{'Slack':<12}"
        f"{'Reward(calc)':<16}"
        f"{'Reward(exp)':<16}"
        f"{'Pass':<8}"
    )

    all_passed = True

    for step_idx in range(1, num_steps + 1):

        # 👉 Action 選擇
        if use_fixed_action:
            action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
        else:
            action = env.action_space.sample()

        next_obs, reward, terminated, truncated, step_info = env.step(action)

        delay = step_info["delay"]
        deadline = step_info["deadline"]
        slack = step_info["slack"]

        # 👉 理論 reward（對應你的公式）
        expected_reward = -(delay / 20.0 + beta * slack / 20.0)

        passed = abs(reward - expected_reward) < EPS
        if not passed:
            all_passed = False

        print(
            f"{step_idx:<6}"
            f"{delay:<12.4f}"
            f"{deadline:<12.4f}"
            f"{slack:<12.4f}"
            f"{reward:<16.6f}"
            f"{expected_reward:<16.6f}"
            f"{str(passed):<8}"
        )

        if terminated or truncated:
            break

        obs = next_obs

    print()
    print(f"Numerical tolerance used: {EPS:.0e}")

    if all_passed:
        print("All environment reward consistency checks passed.")
        print("- Reward matches: -(delay/20 + beta*slack/20)")
        print("- Formula successfully embedded into environment.")
    else:
        print("Some environment reward consistency checks failed.")


# ============================================================
# 主程式
# ============================================================

def main():
    test_env_reward_consistency(
        num_steps=10,
        seed=42,
        beta=10.0,
        use_fixed_action=False,  # 👉 想穩定一點可改 True
    )


if __name__ == "__main__":
    main()