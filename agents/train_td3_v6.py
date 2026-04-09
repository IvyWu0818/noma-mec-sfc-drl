import os
import numpy as np
import json
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from envs.iiot_env_v6 import IIoTEnvV6

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

class V6MetricsCallback(BaseCallback):
    """
    收集每 episode 的訓練指標：
    - episode_rewards      : 累計 reward
    - episode_avg_delay    : 平均端到端延遲
    - episode_avg_slack    : 平均超時 slack
    - episode_timeout_ratio: 超時任務比例
    - episode_avg_cpu_viol : 平均 CPU 違規量 (V6 新增)
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {
            "episode_rewards": [],
            "episode_avg_delay": [],
            "episode_avg_slack": [],
            "episode_timeout_ratio": [],
            "episode_avg_cpu_viol": [],  # V6 新增
        }
        self._reset_ep()

    def _reset_ep(self):
        self._ep_reward = 0.0
        self._ep_delays = []
        self._ep_slacks = []
        self._ep_cpu_viols = []
        self._ep_timeouts = 0
        self._ep_len = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        info = self.locals["infos"][0]

        self._ep_reward += reward
        self._ep_len += 1

        if "delay" in info:
            self._ep_delays.append(info["delay"])
        if "slack" in info:
            self._ep_slacks.append(info["slack"])
            if info["slack"] > 0:
                self._ep_timeouts += 1
        if "cpu_viol" in info:
            self._ep_cpu_viols.append(info["cpu_viol"])

        if self.locals["dones"][0]:
            self.metrics["episode_rewards"].append(float(self._ep_reward))
            self.metrics["episode_avg_delay"].append(float(np.mean(self._ep_delays)) if self._ep_delays else 0.0)
            self.metrics["episode_avg_slack"].append(float(np.mean(self._ep_slacks)) if self._ep_slacks else 0.0)
            self.metrics["episode_timeout_ratio"].append(self._ep_timeouts / max(self._ep_len, 1))
            self.metrics["episode_avg_cpu_viol"].append(float(np.mean(self._ep_cpu_viols)) if self._ep_cpu_viols else 0.0)
            self._reset_ep()

        return True


def main():
    env = Monitor(IIoTEnvV6(), "logs/v6_")
    n_act = env.action_space.shape[-1]

    # Action noise: 略降 sigma (V5 0.12 → V6 0.10)，obs 維度增加後探索更穩定
    action_noise = NormalActionNoise(
        mean=np.zeros(n_act),
        sigma=0.10 * np.ones(n_act)
    )

    callback = V6MetricsCallback()

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        buffer_size=400_000,
        learning_starts=15_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[400, 300]),
        verbose=1,
        device="cpu",
    )

    print("🚀 V6 訓練啟動 (obs=14, action=12, fixes: SINR obs, Q_t, f_alloc coupling)")
    model.learn(total_timesteps=350_000, callback=callback)
    model.save("models/td3_iiot_v6_final")

    output_path = "results/td3_v6_training_metrics.json"
    with open(output_path, "w") as f:
        json.dump(callback.metrics, f, indent=2)
    print(f"✅ V6 數據已存至 {output_path}")


if __name__ == "__main__":
    main()
