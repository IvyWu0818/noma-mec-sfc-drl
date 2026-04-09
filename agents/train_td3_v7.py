import os
import numpy as np
import json
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from envs.iiot_env_v7 import IIoTEnvV7

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)


class V7MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {
            "episode_rewards": [],
            "episode_avg_delay": [],
            "episode_avg_slack": [],
            "episode_timeout_ratio": [],
            "episode_avg_cpu_viol": [],
            "episode_avg_t_ul": [],
            "episode_avg_t_comp": [],
            "episode_avg_t_link": [],
        }
        self._reset_ep()

    def _reset_ep(self):
        self._ep_reward = 0.0
        self._ep_delays, self._ep_slacks = [], []
        self._ep_cpu_viols = []
        self._ep_t_ul, self._ep_t_comp, self._ep_t_link = [], [], []
        self._ep_timeouts = 0
        self._ep_len = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        info = self.locals["infos"][0]

        self._ep_reward += reward
        self._ep_len += 1

        if "delay" in info:    self._ep_delays.append(info["delay"])
        if "slack" in info:
            self._ep_slacks.append(info["slack"])
            if info["slack"] > 0: self._ep_timeouts += 1
        if "cpu_viol" in info: self._ep_cpu_viols.append(info["cpu_viol"])
        if "t_ul" in info:     self._ep_t_ul.append(info["t_ul"])
        if "t_comp" in info:   self._ep_t_comp.append(info["t_comp"])
        if "t_link" in info:   self._ep_t_link.append(info["t_link"])

        if self.locals["dones"][0]:
            def _mean(lst): return float(np.mean(lst)) if lst else 0.0
            self.metrics["episode_rewards"].append(float(self._ep_reward))
            self.metrics["episode_avg_delay"].append(_mean(self._ep_delays))
            self.metrics["episode_avg_slack"].append(_mean(self._ep_slacks))
            self.metrics["episode_timeout_ratio"].append(self._ep_timeouts / max(self._ep_len, 1))
            self.metrics["episode_avg_cpu_viol"].append(_mean(self._ep_cpu_viols))
            self.metrics["episode_avg_t_ul"].append(_mean(self._ep_t_ul))
            self.metrics["episode_avg_t_comp"].append(_mean(self._ep_t_comp))
            self.metrics["episode_avg_t_link"].append(_mean(self._ep_t_link))
            self._reset_ep()

        return True


def main():
    env = Monitor(IIoTEnvV7(), "logs/v7_")
    n_act = env.action_space.shape[-1]

    action_noise = NormalActionNoise(
        mean=np.zeros(n_act),
        sigma=0.12 * np.ones(n_act)
    )

    callback = V7MetricsCallback()

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

    print("🚀 V7 (修正版) 訓練啟動")
    print("   Reward: 1.0·delay + 12.0·slack + 5.0·cpu_viol + 0.5·t_comp  (scale=50)")
    print("   f_alloc: 自適應下限 max(cpu/deadline, cap×0.15)")
    model.learn(total_timesteps=350_000, callback=callback)
    model.save("models/td3_iiot_v7_final")

    output_path = "results/td3_v7_training_metrics.json"
    with open(output_path, "w") as f:
        json.dump(callback.metrics, f, indent=2)
    print(f"✅ V7 數據已存至 {output_path}")


if __name__ == "__main__":
    main()
