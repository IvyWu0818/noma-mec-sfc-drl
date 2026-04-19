import os
import json
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from envs.iiot_env_v10 import IIoTEnvV10

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)


class V10MetricsCallback(BaseCallback):
    """
    收集每 episode (100 tasks) 的訓練指標。

    V10 新增指標：
      - episode_avg_rho            : 平均部分卸載比例 ρ_u
      - episode_avg_queue_delta    : 平均佇列變化率 |Δqueue|
      - episode_avg_t_link         : 平均串接延遲（異質節點矩陣）
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {
            "episode_rewards":               [],
            "episode_avg_delay":             [],
            "episode_avg_slack":             [],
            "episode_timeout_ratio":         [],
            "episode_avg_cpu_viol":          [],
            "episode_avg_t_ul":              [],
            "episode_avg_t_comp":            [],
            "episode_avg_t_link":            [],
            "episode_avg_deadline_pressure": [],
            "episode_task_mix_urgent_ratio": [],
            # V9 繼承
            "episode_avg_sinr":                  [],
            "episode_avg_channel_rate":          [],
            "episode_channel_overflow_ratio":    [],
            "episode_avg_channel_entropy":       [],
            # V10 新增
            "episode_avg_rho":               [],
            "episode_avg_queue_delta":       [],
        }
        self._reset_ep()

    def _reset_ep(self):
        self._ep_reward = 0.0
        self._ep_delays, self._ep_slacks = [], []
        self._ep_cpu_viols = []
        self._ep_t_ul, self._ep_t_comp, self._ep_t_link = [], [], []
        self._ep_deadline_pressure = []
        self._ep_task_type_ids = []
        self._ep_sinrs = []
        self._ep_channel_rates = []
        self._ep_channel_overflows = []
        self._ep_assigned_chs = []
        self._ep_rhos = []
        self._ep_queue_deltas = []
        self._ep_timeouts = 0
        self._ep_len = 0

    def _channel_entropy(self, assignments: list) -> float:
        if not assignments:
            return 0.0
        from collections import Counter
        counts = Counter(assignments)
        n = len(assignments)
        entropy = 0.0
        for c in counts.values():
            p = c / n
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        info   = self.locals["infos"][0]

        self._ep_reward += reward
        self._ep_len   += 1

        if "delay"    in info: self._ep_delays.append(info["delay"])
        if "slack"    in info:
            self._ep_slacks.append(info["slack"])
            if info["slack"] > 0: self._ep_timeouts += 1
        if "cpu_viol" in info: self._ep_cpu_viols.append(info["cpu_viol"])
        if "t_ul"     in info: self._ep_t_ul.append(info["t_ul"])
        if "t_comp"   in info: self._ep_t_comp.append(info["t_comp"])
        if "t_link"   in info: self._ep_t_link.append(info["t_link"])
        if "deadline_pressure" in info:
            self._ep_deadline_pressure.append(info["deadline_pressure"])
        if "task_type_id" in info:
            self._ep_task_type_ids.append(info["task_type_id"])
        if "sinr"     in info: self._ep_sinrs.append(info["sinr"])
        if "ru_k"     in info: self._ep_channel_rates.append(info["ru_k"])
        if "channel_overflow" in info:
            self._ep_channel_overflows.append(info["channel_overflow"])
        if "assigned_ch" in info:
            self._ep_assigned_chs.append(info["assigned_ch"])
        if "rho"      in info:
            self._ep_rhos.append(info["rho"])

        # 佇列 delta：從 obs 中提取（obs[-3:] 為 dq0, dq1, dq2）
        obs = self.locals.get("new_obs")
        if obs is not None:
            dq = obs[0][-3:]   # shape: (3,)
            self._ep_queue_deltas.append(float(np.mean(np.abs(dq))))

        if self.locals["dones"][0]:
            def _mean(lst): return float(np.mean(lst)) if lst else 0.0

            urgent_ratio = 0.0
            if self._ep_task_type_ids:
                urgent_ratio = float(
                    np.mean(np.array(self._ep_task_type_ids) == 0)
                )

            self.metrics["episode_rewards"].append(float(self._ep_reward))
            self.metrics["episode_avg_delay"].append(_mean(self._ep_delays))
            self.metrics["episode_avg_slack"].append(_mean(self._ep_slacks))
            self.metrics["episode_timeout_ratio"].append(
                self._ep_timeouts / max(self._ep_len, 1)
            )
            self.metrics["episode_avg_cpu_viol"].append(_mean(self._ep_cpu_viols))
            self.metrics["episode_avg_t_ul"].append(_mean(self._ep_t_ul))
            self.metrics["episode_avg_t_comp"].append(_mean(self._ep_t_comp))
            self.metrics["episode_avg_t_link"].append(_mean(self._ep_t_link))
            self.metrics["episode_avg_deadline_pressure"].append(
                _mean(self._ep_deadline_pressure)
            )
            self.metrics["episode_task_mix_urgent_ratio"].append(urgent_ratio)
            self.metrics["episode_avg_sinr"].append(_mean(self._ep_sinrs))
            self.metrics["episode_avg_channel_rate"].append(_mean(self._ep_channel_rates))
            self.metrics["episode_channel_overflow_ratio"].append(
                _mean(self._ep_channel_overflows)
            )
            self.metrics["episode_avg_channel_entropy"].append(
                self._channel_entropy(self._ep_assigned_chs)
            )
            self.metrics["episode_avg_rho"].append(_mean(self._ep_rhos))
            self.metrics["episode_avg_queue_delta"].append(_mean(self._ep_queue_deltas))
            self._reset_ep()

        return True


def main():
    env = Monitor(IIoTEnvV10(), "logs/v10_")
    n_act = env.action_space.shape[-1]  # 16

    # sigma 略調整（16 維）
    action_noise = NormalActionNoise(
        mean=np.zeros(n_act),
        sigma=0.13 * np.ones(n_act)
    )

    callback = V10MetricsCallback()

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

    print("🚀 V10 訓練啟動")
    print("   obs=21 (含 ch_rem×3 + Δqueue×3), action=16 (含 rho 部分卸載)")
    print("   NOMA: K=3 channels, M=2 max/channel, Gumbel softmax 採樣")
    print("   算力投影: 硬約束可行化（公式5、6）")
    print("   串接延遲: 異質 backhaul 矩陣（公式12）")
    print("   Reward: delay + beta*slack + 5*cpu_viol + 0.5*t_comp")
    print("           + 1.5*deadline_pressure + 8.0*channel_overflow（↑ 強化）")
    print("   Total timesteps: 1,000,000 | Episode length: 100 tasks")
    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save("models/td3_iiot_v10_final")

    output_path = "results/td3_v10_training_metrics.json"
    with open(output_path, "w") as f:
        json.dump(callback.metrics, f, indent=2)
    print(f"✅ V10 數據已存至 {output_path}")


if __name__ == "__main__":
    main()
