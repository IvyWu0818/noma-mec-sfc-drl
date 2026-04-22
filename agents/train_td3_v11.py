import os
import json
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from envs.iiot_env_v11 import IIoTEnvV11

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)


class V11MetricsCallback(BaseCallback):
    """
    V11 新增追蹤：
      - episode_slot_overflow_ratio : per-slot 真實 overflow 比例
        （應從 V10 的 0.956 降至接近 0，這才是學習成功的標誌）
      - episode_avg_slot_id         : 平均 slot 編號（診斷用）
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
            "episode_avg_sinr":              [],
            "episode_avg_channel_rate":      [],
            "episode_channel_overflow_ratio":[],
            "episode_avg_channel_entropy":   [],
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
        self._ep_sinrs, self._ep_channel_rates = [], []
        self._ep_channel_overflows = []
        self._ep_assigned_chs = []
        self._ep_rhos = []
        self._ep_queue_deltas = []
        self._ep_timeouts = 0
        self._ep_len = 0

    def _channel_entropy(self, assignments):
        if not assignments:
            return 0.0
        from collections import Counter
        counts = Counter(assignments)
        n = len(assignments)
        return -sum((c / n) * np.log2(c / n) for c in counts.values())

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
        if "rho"      in info: self._ep_rhos.append(info["rho"])

        obs = self.locals.get("new_obs")
        if obs is not None:
            dq = obs[0][-3:]
            self._ep_queue_deltas.append(float(np.mean(np.abs(dq))))

        if self.locals["dones"][0]:
            def _mean(lst): return float(np.mean(lst)) if lst else 0.0

            urgent_ratio = 0.0
            if self._ep_task_type_ids:
                urgent_ratio = float(np.mean(np.array(self._ep_task_type_ids) == 0))

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
    env = Monitor(IIoTEnvV11(), "logs/v11_")
    n_act = env.action_space.shape[-1]  # 16

    action_noise = NormalActionNoise(
        mean=np.zeros(n_act),
        sigma=0.13 * np.ones(n_act)
    )

    callback = V11MetricsCallback()

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
        device="cuda",
    )

    print("🚀 V11 訓練啟動")
    print(f"   obs=21, action=16 (同 V10)")
    print(f"   NOMA: K=3 channels, M=2, "
          f"SLOT_SIZE={IIoTEnvV11.SLOT_TASK_SIZE} tasks/slot")
    print(f"   Channel overflow: per-slot 定義（V11 核心修復）")
    print(f"   100 tasks → ~{100 // IIoTEnvV11.SLOT_TASK_SIZE} slots，"
          f"每 slot 最多 {IIoTEnvV11.SLOT_TASK_SIZE} 任務不衝突")
    print(f"   理論最低 overflow = 0（agent 完全可以學到）")
    print(f"   channel_overflow 懲罰: 5.0（從 V10 的 8.0 降回合理值）")
    print(f"   Total timesteps: 1,000,000 | Episode length: 100 tasks")

    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save("models/td3_iiot_v11_final")

    output_path = "results/td3_v11_training_metrics.json"
    with open(output_path, "w") as f:
        json.dump(callback.metrics, f, indent=2)
    print(f"✅ V11 數據已存至 {output_path}")


if __name__ == "__main__":
    main()