import os
import json
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from envs.iiot_env_v14 import IIoTEnvV14, TX_POWER_W, NOISE_POWER_W, BANDWIDTH_MHZ

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)


class V14MetricsCallback(BaseCallback):
    """
    V14: 同 V13 追蹤項目
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
            "episode_avg_sinr_db":           [],
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
        self._ep_sinrs_db, self._ep_channel_rates = [], []
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
        if "sinr_db"  in info: self._ep_sinrs_db.append(info["sinr_db"])
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
            self.metrics["episode_avg_sinr_db"].append(_mean(self._ep_sinrs_db))
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
    env = Monitor(IIoTEnvV14(), "logs/v14_")
    n_act = env.action_space.shape[-1]  # 16

    action_noise = NormalActionNoise(
        mean=np.zeros(n_act),
        sigma=0.13 * np.ones(n_act)
    )

    callback = V14MetricsCallback()

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

    # 5G NR 參數驗證輸出
    _h_avg = 1.0  # E[|h|²] = 1 (Exp(1))
    _sinr_avg = TX_POWER_W * _h_avg / NOISE_POWER_W
    _rate_avg = BANDWIDTH_MHZ * np.log2(1.0 + _sinr_avg)
    _t_ul_example = 40.0 / _rate_avg  # 40 Kbits / rate

    print("=" * 60)
    print("V14 Training (overflow penalty 15.0 + SIC degradation + norm obs)")
    print("=" * 60)
    print(f"  TX Power     : {TX_POWER_W*1000:.0f} mW = {10*np.log10(TX_POWER_W*1000):.1f} dBm (3GPP PC3)")
    print(f"  Bandwidth/ch : {BANDWIDTH_MHZ} MHz  (total {BANDWIDTH_MHZ*3:.0f} MHz)")
    print(f"  E[SINR]      : {_sinr_avg:.1f} ({10*np.log10(_sinr_avg):.1f} dB)")
    print(f"  E[Rate]      : {_rate_avg:.1f} Mbps  |  t_ul(40Kb): {_t_ul_example:.2f} ms")
    print(f"  Overflow penalty : 15.0 (was 5.0) + ru_k x0.25 on overflow")
    print(f"  Obs[9-11]    : queue_load_norm [0,1] (was raw MCycles/ms)")
    print(f"[Obs=21, Action=16 | Total timesteps: 1,000,000]")
    print("=" * 60)

    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save("models/td3_iiot_v14_final")

    output_path = "results/td3_v14_training_metrics.json"
    with open(output_path, "w") as f:
        json.dump(callback.metrics, f, indent=2)
    print(f"V14 data saved to {output_path}")


if __name__ == "__main__":
    main()
