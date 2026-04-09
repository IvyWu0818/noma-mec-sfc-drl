import os
import numpy as np
import json
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from envs.iiot_env_v5 import IIoTEnvV5

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

class V5MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {"episode_rewards": [], "episode_avg_delay": [], "episode_avg_slack": [], "episode_timeout_ratio": [], "actor_losses": [], "critic_losses": []}
        self._ep_reward, self._ep_delays, self._ep_slacks, self._ep_timeouts, self._ep_len = 0, [], [], 0, 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        info = self.locals["infos"][0]
        self._ep_reward += reward
        self._ep_len += 1
        if "delay" in info: self._ep_delays.append(info["delay"])
        if "slack" in info:
            self._ep_slacks.append(info["slack"])
            if info["slack"] > 0: self._ep_timeouts += 1

        if self.locals["dones"][0]:
            self.metrics["episode_rewards"].append(float(self._ep_reward))
            self.metrics["episode_avg_delay"].append(float(np.mean(self._ep_delays)))
            self.metrics["episode_avg_slack"].append(float(np.mean(self._ep_slacks)))
            self.metrics["episode_timeout_ratio"].append(self._ep_timeouts / self._ep_len)
            self._ep_reward, self._ep_delays, self._ep_slacks, self._ep_timeouts, self._ep_len = 0, [], [], 0, 0
        return True

def main():
    env = Monitor(IIoTEnvV5(), "logs/v5_")
    n_act = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_act), sigma=0.12 * np.ones(n_act))
    callback = V5MetricsCallback()

    model = TD3(
        "MlpPolicy", env, learning_rate=2.5e-4, buffer_size=400000,
        learning_starts=15000, batch_size=256, tau=0.005, gamma=0.99,
        action_noise=action_noise, policy_kwargs=dict(net_arch=[400, 300]),
        verbose=1, device="cpu"
    )

    print("🚀 V5 壓力測試訓練啟動...")
    model.learn(total_timesteps=350000, callback=callback)
    model.save("models/td3_iiot_v5_final")
    
    with open("results/td3_v5_training_metrics.json", "w") as f:
        json.dump(callback.metrics, f, indent=2)
    print("✅ V5 數據已存至 results/td3_v5_training_metrics.json")

if __name__ == "__main__":
    main()