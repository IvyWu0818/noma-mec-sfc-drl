import os
import json
import csv
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise

from envs.iiot_env_v4 import IIoTEnvV4

# ============================================================
# 1. 建立目錄
# ============================================================
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================================
# 2. Callback: 記錄訓練指標 (對齊你的畫圖腳本)
# ============================================================
class TrainingMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_avg_delay = []
        self.episode_avg_slack = []
        self.episode_timeout_ratio = []
        self.actor_losses = []
        self.critic_losses = []
        self.train_steps = []

        # Step buffer
        self._ep_reward = 0.0
        self._ep_delays = []
        self._ep_slacks = []
        self._ep_timeouts = 0
        self._ep_len = 0

    def _on_step(self) -> bool:
        # 讀取當前 Step 的回饋與 info
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]

        self._ep_reward += reward
        self._ep_len += 1
        
        if "delay" in info: self._ep_delays.append(float(info["delay"]))
        if "slack" in info: self._ep_slacks.append(float(info["slack"]))
        if "slack" in info and info["slack"] > 0: self._ep_timeouts += 1

        # 從 Logger 抓 Loss (TD3 專用標籤)
        logger_values = getattr(self.model.logger, "name_to_value", {})
        a_loss = logger_values.get("train/actor_loss")
        c_loss = logger_values.get("train/critic_loss")

        if a_loss is not None:
            self.actor_losses.append(float(a_loss))
            self.train_steps.append(self.num_timesteps)
        if c_loss is not None:
            self.critic_losses.append(float(c_loss))

        if done:
            self.episode_rewards.append(self._ep_reward)
            self.episode_avg_delay.append(float(np.mean(self._ep_delays)) if self._ep_delays else 0.0)
            self.episode_avg_slack.append(float(np.mean(self._ep_slacks)) if self._ep_slacks else 0.0)
            self.episode_timeout_ratio.append(self._ep_timeouts / self._ep_len if self._ep_len > 0 else 0.0)
            
            # Reset ep buffer
            self._ep_reward = 0.0
            self._ep_delays, self._ep_slacks = [], []
            self._ep_timeouts, self._ep_len = 0, 0
        return True

    def save_metrics(self, json_path="results/td3_v4_training_metrics.json"):
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_avg_delay": self.episode_avg_delay,
            "episode_avg_slack": self.episode_avg_slack,
            "episode_timeout_ratio": self.episode_timeout_ratio,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "train_steps": self.train_steps
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✅ V4 Metrics saved to: {json_path}")

# ============================================================
# 3. 主程式
# ============================================================
def main():
    env = Monitor(IIoTEnvV4(num_tasks=100, seed=42), "logs/v4_")
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    metrics_callback = TrainingMetricsCallback(verbose=1)

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=300000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[400, 300]),
        tensorboard_log="./logs/v4_tensorboard/",
        verbose=1,
        device="cpu"
    )

    print("🚀 開始 V4 訓練...")
    try:
        model.learn(total_timesteps=300000, callback=metrics_callback)
    except KeyboardInterrupt:
        print("🛑 訓練被手動停止，正在儲存目前的資料...")
    
    model.save("models/td3_iiot_v4_final")
    metrics_callback.save_metrics() # 確保結束時存檔

if __name__ == "__main__":
    main()