import os
import json
import csv
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise

from envs.iiot_env_v3 import IIoTEnvV3


# ============================================================
# 路徑設定
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/eval_v3", exist_ok=True)
os.makedirs("logs/eval_results_v3", exist_ok=True)
os.makedirs("logs/tb_td3_v3", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 自訂 callback：收集訓練曲線
# ============================================================

class TrainingMetricsCallback(BaseCallback):
    """
    記錄：
    1. episode rewards
    2. actor loss
    3. critic loss
    4. 每個 episode 的 delay/slack/timeout
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        # episode-level
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_avg_delay = []
        self.episode_avg_slack = []
        self.episode_timeout_ratio = []

        # training-level
        self.actor_losses = []
        self.critic_losses = []
        self.train_steps = []

        # 暫存目前 episode 的資訊
        self._current_reward = 0.0
        self._current_len = 0
        self._current_delays = []
        self._current_slacks = []
        self._current_timeouts = 0

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print("TrainingMetricsCallback started.")

    def _on_step(self) -> bool:
        # 單環境訓練，因此取 [0]
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]

        self._current_reward += reward
        self._current_len += 1

        if "delay" in info:
            delay = float(info["delay"])
            self._current_delays.append(delay)

        if "slack" in info:
            slack = float(info["slack"])
            self._current_slacks.append(slack)

        if "delay" in info and "deadline" in info:
            if float(info["delay"]) > float(info["deadline"]):
                self._current_timeouts += 1

        # 盡量從 logger 抓 loss
        logger_values = getattr(self.model.logger, "name_to_value", {})

        actor_loss = logger_values.get("train/actor_loss", None)
        critic_loss = logger_values.get("train/critic_loss", None)

        if actor_loss is not None:
            self.actor_losses.append(float(actor_loss))
            self.train_steps.append(self.num_timesteps)

        if critic_loss is not None:
            self.critic_losses.append(float(critic_loss))

        if done:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_len)

            if len(self._current_delays) > 0:
                self.episode_avg_delay.append(float(np.mean(self._current_delays)))
            else:
                self.episode_avg_delay.append(0.0)

            if len(self._current_slacks) > 0:
                self.episode_avg_slack.append(float(np.mean(self._current_slacks)))
            else:
                self.episode_avg_slack.append(0.0)

            if self._current_len > 0:
                self.episode_timeout_ratio.append(self._current_timeouts / self._current_len)
            else:
                self.episode_timeout_ratio.append(0.0)

            # reset episode buffer
            self._current_reward = 0.0
            self._current_len = 0
            self._current_delays = []
            self._current_slacks = []
            self._current_timeouts = 0

        return True

    def save_metrics(self, output_json="results/td3_v3_training_metrics.json", output_csv="results/td3_v3_episode_metrics.csv"):
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_avg_delay": self.episode_avg_delay,
            "episode_avg_slack": self.episode_avg_slack,
            "episode_timeout_ratio": self.episode_timeout_ratio,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "train_steps": self.train_steps,
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "episode_reward",
                "episode_length",
                "avg_delay",
                "avg_slack",
                "timeout_ratio",
            ])
            for i in range(len(self.episode_rewards)):
                writer.writerow([
                    i + 1,
                    self.episode_rewards[i],
                    self.episode_lengths[i],
                    self.episode_avg_delay[i],
                    self.episode_avg_slack[i],
                    self.episode_timeout_ratio[i],
                ])

        print(f"Saved training metrics to: {output_json}")
        print(f"Saved episode metrics to: {output_csv}")


# ============================================================
# main
# ============================================================

def main():
    env = Monitor(IIoTEnvV3(num_tasks=10, beta=10.0), "logs/")
    eval_env = Monitor(IIoTEnvV3(num_tasks=10, beta=10.0), "logs/eval_v3/")

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    n_actions = env.action_space.shape[-1]

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_td3_v3/",
        log_path="./logs/eval_results_v3/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    metrics_callback = TrainingMetricsCallback(verbose=1)
    callback = CallbackList([eval_callback, metrics_callback])

    model = TD3(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        tensorboard_log="./logs/tb_td3_v3/",
        action_noise=action_noise,
    )

    model.learn(
        total_timesteps=50000,
        callback=callback,
        log_interval=10,
    )

    model.save("models/td3_iiot_env_v3_final")
    print("TD3 V3 training finished.")

    metrics_callback.save_metrics()


if __name__ == "__main__":
    main()