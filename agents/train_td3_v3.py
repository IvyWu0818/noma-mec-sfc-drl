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
# 路徑
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("models/best_td3_v3", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/eval_v3", exist_ok=True)
os.makedirs("logs/eval_results_v3", exist_ok=True)
os.makedirs("logs/tb_td3_v3", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# Callback: 記錄訓練指標
# ============================================================

class TrainingMetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        # episode-level
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_avg_delay = []
        self.episode_avg_slack = []
        self.episode_timeout_ratio = []
        self.episode_avg_cost = []
        self.episode_avg_cpu_violation = []

        # update-level
        self.actor_losses = []
        self.critic_losses = []
        self.train_steps = []

        # current episode buffer
        self._ep_reward = 0.0
        self._ep_len = 0
        self._ep_delays = []
        self._ep_slacks = []
        self._ep_timeouts = 0
        self._ep_costs = []
        self._ep_cpu_violations = []

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]

        self._ep_reward += reward
        self._ep_len += 1

        if "delay" in info:
            self._ep_delays.append(float(info["delay"]))
        if "slack" in info:
            self._ep_slacks.append(float(info["slack"]))
        if "delay" in info and "deadline" in info:
            if float(info["delay"]) > float(info["deadline"]):
                self._ep_timeouts += 1
        if "cost" in info:
            self._ep_costs.append(float(info["cost"]))
        if "cpu_violation" in info:
            self._ep_cpu_violations.append(float(info["cpu_violation"]))

        # 從 logger 取訓練 loss
        logger_values = getattr(self.model.logger, "name_to_value", {})
        actor_loss = logger_values.get("train/actor_loss", None)
        critic_loss = logger_values.get("train/critic_loss", None)

        if actor_loss is not None:
            self.actor_losses.append(float(actor_loss))
            self.train_steps.append(self.num_timesteps)

        if critic_loss is not None:
            self.critic_losses.append(float(critic_loss))

        if done:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_len)

            self.episode_avg_delay.append(
                float(np.mean(self._ep_delays)) if self._ep_delays else 0.0
            )
            self.episode_avg_slack.append(
                float(np.mean(self._ep_slacks)) if self._ep_slacks else 0.0
            )
            self.episode_timeout_ratio.append(
                self._ep_timeouts / self._ep_len if self._ep_len > 0 else 0.0
            )

            self.episode_avg_cost.append(
                float(np.mean(self._ep_costs)) if self._ep_costs else 0.0
            )
            self.episode_avg_cpu_violation.append(
                float(np.mean(self._ep_cpu_violations)) if self._ep_cpu_violations else 0.0
            )

            self._ep_reward = 0.0
            self._ep_len = 0
            self._ep_delays = []
            self._ep_slacks = []
            self._ep_timeouts = 0
            self._ep_costs = []
            self._ep_cpu_violations = []

        return True

    def save_metrics(
        self,
        json_path="results/td3_v3_training_metrics.json",
        csv_path="results/td3_v3_episode_metrics.csv",
    ):
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_avg_delay": self.episode_avg_delay,
            "episode_avg_slack": self.episode_avg_slack,
            "episode_timeout_ratio": self.episode_timeout_ratio,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "train_steps": self.train_steps,
            "episode_avg_cost": self.episode_avg_cost,
            "episode_avg_cpu_violation": self.episode_avg_cpu_violation,

        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "episode_reward",
                "episode_length",
                "avg_delay",
                "avg_slack",
                "timeout_ratio",
                "avg_cost",
                "avg_cpu_violation",
            ])
            for i in range(len(self.episode_rewards)):
                writer.writerow([
                    i + 1,
                    self.episode_rewards[i],
                    self.episode_lengths[i],
                    self.episode_avg_delay[i],
                    self.episode_avg_slack[i],
                    self.episode_timeout_ratio[i],
                    self.episode_avg_cost[i],
                    self.episode_avg_cpu_violation[i],
                ])

        print(f"Saved metrics: {json_path}")
        print(f"Saved episode table: {csv_path}")


# ============================================================
# main
# ============================================================

def main():
    # 訓練環境
    env = Monitor(
        IIoTEnvV3(
            num_tasks=100,
            beta=5.0,
            seed=42,
            timeout_penalty=10.0,
            cpu_violation_penalty=5.0,
            reward_scale=20.0,
        ),
        "logs/"
    )

    eval_env = Monitor(
        IIoTEnvV3(
            num_tasks=100,
            beta=5.0,
            seed=123,
            timeout_penalty=10.0,
            cpu_violation_penalty=5.0,
            reward_scale=20.0,
        ),
        "logs/eval_v3/"
    )

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    n_actions = env.action_space.shape[-1]

    # 比原本 sigma=0.1 稍微大一點，增加探索
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.15 * np.ones(n_actions),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_td3_v3/",
        log_path="./logs/eval_results_v3/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    metrics_callback = TrainingMetricsCallback(verbose=1)
    callback = CallbackList([eval_callback, metrics_callback])

    # 網路加深，lr 降低，learning_starts 拉高
    model = TD3(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        tensorboard_log="./logs/tb_td3_v3/",
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(
        total_timesteps=200000,
        callback=callback,
        log_interval=10,
        progress_bar=False,
    )

    model.save("models/td3_iiot_env_v3_final")
    print("TD3 V3 training finished.")

    metrics_callback.save_metrics()


if __name__ == "__main__":
    main()