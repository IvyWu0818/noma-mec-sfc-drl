import os
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

from envs.iiot_env import IIoTEnv


# 建立資料夾
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/eval", exist_ok=True)

# ✅ 先建立 env（關鍵）
env = Monitor(IIoTEnv(num_tasks=10, beta=10.0), "logs/")
eval_env = Monitor(IIoTEnv(num_tasks=10, beta=10.0), "logs/eval/")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# ✅ 再拿 action space
n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

# 評估 callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_td3/",
    log_path="./logs/eval_results/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# 建立 TD3 模型
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
    tensorboard_log="./logs/tb_td3/",
    action_noise=action_noise,  # ✅ 這裡才用
)

# 訓練
model.learn(
    total_timesteps=50000,
    callback=eval_callback,
    log_interval=10,
)

# 儲存
model.save("models/td3_iiot_env_final")

print("TD3 training finished.")