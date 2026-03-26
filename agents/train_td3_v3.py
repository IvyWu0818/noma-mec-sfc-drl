import os
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

from envs.iiot_env_v3 import IIoTEnvV3


os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("logs/eval_v3", exist_ok=True)

env = Monitor(IIoTEnvV3(num_tasks=10, beta=10.0), "logs/")
eval_env = Monitor(IIoTEnvV3(num_tasks=10, beta=10.0), "logs/eval_v3/")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.1 * np.ones(n_actions)
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_td3_v3/",
    log_path="./logs/eval_results_v3/",
    eval_freq=5000,
    deterministic=True,
    render=False,
)

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
    callback=eval_callback,
    log_interval=10,
)

model.save("models/td3_iiot_env_v3_final")
print("TD3 V3 training finished.")