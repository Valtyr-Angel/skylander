import gymnasium as gym
from stable_baselines3 import PPO
import time

env = gym.make("LunarLander-v3", render_mode="human")

model = PPO.load("ppo_lunarlander")

obs, info = env.reset()
total_reward = 0
frame_skip = 5  # Speed up 5x

for episode in range(5):
    while True:
        for _ in range(frame_skip):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        env.render()

        if terminated or truncated:
            print(f"Episode {episode + 1} reward: {total_reward:.1f}")
            total_reward = 0
            obs, info = env.reset()
            break

env.close()
