import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

checkpoint = "ppo_lunarlander_100000.zip"
print(f"Analyzing: {checkpoint}")
print("=" * 60)

env = gym.make("LunarLander-v3")
model = PPO.load(checkpoint)

num_episodes = 20
rewards = []
successes = 0
crashes = 0

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)
    if total_reward >= 100:
        successes += 1
        print(f"Episode {episode + 1}: {total_reward:.2f} - SUCCESS")
    elif total_reward >= 0:
        print(f"Episode {episode + 1}: {total_reward:.2f} - PARTIAL")
    else:
        crashes += 1
        print(f"Episode {episode + 1}: {total_reward:.2f} - CRASH")

print("=" * 60)
print(f"Mean: {np.mean(rewards):.2f}")
print(f"Success: {successes}/{num_episodes} ({100 * successes / num_episodes:.0f}%)")
print(f"Crash: {crashes}/{num_episodes} ({100 * crashes / num_episodes:.0f}%)")

env.close()
