import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")

model = PPO.load("ppo_lunarlander")

num_episodes = 10
rewards = []

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
    print(f"Episode {episode + 1}: {total_reward:.2f}")

mean_reward = sum(rewards) / len(rewards)
print(f"\nMean Reward: {mean_reward:.2f}")
print(f"Min: {min(rewards):.2f} | Max: {max(rewards):.2f}")

env.close()
