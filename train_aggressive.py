import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os


class BenchmarkCallback(BaseCallback):
    def __init__(self, eval_freq=5000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_env = gym.make("LunarLander-v3")

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            mean_reward = self.evaluate()
            print(
                f"\n[BENCHMARK] Timesteps: {self.num_timesteps} | Mean Reward: {mean_reward:.2f}"
            )
        return True

    def evaluate(self, num_episodes=10):
        rewards = []
        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += float(reward)
                done = terminated or truncated
            rewards.append(episode_reward)
        return sum(rewards) / len(rewards)

    def _on_training_end(self):
        self.eval_env.close()


env = gym.make("LunarLander-v3")
callback = BenchmarkCallback(eval_freq=5000)

print("Creating PPO model with AGGRESSIVE EXPLORATION...")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,  # 5x higher - more exploration!
    policy_kwargs=dict(net_arch=[128, 128]),
)

print(f"\n=== Starting Aggressive Exploration Training ===")
print(f"Entropy coefficient: 0.05 (5x default)")

model.learn(total_timesteps=100000, progress_bar=True)

model.save("ppo_aggressive")
print("Model saved!")

model.save("ppo_lunarlander_aggressive_100k.zip")
print("Backup saved!")
