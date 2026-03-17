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

# Load v3 and train with increased exploration
print("Loading v3 model with increased exploration...")
model = PPO.load("ppo_lunarlander_v3.zip", env=env, learning_rate=0.0001, ent_coef=0.03)

total_timesteps = 100000
checkpoint_freq = 100000
steps_trained = 0

print(f"\n=== Starting v3 + Exploration Training ===")
print(f"Entropy coefficient: 0.03 (3x default)")

while steps_trained < total_timesteps:
    remaining = total_timesteps - steps_trained
    train_steps = min(checkpoint_freq, remaining)

    print(f"\n=== Training {steps_trained + train_steps}/{total_timesteps} steps ===")
    model.learn(total_timesteps=train_steps, progress_bar=True)
    steps_trained += train_steps

    backup_name = f"ppo_v3_explore_{steps_trained}.zip"
    model.save(backup_name)
    print(f"Backup saved: {backup_name}")

model.save("ppo_v3_explored")
print("Model saved!")
