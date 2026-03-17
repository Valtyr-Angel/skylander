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

# IMPROVED HYPERPARAMETERS FOR V2 TRAINING
# Learning rate: 0.0001 (was 0.0003) - more stable
# Network: 128 neurons (was 64) - more capacity
# Clip range: 0.15 (was 0.2) - more conservative
# Entropy: 0.005 (was 0.01) - less exploration

if os.path.exists("ppo_lunarlander.zip"):
    print("Loading existing model...")
    model = PPO.load("ppo_lunarlander", env=env)
else:
    print("Creating new model with IMPROVED hyperparameters...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

total_timesteps = 200000  # 200k per interval
checkpoint_freq = 200000  # Save every 200k
steps_trained = 0

print(f"\n=== Starting V2 Training ===")
print(f"Training for {total_timesteps} steps...")

while steps_trained < total_timesteps:
    remaining = total_timesteps - steps_trained
    train_steps = min(checkpoint_freq, remaining)

    print(f"\n=== Training {steps_trained + train_steps}/{total_timesteps} steps ===")
    model.learn(total_timesteps=train_steps, progress_bar=True)
    steps_trained += train_steps

    backup_name = f"ppo_lunarlander_{steps_trained}_v2.zip"
    model.save(backup_name)
    print(f"Backup saved: {backup_name}")

model.save("ppo_lunarlander")
print("Model saved!")
