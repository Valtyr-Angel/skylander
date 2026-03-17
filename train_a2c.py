import gymnasium as gym
from stable_baselines3 import A2C
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

print("Creating A2C model with tuned hyperparameters...")

model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    n_steps=512,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[64, 64]),
    use_sde=False,
    normalize_advantage=True,
)

total_timesteps = 500000
checkpoint_freq = 200000

print(f"\n=== Starting A2C Training ===")
print(f"Training for {total_timesteps} steps...")

model.learn(total_timesteps=total_timesteps, progress_bar=True)

model.save("a2c_lunarlander")
print("Model saved!")

backup_name = f"a2c_lunarlander_200k.zip"
model.save(backup_name)
print(f"Backup saved: {backup_name}")
