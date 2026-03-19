import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os


class RewardShapingEnv(gym.Wrapper):
    """Custom environment with reward shaping for Lunar Lander"""

    def __init__(self, env):
        super().__init__(env)
        self.prev_velocity_x = None
        self.initial_velocity_x = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_velocity_x = abs(obs[2])
        self.initial_velocity_x = abs(obs[2])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        velocity_x = obs[2]
        velocity_y = obs[3]
        position_y = obs[1]

        current_vx = abs(velocity_x)
        vx_reduction = self.prev_velocity_x - current_vx

        if vx_reduction > 0 and position_y > 0.5:
            reward += 0.5 * vx_reduction

        velocity_magnitude = (velocity_x**2 + velocity_y**2) ** 0.5
        if velocity_magnitude < 0.3 and position_y < 1.0:
            reward += 0.2 * (0.3 - velocity_magnitude)

        if action in [1, 3]:
            if current_vx > 0.1:
                reward += 0.1

        self.prev_velocity_x = current_vx

        return obs, reward, terminated, truncated, info


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
env = RewardShapingEnv(env)

callback = BenchmarkCallback(eval_freq=5000)

print("Loading reward shaped model for more training...")

model = PPO.load("ppo_lunarlander_reward_shaped.zip", env=env, learning_rate=0.0001)

total_timesteps = 200000
checkpoint_freq = 100000
steps_trained = 0

print(f"\n=== Continuing Reward Shaping Training ===")

while steps_trained < total_timesteps:
    remaining = total_timesteps - steps_trained
    train_steps = min(checkpoint_freq, remaining)

    print(f"\n=== Training {steps_trained + train_steps}/{total_timesteps} steps ===")
    model.learn(total_timesteps=train_steps, progress_bar=True)
    steps_trained += train_steps

    backup_name = f"ppo_reward_shaped_{steps_trained}.zip"
    model.save(backup_name)
    print(f"Backup saved: {backup_name}")

model.save("ppo_lunarlander_reward_shaped")
print("Model saved!")
