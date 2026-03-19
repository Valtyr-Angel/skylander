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
        self.side_engine_bonus_accumulated = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_velocity_x = abs(obs[2])  # Vx
        self.initial_velocity_x = abs(obs[2])
        self.side_engine_bonus_accumulated = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get current state
        velocity_x = obs[2]
        velocity_y = obs[3]
        position_y = obs[1]  # Height

        # 1. Horizontal velocity correction bonus
        current_vx = abs(velocity_x)
        vx_reduction = self.prev_velocity_x - current_vx

        bonus_hz_correction = 0.0
        if vx_reduction > 0 and position_y > 0.5:  # Only reward when above 0.5 height
            bonus_hz_correction = 0.5 * vx_reduction
            reward += bonus_hz_correction

        # 2. Low velocity maintenance bonus
        velocity_magnitude = (velocity_x**2 + velocity_y**2) ** 0.5
        if velocity_magnitude < 0.3 and position_y < 1.0:
            reward += 0.2 * (0.3 - velocity_magnitude)

        # 3. Side engine usage bonus when horizontal correction needed
        if action in [1, 3]:  # Left or Right engine
            if current_vx > 0.1:
                reward += 0.1

        # 4. Crash penalty (applied at end)
        # We'll handle this in the callback

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


# Create wrapped environment with reward shaping
env = gym.make("LunarLander-v3")
env = RewardShapingEnv(env)

callback = BenchmarkCallback(eval_freq=5000)

print("Loading v3 with reward shaping...")

# Load v3 model with reward shaping
model = PPO.load("ppo_lunarlander_v3_checkpoint.zip", env=env, learning_rate=0.0001)

total_timesteps = 200000

print(f"\n=== Starting Reward Shaping Training ===")
print("Components:")
print("  1. Horizontal velocity correction bonus")
print("  2. Low velocity maintenance bonus")
print("  3. Side engine usage bonus")

model.learn(total_timesteps=total_timesteps, progress_bar=True)

model.save("ppo_reward_shaped")
print("Model saved!")

model.save("ppo_lunarlander_reward_shaped.zip")
print("Backup saved!")
