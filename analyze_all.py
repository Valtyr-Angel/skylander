import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os
import glob

checkpoints = sorted(glob.glob("ppo_lunarlander_*.zip"))
results = []

print("=" * 70)
print("ANALYZING ALL CHECKPOINTS")
print("=" * 70)

for checkpoint in checkpoints:
    name = os.path.basename(checkpoint).replace(".zip", "")
    print(f"\nAnalyzing {name}...")

    env = gym.make("LunarLander-v3")
    model = PPO.load(checkpoint)

    rewards = []
    successes = 0
    crashes = 0

    for episode in range(250):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        rewards.append(total_reward)
        if total_reward >= 100:
            successes += 1
        elif total_reward < 0:
            crashes += 1

    env.close()

    result = {
        "name": name,
        "steps": int(name.split("_")[-1]),
        "success_rate": successes / 250 * 100,
        "mean_reward": np.mean(rewards),
        "best_score": max(rewards),
        "worst_score": min(rewards),
        "crash_rate": crashes / 250 * 100,
    }
    results.append(result)
    print(
        f"  Success: {result['success_rate']:.0f}% | Mean: {result['mean_reward']:.1f} | Best: {result['best_score']:.1f} | Crash: {result['crash_rate']:.0f}%"
    )

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(
    f"{'Checkpoint':<20} {'Steps':>8} {'Success':>8} {'Mean':>10} {'Best':>10} {'Worst':>10} {'Crash':>8}"
)
print("-" * 70)

for r in results:
    print(
        f"{r['name']:<20} {r['steps']:>8,} {r['success_rate']:>7.0f}% {r['mean_reward']:>10.1f} {r['best_score']:>10.1f} {r['worst_score']:>10.1f} {r['crash_rate']:>7.0f}%"
    )

best = max(results, key=lambda x: x["success_rate"])
print("\n" + "=" * 70)
print(f"BEST MODEL: {best['name']}")
print(f"  Success Rate: {best['success_rate']:.0f}%")
print(f"  Mean Reward: {best['mean_reward']:.1f}")
print(f"  Best Score: {best['best_score']:.1f}")
print("=" * 70)
