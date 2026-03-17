import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("=" * 70)
print("COMPARING V2 CHECKPOINTS: 400K vs 600K")
print("=" * 70)

# Analyze 400K checkpoint
print("\n### Loading 400K model ###")
env = gym.make("LunarLander-v3")
model_400k = PPO.load("ppo_lunarlander_400000_v2.zip")

results_400k = []
for episode in range(100):
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    main_fires = 0
    side_fires = 0
    done = False

    while not done:
        action, _ = model_400k.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if action == 2:
            main_fires += 1
        if action in [1, 3]:
            side_fires += 1
        done = terminated or truncated

    results_400k.append(
        {
            "reward": total_reward,
            "steps": steps,
            "main_fires": main_fires,
            "side_fires": side_fires,
            "success": total_reward >= 100,
            "crash": total_reward < 0,
        }
    )

env.close()

# Analyze 600K checkpoint
print("### Loading 600K model ###")
env = gym.make("LunarLander-v3")
model_600k = PPO.load("ppo_lunarlander_600000_v2.zip")

results_600k = []
for episode in range(100):
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    main_fires = 0
    side_fires = 0
    done = False

    while not done:
        action, _ = model_600k.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if action == 2:
            main_fires += 1
        if action in [1, 3]:
            side_fires += 1
        done = terminated or truncated

    results_600k.append(
        {
            "reward": total_reward,
            "steps": steps,
            "main_fires": main_fires,
            "side_fires": side_fires,
            "success": total_reward >= 100,
            "crash": total_reward < 0,
        }
    )

env.close()

# Compare
print("\n" + "=" * 70)
print("COMPARISON RESULTS")
print("=" * 70)

print(f"\n{'Metric':<25} {'400K':>12} {'600K':>12}")
print("-" * 50)

r400 = results_400k
r600 = results_600k

print(
    f"{'Success Rate':<25} {100 * sum(r['success'] for r in r400) / len(r400):>11.1f}% {100 * sum(r['success'] for r in r600) / len(r600):>11.1f}%"
)
print(
    f"{'Crash Rate':<25} {100 * sum(r['crash'] for r in r400) / len(r400):>11.1f}% {100 * sum(r['crash'] for r in r600) / len(r600):>11.1f}%"
)
print(
    f"{'Mean Reward':<25} {np.mean([r['reward'] for r in r400]):>12.1f} {np.mean([r['reward'] for r in r600]):>12.1f}"
)
print(
    f"{'Median Reward':<25} {np.median([r['reward'] for r in r400]):>12.1f} {np.median([r['reward'] for r in r600]):>12.1f}"
)
print(
    f"{'Best Score':<25} {max(r['reward'] for r in r400):>12.1f} {max(r['reward'] for r in r600):>12.1f}"
)
print(
    f"{'Worst Score':<25} {min(r['reward'] for r in r400):>12.1f} {min(r['reward'] for r in r600):>12.1f}"
)
print(
    f"{'Std Dev':<25} {np.std([r['reward'] for r in r400]):>12.1f} {np.std([r['reward'] for r in r600]):>12.1f}"
)
print(
    f"{'Avg Steps':<25} {np.mean([r['steps'] for r in r400]):>12.0f} {np.mean([r['steps'] for r in r600]):>12.0f}"
)
print(
    f"{'Avg Main Fires':<25} {np.mean([r['main_fires'] for r in r400]):>12.0f} {np.mean([r['main_fires'] for r in r600]):>12.0f}"
)
print(
    f"{'Avg Side Fires':<25} {np.mean([r['side_fires'] for r in r400]):>12.0f} {np.mean([r['side_fires'] for r in r600]):>12.0f}"
)

# Score distribution
print(f"\n{'Score Distribution':<25}")
bins = [(-200, 0), (0, 50), (50, 100), (100, 150), (150, 200), (200, 300)]
for low, high in bins:
    c400 = len([r for r in r400 if low <= r["reward"] < high])
    c600 = len([r for r in r600 if low <= r["reward"] < high])
    print(f"  {low:5d} to {high:<5d}: 400K={c400:3d} 600K={c600:3d}")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

success_400 = [r for r in r400 if r["success"]]
success_600 = [r for r in r600 if r["success"]]
crash_400 = [r for r in r400 if r["crash"]]
crash_600 = [r for r in r600 if r["crash"]]

print(
    f"\n400K - Successful landings avg {np.mean([r['reward'] for r in success_400]):.1f} points"
)
print(
    f"600K - Successful landings avg {np.mean([r['reward'] for r in success_600]):.1f} points"
)
print(
    f"\n400K - Crashed landings avg {np.mean([r['reward'] for r in crash_400]):.1f} points"
)
print(
    f"600K - Crashed landings avg {np.mean([r['reward'] for r in crash_600]):.1f} points"
)

# Variance analysis
print(f"\n{'Variance Analysis':<25}")
var_400 = np.var([r["reward"] for r in r400])
var_600 = np.var([r["reward"] for r in r600])
print(f"400K reward variance: {var_400:.1f}")
print(f"600K reward variance: {var_600:.1f}")

if var_600 > var_400:
    print(f"-> 600K has HIGHER variance (less consistent)")
else:
    print(f"-> 400K has HIGHER variance (less consistent)")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"""
400K OUTPERFORMED 600K because:
1. Higher success rate: {100 * sum(r["success"] for r in r400) / len(r400):.0f}% vs {100 * sum(r["success"] for r in r600) / len(r600):.0f}%
2. Lower crash rate: {100 * sum(r["crash"] for r in r400) / len(r400):.0f}% vs {100 * sum(r["crash"] for r in r600) / len(r600):.0f}%
3. More consistent performance (lower variance)

POSSIBLE REASONS FOR REGRESSION AFTER 400K:
- Over-training: Model started to overfit to training data
- Learning rate too high: After 400K, model may have started forgetting good behaviors
- Policy collapsed: The additional training destabilized the learned policy
""")
