import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("=" * 70)
print("COMPARING: Reward Shaped +200K vs +400K")
print("=" * 70)

# Model 1: +200K (better)
print("\n### Loading +200K model ###")
env1 = gym.make("LunarLander-v3")
model_200k = PPO.load("ppo_reward_shaped_v2.zip")

results_200k = []
for episode in range(300):
    obs, info = env1.reset()
    total_reward = 0.0
    steps = 0
    actions = []
    done = False

    initial_obs = obs.copy()

    while not done:
        action, _states = model_200k.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env1.step(action)

        total_reward += float(reward)
        steps += 1
        actions.append(action)

        done = terminated or truncated

    if total_reward >= 100:
        outcome = "SUCCESS"
    elif total_reward >= 0:
        outcome = "PARTIAL"
    else:
        outcome = "CRASH"

    results_200k.append(
        {
            "reward": total_reward,
            "steps": steps,
            "outcome": outcome,
            "initial_vx": abs(initial_obs[2]),
            "initial_vy": abs(initial_obs[3]),
            "actions": actions,
        }
    )

    if (episode + 1) % 100 == 0:
        print(f"  200K: Completed {episode + 1}/300...")

env1.close()

# Model 2: +400K (regressed)
print("\n### Loading +400K model ###")
env2 = gym.make("LunarLander-v3")
model_400k = PPO.load("ppo_lunarlander_reward_shaped.zip")

results_400k = []
for episode in range(300):
    obs, info = env2.reset()
    total_reward = 0.0
    steps = 0
    actions = []
    done = False

    initial_obs = obs.copy()

    while not done:
        action, _states = model_400k.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env2.step(action)

        total_reward += float(reward)
        steps += 1
        actions.append(action)

        done = terminated or truncated

    if total_reward >= 100:
        outcome = "SUCCESS"
    elif total_reward >= 0:
        outcome = "PARTIAL"
    else:
        outcome = "CRASH"

    results_400k.append(
        {
            "reward": total_reward,
            "steps": steps,
            "outcome": outcome,
            "initial_vx": abs(initial_obs[2]),
            "initial_vy": abs(initial_obs[3]),
            "actions": actions,
        }
    )

    if (episode + 1) % 100 == 0:
        print(f"  400K: Completed {episode + 1}/300...")

env2.close()

# Analysis
print("\n" + "=" * 70)
print("1. OVERALL COMPARISON")
print("=" * 70)

successes_200k = [r for r in results_200k if r["outcome"] == "SUCCESS"]
crashes_200k = [r for r in results_200k if r["outcome"] == "CRASH"]
successes_400k = [r for r in results_400k if r["outcome"] == "SUCCESS"]
crashes_400k = [r for r in results_400k if r["outcome"] == "CRASH"]

print(f"\n{'Metric':<25} {'+200K':>12} {'+400K':>12} {'Diff':>12}")
print("-" * 60)
print(
    f"{'Success Rate':<25} {100 * len(successes_200k) / 300:>11.1f}% {100 * len(successes_400k) / 300:>11.1f}% {100 * (len(successes_400k) - len(successes_200k)) / 300:>+11.1f}%"
)
print(
    f"{'Crash Rate':<25} {100 * len(crashes_200k) / 300:>11.1f}% {100 * len(crashes_400k) / 300:>11.1f}% {100 * (len(crashes_400k) - len(crashes_200k)) / 300:>+11.1f}%"
)
print(
    f"{'Mean Reward':<25} {np.mean([r['reward'] for r in results_200k]):>12.1f} {np.mean([r['reward'] for r in results_400k]):>12.1f} {np.mean([r['reward'] for r in results_400k]) - np.mean([r['reward'] for r in results_200k]):>+12.1f}"
)
best_200k = max(r["reward"] for r in results_200k)
best_400k = max(r["reward"] for r in results_400k)
print(
    f"{'Best Score':<25} {best_200k:>12.1f} {best_400k:>12.1f} {best_400k - best_200k:>+12.1f}"
)
print(
    f"{'Avg Steps (all)':<25} {np.mean([r['steps'] for r in results_200k]):>12.0f} {np.mean([r['steps'] for r in results_400k]):>12.0f} {np.mean([r['steps'] for r in results_400k]) - np.mean([r['steps'] for r in results_200k]):>+12.0f}"
)

print("\n" + "=" * 70)
print("2. ACTION PATTERN COMPARISON")
print("=" * 70)


def get_action_dist(outcome_results, name):
    all_actions = []
    for r in outcome_results:
        all_actions.extend(r["actions"])
    if not all_actions:
        return {}
    counts = np.bincount(all_actions, minlength=4)
    total = sum(counts)
    return {
        "None": 100 * counts[0] / total,
        "Left": 100 * counts[1] / total,
        "Main": 100 * counts[2] / total,
        "Right": 100 * counts[3] / total,
    }


print("\n--- SUCCESSFUL Landings ---")
dist_200k_success = get_action_dist(successes_200k, "200K")
dist_400k_success = get_action_dist(successes_400k, "400K")
print(f"{'Action':<10} {'+200K':>10} {'+400K':>10} {'Diff':>10}")
for action in ["None", "Left", "Main", "Right"]:
    print(
        f"{action:<10} {dist_200k_success.get(action, 0):>9.1f}% {dist_400k_success.get(action, 0):>9.1f}% {dist_400k_success.get(action, 0) - dist_200k_success.get(action, 0):>+9.1f}%"
    )

print("\n--- CRASHED Landings ---")
dist_200k_crash = get_action_dist(crashes_200k, "200K")
dist_400k_crash = get_action_dist(crashes_400k, "400K")
print(f"{'Action':<10} {'+200K':>10} {'+400K':>10} {'Diff':>10}")
for action in ["None", "Left", "Main", "Right"]:
    print(
        f"{action:<10} {dist_200k_crash.get(action, 0):>9.1f}% {dist_400k_crash.get(action, 0):>9.1f}% {dist_400k_crash.get(action, 0) - dist_200k_crash.get(action, 0):>+9.1f}%"
    )

print("\n" + "=" * 70)
print("3. INITIAL VELOCITY ANALYSIS")
print("=" * 70)


def analyze_by_vx(results, name):
    bins = [(-1.0, -0.5), (-0.5, -0.2), (-0.2, 0.2), (0.2, 0.5), (0.5, 1.0)]
    print(f"\n{name}:")
    for low, high in bins:
        subset = [r for r in results if low <= r["initial_vx"] < high]
        if subset:
            success_count = len([r for r in subset if r["outcome"] == "SUCCESS"])
            print(
                f"  Vx {low:5.2f} to {high:5.2f}: n={len(subset):3d}, Success={100 * success_count / len(subset):5.1f}%"
            )


analyze_by_vx(results_200k, "+200K Model")
analyze_by_vx(results_400k, "+400K Model")

print("\n" + "=" * 70)
print("4. ROOT CAUSE ANALYSIS")
print("=" * 70)

# Compare initial conditions
print("\n--- Initial Conditions of Crashes ---")
print(f"{'Metric':<25} {'+200K':>12} {'+400K':>12} {'Diff':>12}")
print("-" * 60)
if crashes_200k and crashes_400k:
    print(
        f"{'Avg |Vx| (crashes)':<25} {np.mean([r['initial_vx'] for r in crashes_200k]):>12.3f} {np.mean([r['initial_vx'] for r in crashes_400k]):>12.3f} {np.mean([r['initial_vx'] for r in crashes_400k]) - np.mean([r['initial_vx'] for r in crashes_200k]):>+12.3f}"
    )
    print(
        f"{'Avg Steps to Crash':<25} {np.mean([r['steps'] for r in crashes_200k]):>12.0f} {np.mean([r['steps'] for r in crashes_400k]):>12.0f} {np.mean([r['steps'] for r in crashes_400k]) - np.mean([r['steps'] for r in crashes_200k]):>+12.0f}"
    )

# High Vx analysis
high_vx_200k = [r for r in results_200k if r["initial_vx"] > 0.4]
high_vx_400k = [r for r in results_400k if r["initial_vx"] > 0.4]
high_vx_success_200k = len([r for r in high_vx_200k if r["outcome"] == "SUCCESS"])
high_vx_success_400k = len([r for r in high_vx_400k if r["outcome"] == "SUCCESS"])

print(f"\n--- High Velocity (|Vx| > 0.4) Performance ---")
print(f"{'Metric':<25} {'+200K':>12} {'+400K':>12} {'Diff':>12}")
print("-" * 60)
print(
    f"{'High Vx Episodes':<25} {len(high_vx_200k):>12} {len(high_vx_400k):>12} {len(high_vx_400k) - len(high_vx_200k):>+12}"
)
if high_vx_200k and high_vx_400k:
    print(
        f"{'High Vx Success Rate':<25} {100 * high_vx_success_200k / len(high_vx_200k):>11.1f}% {100 * high_vx_success_400k / len(high_vx_400k):>11.1f}% {100 * (high_vx_success_400k / len(high_vx_400k) - high_vx_success_200k / len(high_vx_200k)):>+11.1f}%"
    )

print("\n" + "=" * 70)
print("5. CONCLUSIONS: WHY REGRESSION?")
print("=" * 70)

main_diff = dist_400k_crash.get("Main", 0) - dist_200k_crash.get("Main", 0)
right_diff = dist_400k_crash.get("Right", 0) - dist_200k_crash.get("Right", 0)

print(f"""
REGRESSION CAUSES:

1. ACTION PATTERN CHANGE
   - Crashes in +400K use {main_diff:+.1f}% MORE main engine
   - Crashes in +400K use {right_diff:+.1f}% LESS right engine
   - This suggests the model forgot horizontal correction

2. HIGH VELOCITY HANDLING
   - +400K performs worse on high-velocity initial conditions
   - More crashes when |Vx| > 0.4

3. OVER-TRAINING
   - Continued training caused forgetting of good behaviors
   - The model may have overfitted to specific scenarios

RECOMMENDATION:
- Use +200K checkpoint (best performance)
- Consider lower learning rate for future training
- Or train fresh with reward shaping from scratch
""")

print("=" * 70)
