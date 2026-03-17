import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

env = gym.make("LunarLander-v3")
model = PPO.load("ppo_lunarlander_v3.zip")

print("=" * 70)
print("DETAILED WEAKNESS ANALYSIS: V3 MODEL")
print("=" * 70)

num_episodes = 500
results = []

print(f"\nRunning {num_episodes} episodes for analysis...")

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    actions = []
    observations = []
    done = False

    initial_obs = obs.copy()

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        actions.append(action)
        observations.append(obs.copy())

        done = terminated or truncated

    if total_reward >= 100:
        outcome = "SUCCESS"
    elif total_reward >= 0:
        outcome = "PARTIAL"
    else:
        outcome = "CRASH"

    results.append(
        {
            "episode": episode + 1,
            "reward": total_reward,
            "steps": steps,
            "outcome": outcome,
            "initial_vx": initial_obs[2],
            "initial_vy": initial_obs[3],
            "initial_angle": initial_obs[4],
            "initial_ang_vel": initial_obs[5],
            "final_vx": observations[-1][2],
            "final_vy": observations[-1][3],
            "actions": actions,
        }
    )

    if (episode + 1) % 100 == 0:
        print(f"  Completed {episode + 1}/{num_episodes}...")

env.close()

rewards = [r["reward"] for r in results]
outcomes = [r["outcome"] for r in results]
successes = [r for r in results if r["outcome"] == "SUCCESS"]
partials = [r for r in results if r["outcome"] == "PARTIAL"]
crashes = [r for r in results if r["outcome"] == "CRASH"]

print("\n" + "=" * 70)
print("1. OVERALL STATISTICS")
print("=" * 70)
print(f"Total Episodes: {num_episodes}")
print(f"SUCCESS: {len(successes)} ({100 * len(successes) / num_episodes:.1f}%)")
print(f"PARTIAL: {len(partials)} ({100 * len(partials) / num_episodes:.1f}%)")
print(f"CRASH: {len(crashes)} ({100 * len(crashes) / num_episodes:.1f}%)")

print("\n" + "=" * 70)
print("2. INITIAL VELOCITY ANALYSIS")
print("=" * 70)

print("\n--- Initial Vx (Horizontal Velocity) ---")
bins_vx = [
    (-1.0, -0.6),
    (-0.6, -0.4),
    (-0.4, -0.2),
    (-0.2, 0.0),
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 1.0),
]
for low, high in bins_vx:
    subset = [r for r in results if low <= r["initial_vx"] < high]
    if subset:
        success_count = len([r for r in subset if r["outcome"] == "SUCCESS"])
        avg_reward = np.mean([r["reward"] for r in subset])
        avg_steps = np.mean([r["steps"] for r in subset])
        print(
            f"  Vx {low:5.2f} to {high:5.2f}: n={len(subset):3d}, "
            f"Success={100 * success_count / len(subset):5.1f}%, "
            f"Avg Reward={avg_reward:7.1f}, Steps={avg_steps:5.0f}"
        )

print("\n--- Initial Vy (Vertical Velocity) ---")
bins_vy = [(-0.8, -0.4), (-0.4, 0.0), (0.0, 0.4), (0.4, 0.8)]
for low, high in bins_vy:
    subset = [r for r in results if low <= r["initial_vy"] < high]
    if subset:
        success_count = len([r for r in subset if r["outcome"] == "SUCCESS"])
        avg_reward = np.mean([r["reward"] for r in subset])
        print(
            f"  Vy {low:5.2f} to {high:5.2f}: n={len(subset):3d}, "
            f"Success={100 * success_count / len(subset):5.1f}%, "
            f"Avg Reward={avg_reward:7.1f}"
        )

print("\n" + "=" * 70)
print("3. TIME-TO-FAILURE ANALYSIS")
print("=" * 70)

if crashes:
    crash_steps = [r["steps"] for r in crashes]
    print(f"\nCrashes occur at:")
    print(f"  Mean steps: {np.mean(crash_steps):.1f}")
    print(f"  Median steps: {np.median(crash_steps):.1f}")
    print(f"  Min steps: {min(crash_steps)}")
    print(f"  Max steps: {max(crash_steps)}")

    early_crashes = len([s for s in crash_steps if s < 200])
    mid_crashes = len([s for s in crash_steps if 200 <= s < 400])
    late_crashes = len([s for s in crash_steps if s >= 400])
    print(
        f"\n  Early (<200 steps): {early_crashes} ({100 * early_crashes / len(crashes):.1f}%)"
    )
    print(
        f"  Mid (200-400 steps): {mid_crashes} ({100 * mid_crashes / len(crashes):.1f}%)"
    )
    print(
        f"  Late (>=400 steps): {late_crashes} ({100 * late_crashes / len(crashes):.1f}%)"
    )

print("\n" + "=" * 70)
print("4. ACTION PATTERN ANALYSIS")
print("=" * 70)


def analyze_actions(outcome_results, name):
    if not outcome_results:
        return
    all_actions = []
    for r in outcome_results:
        all_actions.extend(r["actions"])

    action_counts = np.bincount(all_actions, minlength=4)
    total = sum(action_counts)

    action_names = {0: "None", 1: "Left", 2: "Main", 3: "Right"}
    print(f"\n{name}:")
    for i, count in enumerate(action_counts):
        pct = 100 * count / total
        bar = "#" * int(pct / 2)
        print(f"  {action_names[i]}: {count:5d} ({pct:5.1f}%) {bar}")


analyze_actions(successes, "SUCCESSFUL landings")
analyze_actions(partials, "PARTIAL landings")
analyze_actions(crashes, "CRASHED landings")

print("\n" + "=" * 70)
print("5. INITIAL CONDITIONS: SUCCESS vs CRASH")
print("=" * 70)

if successes and crashes:
    print(f"\n{'Parameter':<20} {'SUCCESS':>12} {'CRASH':>12} {'Diff':>12}")
    print("-" * 56)

    params = [
        ("Initial Vx", "initial_vx"),
        ("Initial Vy", "initial_vy"),
        ("Initial Angle", "initial_angle"),
        ("Initial Ang Vel", "initial_ang_vel"),
    ]

    for name, key in params:
        s_mean = np.mean([abs(r[key]) for r in successes])
        c_mean = np.mean([abs(r[key]) for r in crashes])
        diff = c_mean - s_mean
        diff_pct = 100 * diff / max(s_mean, 0.001)
        print(
            f"{name:<20} {s_mean:>12.3f} {c_mean:>12.3f} {diff:>+12.3f} ({diff_pct:>+6.1f}%)"
        )

print("\n" + "=" * 70)
print("6. CORRELATION ANALYSIS")
print("=" * 70)

# Calculate correlations with success
initial_features = {
    "|Vx|": [abs(r["initial_vx"]) for r in results],
    "|Vy|": [abs(r["initial_vy"]) for r in results],
    "|Angle|": [abs(r["initial_angle"]) for r in results],
    "|Ang Vel|": [abs(r["initial_ang_vel"]) for r in results],
}

binary_success = [1 if r["outcome"] == "SUCCESS" else 0 for r in results]

print("\nCorrelation with Success:")
for name, values in initial_features.items():
    correlation = np.corrcoef(values, binary_success)[0, 1]
    bar = "#" * int(abs(correlation) * 20)
    print(f"  {name:<12}: {correlation:>+6.3f} {bar}")

print("\n" + "=" * 70)
print("7. CRASH SCENARIOS")
print("=" * 70)

# Identify most common crash patterns
crash_patterns = []
for r in crashes:
    pattern = ""
    # Vx direction
    if r["initial_vx"] < -0.3:
        pattern += "LEFT_DRIFT_"
    elif r["initial_vx"] > 0.3:
        pattern += "RIGHT_DRIFT_"
    else:
        pattern += "LOW_DRIFT_"

    # Vy direction
    if r["initial_vy"] < -0.2:
        pattern += "DOWN_"
    elif r["initial_vy"] > 0.2:
        pattern += "UP_"
    else:
        pattern += "NEUTRAL_"

    # Angle
    if abs(r["initial_angle"]) > 0.3:
        pattern += "TILTED"
    else:
        pattern += "LEVEL"

    crash_patterns.append(pattern)

from collections import Counter

pattern_counts = Counter(crash_patterns)

print("\nMost Common Crash Patterns:")
for pattern, count in pattern_counts.most_common(10):
    pct = 100 * count / len(crashes)
    print(f"  {pattern:<25}: {count:3d} ({pct:5.1f}%)")

print("\n" + "=" * 70)
print("8. SUMMARY: KEY WEAKNESSES")
print("=" * 70)

# Calculate key metrics
high_vx_crashes = len([r for r in crashes if abs(r["initial_vx"]) > 0.4])
high_vx_total = len([r for r in results if abs(r["initial_vx"]) > 0.4])
early_crashes = len([r for r in crashes if r["steps"] < 250])

print(f"""
Based on this analysis, the KEY WEAKNESSES are:

1. HIGH HORIZONTAL VELOCITY (PRIMARY ISSUE)
   - {100 * high_vx_crashes / high_vx_total:.1f}% of high-drift episodes end in crash
   - Initial |Vx| > 0.4 is a strong predictor of failure
   - Correlation with success: {np.corrcoef(initial_features["|Vx|"], binary_success)[0, 1]:.3f}

2. EARLY CRASHES (SECONDARY ISSUE)
   - {100 * early_crashes / len(crashes):.1f}% of crashes happen within 250 steps
   - Model fails quickly without recovery attempts

3. ANGULAR VELOCITY
   - Initial angular velocity correlates negatively with success
   - Tilted starts lead to more failures

RECOMMENDATIONS FOR IMPROVEMENT:
- Train specifically on high-drift scenarios
- Use curriculum learning with increasing difficulty
- Add reward for stable hover before landing
""")

print("=" * 70)
