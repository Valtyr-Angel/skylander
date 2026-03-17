import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

env = gym.make("LunarLander-v3")
model = PPO.load("ppo_lunarlander_900000.zip")

num_episodes = 250
results = []

print(f"Running {num_episodes} episodes for in-depth analysis...")

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    main_engine_fires = 0
    side_engine_fires = 0

    initial_obs = obs.copy()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        if action == 2:
            main_engine_fires += 1
        if action in [1, 3]:
            side_engine_fires += 1

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
            "main_engine_fires": main_engine_fires,
            "side_engine_fires": side_engine_fires,
            "outcome": outcome,
            "initial_x": initial_obs[0],
            "initial_y": initial_obs[1],
            "initial_vx": initial_obs[2],
            "initial_vy": initial_obs[3],
            "initial_angle": initial_obs[4],
            "initial_ang_vel": initial_obs[5],
        }
    )

    if (episode + 1) % 50 == 0:
        print(f"Completed {episode + 1}/{num_episodes} episodes...")

env.close()

print("\n" + "=" * 70)
print("IN-DEPTH ANALYSIS: 900K MODEL")
print("=" * 70)

rewards = [r["reward"] for r in results]
outcomes = [r["outcome"] for r in results]
successes = [r for r in results if r["outcome"] == "SUCCESS"]
partials = [r for r in results if r["outcome"] == "PARTIAL"]
crashes = [r for r in results if r["outcome"] == "CRASH"]

print(f"\n{'=' * 70}")
print("1. OVERALL STATISTICS")
print("=" * 70)
print(f"Episodes: {num_episodes}")
print(f"Mean Reward: {np.mean(rewards):.2f}")
print(f"Median Reward: {np.median(rewards):.2f}")
print(f"Std Dev: {np.std(rewards):.2f}")
print(f"Min: {min(rewards):.2f} | Max: {max(rewards):.2f}")
print(f"Range: {max(rewards) - min(rewards):.2f}")

print(f"\n{'=' * 70}")
print("2. SUCCESS RATE BREAKDOWN")
print("=" * 70)
print(
    f"SUCCESS (>= 100):  {len(successes)}/{num_episodes} ({100 * len(successes) / num_episodes:.1f}%)"
)
print(
    f"PARTIAL (0-99):   {len(partials)}/{num_episodes} ({100 * len(partials) / num_episodes:.1f}%)"
)
print(
    f"CRASH (< 0):      {len(crashes)}/{num_episodes} ({100 * len(crashes) / num_episodes:.1f}%)"
)

print(f"\n{'=' * 70}")
print("3. AVERAGES BY OUTCOME")
print("=" * 70)
print(
    f"{'Outcome':<12} {'Steps':>8} {'Main Eng':>10} {'Side Eng':>10} {'Total Fuel':>12}"
)
print("-" * 52)
if successes:
    print(
        f"{'SUCCESS':<12} {np.mean([r['steps'] for r in successes]):>8.0f} {np.mean([r['main_engine_fires'] for r in successes]):>10.0f} {np.mean([r['side_engine_fires'] for r in successes]):>10.0f} {np.mean([r['main_engine_fires'] + r['side_engine_fires'] for r in successes]):>12.0f}"
    )
if partials:
    print(
        f"{'PARTIAL':<12} {np.mean([r['steps'] for r in partials]):>8.0f} {np.mean([r['main_engine_fires'] for r in partials]):>10.0f} {np.mean([r['side_engine_fires'] for r in partials]):>10.0f} {np.mean([r['main_engine_fires'] + r['side_engine_fires'] for r in partials]):>12.0f}"
    )
if crashes:
    print(
        f"{'CRASH':<12} {np.mean([r['steps'] for r in crashes]):>8.0f} {np.mean([r['main_engine_fires'] for r in crashes]):>10.0f} {np.mean([r['side_engine_fires'] for r in crashes]):>10.0f} {np.mean([r['main_engine_fires'] + r['side_engine_fires'] for r in crashes]):>12.0f}"
    )

print(f"\n{'=' * 70}")
print("4. REWARD DISTRIBUTION")
print("=" * 70)
bins = [
    (-150, -100),
    (-100, -50),
    (-50, 0),
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
    (200, 250),
    (250, 300),
]
for low, high in bins:
    count = len([r for r in results if low <= r["reward"] < high])
    pct = 100 * count / num_episodes
    bar = "#" * int(pct / 2)
    print(f"{low:5d} to {high:5d}: {count:3d} ({pct:5.1f}%) {bar}")

print(f"\n{'=' * 70}")
print("5. INITIAL CONDITIONS ANALYSIS")
print("=" * 70)


def analyze_condition(field, name, bins):
    print(f"\n{name}:")
    for low, high in bins:
        subset = [r for r in results if low <= r[field] < high]
        if subset:
            success_count = len([r for r in subset if r["outcome"] == "SUCCESS"])
            avg_reward = np.mean([r["reward"] for r in subset])
            pct = 100 * success_count / len(subset)
            bar = "#" * int(pct / 5)
            print(
                f"  {low:6.2f} to {high:6.2f}: {len(subset):3d} ep, {pct:5.1f}% success, avg reward: {avg_reward:7.1f} {bar}"
            )


analyze_condition(
    "initial_vx",
    "Initial X Velocity (Vx)",
    [(-1.0, -0.5), (-0.5, -0.2), (-0.2, 0.2), (0.2, 0.5), (0.5, 1.0)],
)
analyze_condition(
    "initial_vy",
    "Initial Y Velocity (Vy)",
    [(-0.6, -0.3), (-0.3, 0.0), (0.0, 0.3), (0.3, 0.6)],
)
analyze_condition(
    "initial_y",
    "Initial Altitude (Y)",
    [(1.30, 1.35), (1.35, 1.40), (1.40, 1.45), (1.45, 1.50)],
)

print(f"\n{'=' * 70}")
print("6. INITIAL CONDITIONS COMPARISON: SUCCESS vs CRASH")
print("=" * 70)
if successes and crashes:
    print(f"{'Parameter':<20} {'SUCCESS':>12} {'CRASH':>12} {'Difference':>12}")
    print("-" * 56)
    print(
        f"{'Initial Vx':<20} {np.mean([r['initial_vx'] for r in successes]):>12.3f} {np.mean([r['initial_vx'] for r in crashes]):>12.3f} {np.mean([r['initial_vx'] for r in crashes]) - np.mean([r['initial_vx'] for r in successes]):>12.3f}"
    )
    print(
        f"{'Initial Vy':<20} {np.mean([r['initial_vy'] for r in successes]):>12.3f} {np.mean([r['initial_vy'] for r in crashes]):>12.3f} {np.mean([r['initial_vy'] for r in crashes]) - np.mean([r['initial_vy'] for r in successes]):>12.3f}"
    )
    print(
        f"{'Initial Angle':<20} {np.mean([r['initial_angle'] for r in successes]):>12.3f} {np.mean([r['initial_angle'] for r in crashes]):>12.3f} {np.mean([r['initial_angle'] for r in crashes]) - np.mean([r['initial_angle'] for r in successes]):>12.3f}"
    )
    print(
        f"{'Initial Ang Vel':<20} {np.mean([r['initial_ang_vel'] for r in successes]):>12.3f} {np.mean([r['initial_ang_vel'] for r in crashes]):>12.3f} {np.mean([r['initial_ang_vel'] for r in crashes]) - np.mean([r['initial_ang_vel'] for r in successes]):>12.3f}"
    )

print(f"\n{'=' * 70}")
print("7. SCORE RANGES")
print("=" * 70)
excellent = len([r for r in results if r["reward"] >= 200])
good = len([r for r in results if 150 <= r["reward"] < 200])
okay = len([r for r in results if 100 <= r["reward"] < 150])
poor = len([r for r in results if 0 <= r["reward"] < 100])
bad = len([r for r in results if r["reward"] < 0])

print(f"Excellent (200+): {excellent:3d} ({100 * excellent / num_episodes:.1f}%)")
print(f"Good (150-199):   {good:3d} ({100 * good / num_episodes:.1f}%)")
print(f"Okay (100-149):   {okay:3d} ({100 * okay / num_episodes:.1f}%)")
print(f"Poor (0-99):      {poor:3d} ({100 * poor / num_episodes:.1f}%)")
print(f"Bad (<0):         {bad:3d} ({100 * bad / num_episodes:.1f}%)")

print(f"\n{'=' * 70}")
print("8. KEY INSIGHTS")
print("=" * 70)
avg_vx_success = np.mean([abs(r["initial_vx"]) for r in successes])
avg_vx_crash = np.mean([abs(r["initial_vx"]) for r in crashes])
print(
    f"- Model achieves {len(successes) / num_episodes * 100:.0f}% success rate overall"
)
print(f"- Crash rate: {len(crashes) / num_episodes * 100:.0f}%")
print(f"- Average |Vx| for success: {avg_vx_success:.3f}")
print(f"- Average |Vx| for crash: {avg_vx_crash:.3f}")
if avg_vx_crash > avg_vx_success * 1.2:
    print(f"- High initial horizontal velocity is a major crash factor")
print(
    f"- Median reward for successful landings: {np.median([r['reward'] for r in successes]):.1f}"
)
print(
    f"- Most landings ({(excellent + good + okay) / num_episodes * 100:.0f}%) are above partial success"
)

print("\n" + "=" * 70)
