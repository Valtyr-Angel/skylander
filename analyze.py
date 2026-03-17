import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

env = gym.make("LunarLander-v3")
model = PPO.load("ppo_lunarlander")

num_episodes = 100
results = []

print(f"Running {num_episodes} episodes...")

for episode in range(num_episodes):
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    main_engine_fires = 0
    side_engine_fires = 0

    initial_obs = obs.copy()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
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

    if (episode + 1) % 20 == 0:
        print(f"Completed {episode + 1}/{num_episodes} episodes...")

print("\n" + "=" * 70)
print("DETAILED ANALYSIS")
print("=" * 70)

rewards = [r["reward"] for r in results]
outcomes = [r["outcome"] for r in results]
successes = [r for r in results if r["outcome"] == "SUCCESS"]
partials = [r for r in results if r["outcome"] == "PARTIAL"]
crashes = [r for r in results if r["outcome"] == "CRASH"]

print(f"\n--- OVERALL STATS ---")
print(f"Episodes: {num_episodes}")
print(f"Mean Reward: {np.mean(rewards):.2f}")
print(f"Median Reward: {np.median(rewards):.2f}")
print(f"Std Dev: {np.std(rewards):.2f}")
print(f"Min: {min(rewards):.2f} | Max: {max(rewards):.2f}")

print(f"\n--- SUCCESS RATE ---")
print(
    f"SUCCESS:  {len(successes)}/{num_episodes} ({100 * len(successes) / num_episodes:.1f}%)"
)
print(
    f"PARTIAL:  {len(partials)}/{num_episodes} ({100 * len(partials) / num_episodes:.1f}%)"
)
print(
    f"CRASH:    {len(crashes)}/{num_episodes} ({100 * len(crashes) / num_episodes:.1f}%)"
)

print(f"\n--- AVERAGES BY OUTCOME ---")
if successes:
    print(
        f"SUCCESS:  Steps={np.mean([r['steps'] for r in successes]):.0f}, "
        f"Main={np.mean([r['main_engine_fires'] for r in successes]):.0f}, "
        f"Side={np.mean([r['side_engine_fires'] for r in successes]):.0f}"
    )
if partials:
    print(
        f"PARTIAL:  Steps={np.mean([r['steps'] for r in partials]):.0f}, "
        f"Main={np.mean([r['main_engine_fires'] for r in partials]):.0f}, "
        f"Side={np.mean([r['side_engine_fires'] for r in partials]):.0f}"
    )
if crashes:
    print(
        f"CRASH:    Steps={np.mean([r['steps'] for r in crashes]):.0f}, "
        f"Main={np.mean([r['main_engine_fires'] for r in crashes]):.0f}, "
        f"Side={np.mean([r['side_engine_fires'] for r in crashes]):.0f}"
    )

print(f"\n--- INITIAL CONDITIONS ANALYSIS ---")


def analyze_by_condition(results, field, name, bins):
    print(f"\n{name}:")
    for low, high in bins:
        subset = [r for r in results if low <= r[field] < high]
        if subset:
            success_count = len([r for r in subset if r["outcome"] == "SUCCESS"])
            avg_reward = np.mean([r["reward"] for r in subset])
            print(
                f"  {low:5.2f} to {high:5.2f}: {len(subset):3d} episodes, "
                f"Success={100 * success_count / len(subset):5.1f}%, Avg Reward={avg_reward:7.2f}"
            )


bins_vx = [(-1.0, -0.5), (-0.5, -0.2), (-0.2, 0.2), (0.2, 0.5), (0.5, 1.0)]
bins_angle = [(-0.6, -0.3), (-0.3, -0.1), (-0.1, 0.1), (0.1, 0.3), (0.3, 0.6)]
bins_y = [(1.3, 1.35), (1.35, 1.4), (1.4, 1.45), (1.45, 1.5)]

analyze_by_condition(results, "initial_vx", "Initial X Velocity (Vx)", bins_vx)
analyze_by_condition(results, "initial_angle", "Initial Angle", bins_angle)
analyze_by_condition(results, "initial_y", "Initial Altitude (Y)", bins_y)

print(f"\n--- INITIAL CONDITIONS COMPARISON ---")
if successes and crashes:
    print(
        f"Initial Vx - Success: {np.mean([r['initial_vx'] for r in successes]):.3f}, "
        f"Crash: {np.mean([r['initial_vx'] for r in crashes]):.3f}"
    )
    print(
        f"Initial Vy - Success: {np.mean([r['initial_vy'] for r in successes]):.3f}, "
        f"Crash: {np.mean([r['initial_vy'] for r in crashes]):.3f}"
    )
    print(
        f"Initial Angle - Success: {np.mean([r['initial_angle'] for r in successes]):.3f}, "
        f"Crash: {np.mean([r['initial_angle'] for r in crashes]):.3f}"
    )
    print(
        f"Initial Ang Vel - Success: {np.mean([r['initial_ang_vel'] for r in successes]):.3f}, "
        f"Crash: {np.mean([r['initial_ang_vel'] for r in crashes]):.3f}"
    )

print(f"\n--- REWARD DISTRIBUTION ---")
bins_reward = [
    (-200, -100),
    (-100, -50),
    (-50, 0),
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
    (200, 300),
]
for low, high in bins_reward:
    count = len([r for r in results if low <= r["reward"] < high])
    bar = "#" * int(count / 2)
    print(f"{low:5d} to {high:5d}: {count:3d} {bar}")

print("\n" + "=" * 70)
print("IMPROVEMENT RECOMMENDATIONS")
print("=" * 70)

crash_rate = 100 * len(crashes) / num_episodes
success_rate = 100 * len(successes) / num_episodes

if crash_rate > 20:
    print(f"\n- High crash rate ({crash_rate:.1f}%) needs improvement")
    avg_vx_crash = np.mean([abs(r["initial_vx"]) for r in crashes])
    avg_vx_success = np.mean([abs(r["initial_vx"]) for r in successes])
    if avg_vx_crash > avg_vx_success * 1.2:
        print(f"  -> Model struggles with high initial horizontal velocity")
        print(f"  -> Consider: More training on high-drift scenarios")

    avg_steps_crash = np.mean([r["steps"] for r in crashes])
    if avg_steps_crash < 250:
        print(f"  -> Quick crashes suggest early guidance issues")
        print(f"  -> Consider: Lower learning rate for faster stabilization")

if success_rate > 80:
    print(f"\n- Excellent success rate ({success_rate:.1f}%)!")
    print(f"  -> Model is well-trained, minor improvements possible")

avg_successful_reward = np.mean([r["reward"] for r in successes])
if avg_successful_reward < 180:
    print(
        f"\n- Successful landings average {avg_successful_reward:.1f} (potential: 200+)"
    )
    print(f"  -> Could improve by reducing fuel usage or landing time")

env.close()
