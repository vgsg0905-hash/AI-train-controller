#evaluate.py
import numpy as np
import matplotlib.pyplot as plt
from train_env import MultiTrainCorridorEnv
from stable_baselines3 import PPO

def baseline_controller(env):
    n = env.n_trains
    obs = env._get_obs()
    actions = np.zeros(n, dtype=np.float32)
    target_speed = 0.6 * env.max_speed
    for i in range(n):
        vel = obs[n + i]
        headway = (env.positions[i+1] - env.positions[i]) if i < n-1 else env.L
        dv = target_speed - vel
        a = np.clip(0.2 * dv, -env.max_accel, env.max_accel)
        if headway < env.min_gap * 1.2:
            a = min(a, -1.0)
        actions[i] = a
    return actions

def run_episode(env, policy=None, render=False):
    obs, _ = env.reset()
    terminated = False
    traj = {"pos": [], "vel": [], "tracks": [], "passed": [], "alerts": []}
    total_reward = 0.0

    while not terminated:
        if policy is None:
            action = baseline_controller(env)
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)
        total_reward += reward
        traj["pos"].append(env.positions.copy())
        traj["vel"].append(env.velocities.copy())
        traj["tracks"].append(env.tracks.copy())
        traj["passed"].append(env.passed_count)
        traj["alerts"].append(env.alerts.copy())
        if render:
            env.render()
    
    for key in traj:
        traj[key] = np.array(traj[key], dtype=object)
    return total_reward, traj

def plot_traj(traj, title=""):
    pos = traj["pos"]
    T, n = len(pos), len(pos[0])
    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.plot([pos[t][i] for t in range(T)], label=f"Train {i}")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env = MultiTrainCorridorEnv(n_trains=10, n_tracks=5, n_stations=4, corridor_length=10000.0, episode_time=400)

    # Baseline
    r_base, traj_base = run_episode(env, policy=None, render=False)
    print("Baseline total reward:", r_base, "passed:", traj_base["passed"][-1])
    plot_traj(traj_base, "Baseline positions")

    # RL Agent
    try:
        model = PPO.load("./models/ppo_multitrain")
        r_agent, traj_agent = run_episode(env, policy=model, render=False)
        print("RL agent total reward:", r_agent, "passed:", traj_agent["passed"][-1])
        plot_traj(traj_agent, "RL Policy positions")
    except Exception as e:
        print("Could not load RL model:", e)
