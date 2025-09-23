#animate_train
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from train_env import MultiTrainCorridorEnv
from stable_baselines3 import PPO
import os


# --- Run episode and record positions + tracks ---
def run_episode(env, policy=None):
    obs, info = env.reset()
    terminated, truncated = False, False
    positions, tracks = [], []

    while not (terminated or truncated):
        if policy is None:
            action = np.ones(env.n_trains) * 0.5 * env.max_accel
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(env.positions.copy())
        tracks.append(env.tracks.copy())

    return np.array(positions), np.array(tracks)


# --- Save GIF animation ---
def save_positions_animation(positions, tracks, env, out_path,
                             fps=6, interval_ms=150, figsize=(12, 5)):
    positions = np.asarray(positions, dtype=float)
    tracks = np.asarray(tracks, dtype=int)
    T, n_trains = positions.shape
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, float(env.L))
    ax.set_ylim(-1, env.n_tracks)
    ax.set_yticks(range(env.n_tracks))
    ax.set_yticklabels([f"Track {i}" for i in range(env.n_tracks)])
    ax.set_xlabel("Position along corridor (m)")
    ax.set_ylabel("Track")
    ax.set_title("Train Movements with Track Switching & Stations")

    # --- Draw station markers ---
    station_positions = np.linspace(0, env.L, env.n_stations + 2)[1:-1]  # equally spaced stations
    for s in station_positions:
        ax.axvline(x=s, color="red", linestyle="--", alpha=0.7)
        ax.text(s, env.n_tracks - 0.3, f"Station {int(s)}m",
                rotation=90, verticalalignment="top",
                fontsize=8, color="red")

    # --- Train scatter plot ---
    scat = ax.scatter(positions[0], tracks[0], s=200, marker="s")

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return (scat,)

    def update(frame):
        offsets = np.column_stack((positions[frame], tracks[frame]))
        scat.set_offsets(offsets)
        return (scat,)

    ani = FuncAnimation(fig, update, frames=T, init_func=init,
                        interval=interval_ms, blit=False, repeat=False)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return out_path


# --- Main execution ---
if __name__ == "__main__":
    env = MultiTrainCorridorEnv(n_trains=10, n_tracks=5, n_stations=4,
                                corridor_length=10000, episode_time=300)

    # Try RL agent
    try:
        model = PPO.load("./models/ppo_multitrain")
        positions, tracks = run_episode(env, policy=model)
        title = "RL Policy"
    except Exception:
        positions, tracks = run_episode(env, policy=None)
        title = "Baseline"

    gif_path = f"results/{title.replace(' ','_')}.gif"
    save_positions_animation(positions, tracks, env, gif_path)
    print(f"Animation saved at {gif_path}")
