#train_agent.py
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train_env import MultiTrainCorridorEnv

def make_env(n_trains, n_tracks, n_stations, corridor_length, episode_time):
    def _init():
        return MultiTrainCorridorEnv(
            n_trains=n_trains,
            n_tracks=n_tracks,
            n_stations=n_stations,
            corridor_length=corridor_length,
            episode_time=episode_time
        )
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trains", type=int, default=10)
    parser.add_argument("--n-tracks", type=int, default=5)
    parser.add_argument("--n-stations", type=int, default=4)
    parser.add_argument("--length", type=float, default=10000.0)
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--episode-time", type=int, default=400)
    parser.add_argument("--save-dir", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env = DummyVecEnv([make_env(args.n_trains, args.n_tracks, args.n_stations,
                                args.length, args.episode_time)])
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64)
    total_timesteps = args.episodes * args.episode_time
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(args.save_dir, "ppo_multitrain"))
    print("✅ RL model saved at:", os.path.join(args.save_dir, "ppo_multitrain.zip"))

if __name__ == "__main__":
    main()
