# train_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MultiTrainCorridorEnv(gym.Env):
    """Custom Environment for Multi-Train Traffic Control."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, n_trains=5, n_tracks=2, n_stations=3, episode_time=400, max_trains=10):
        super().__init__()

        # Parameters
        self.n_trains = n_trains
        self.n_tracks = n_tracks
        self.n_stations = n_stations
        self.episode_time = episode_time
        self.max_trains = max_trains  # max trains for fixed obs size

        # Train physics parameters
        self.L = 100_000        # track length (meters)
        self.max_speed = 40.0   # m/s (~144 km/h)
        self.max_accel = 2.0    # m/s^2
        self.min_gap = 500.0    # minimum safe distance

        # State variables
        self.positions = np.zeros(self.n_trains)
        self.velocities = np.zeros(self.n_trains)
        self.tracks = np.zeros(self.n_trains, dtype=int)

        # Step counter
        self.t = 0

        # Action space: acceleration for each train (fixed size)
        self.action_space = spaces.Box(
            low=-self.max_accel, high=self.max_accel, shape=(self.max_trains,), dtype=np.float32
        )

        # Observation space: fixed size for max_trains
        high = np.array(
            [self.L] * self.max_trains +
            [self.max_speed] * self.max_trains +
            [self.n_tracks] * self.max_trains,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=high,
            shape=(3 * self.max_trains,),
            dtype=np.float32
        )

    def _get_obs(self):
        """Return fixed-size observation padded or truncated to max_trains."""
        pos = np.zeros(self.max_trains, dtype=np.float32)
        vel = np.zeros(self.max_trains, dtype=np.float32)
        trk = np.zeros(self.max_trains, dtype=np.float32)

        n = min(self.n_trains, self.max_trains)
        pos[:n] = self.positions[:n]
        vel[:n] = self.velocities[:n]
        trk[:n] = self.tracks[:n].astype(np.float32)

        return np.concatenate([pos, vel, trk])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        spacing = self.L / self.n_trains
        self.positions = np.arange(0, self.n_trains) * spacing
        self.velocities = np.zeros(self.n_trains)
        self.tracks = np.random.randint(0, self.n_tracks, size=self.n_trains)
        self.t = 0

        obs = self._get_obs()
        info = {"passed_count": 0, "violations": 0, "fuel_proxy": 0.0}
        return obs, info

    def step(self, action):
        # Clip actions
        action = np.clip(action, -self.max_accel, self.max_accel)

        # Pad or truncate action to match n_trains
        if len(action) < self.n_trains:
            padded_action = np.zeros(self.n_trains, dtype=action.dtype)
            padded_action[:len(action)] = action
            action = padded_action
        else:
            action = action[:self.n_trains]

        # Update velocities
        self.velocities = np.clip(self.velocities + action, 0, self.max_speed)

        # Update positions
        self.positions = (self.positions + self.velocities) % self.L

        # KPIs
        passed_count = np.sum(self.positions > self.L * 0.9)

        # Check spacing violations per track
        violations = 0
        for track_id in range(self.n_tracks):
            positions_on_track = self.positions[self.tracks == track_id]
            sorted_pos = np.sort(positions_on_track)
            if len(sorted_pos) > 1:
                violations += np.sum(np.diff(sorted_pos) < self.min_gap)

        fuel_proxy = np.sum(np.abs(action))

        self.t += 1
        terminated = self.t >= self.episode_time
        truncated = False

        reward = passed_count - 10 * violations - 0.1 * fuel_proxy

        obs = self._get_obs()
        info = {"passed_count": passed_count, "violations": violations, "fuel_proxy": fuel_proxy}

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"t={self.t}, positions={self.positions}, velocities={self.velocities}, tracks={self.tracks}")

    def close(self):
        pass
