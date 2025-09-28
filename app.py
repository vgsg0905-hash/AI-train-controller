# app.py
import streamlit as st
import numpy as np
import pandas as pd
import random
from train_env import MultiTrainCorridorEnv
from stable_baselines3 import PPO
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ===============================
# 🚆 Streamlit Setup
# ===============================
st.set_page_config(layout="wide", page_title="MaxRail AI Dashboard")
st.title("MaxRail AI Dashboard")

# ===============================
# 🎨 Custom IRCTC Control Room Styling
# ===============================

st.markdown("""
    Welcome to **MaxRail AI**!  
    - Analyze your rail simulations quickly.  
    - Optimize routes and efficiency.  
    - Interactive and easy-to-use interface.
    - Harness AI-driven insights to reduce delays and improve scheduling.  
    - Visualize rail network performance in real time for smarter decisions.

    """)

st.markdown("""
   <style>
    body {background-color: #0f172a; color: #f8fafc; font-family: "Segoe UI", sans-serif;}
    h1, h2, h3 {color: #38bdf8; font-weight: bold;}
    .css-1d391kg {background-color: #1e293b;} /* Sidebar */
    .css-1d391kg h2, .css-1d391kg label {color: white;}


    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        padding: 18px;
        border-radius: 15px;
        text-align: center;
        margin: 10px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.6);
    }
    .kpi-value {font-size: 32px; font-weight: bold;}
    .kpi-label {font-size: 14px; color: #e2e8f0;}

    /* Alerts blinking */
    @keyframes blink {
        0% {opacity: 1;}
        50% {opacity: 0.4;}
        100% {opacity: 1;}
    }
    .alert-signal {
        animation: blink 1s infinite;
        font-weight: bold;
        padding: 8px;
        margin: 5px 0;
        border-radius: 8px;
    }
    .safe {background: #064e3b; color: #22c55e;}
    .warn {background: #78350f; color: #fbbf24;}
    .danger {background: #7f1d1d; color: #ef4444;}
    </style>
""", unsafe_allow_html=True)



#----Sidebar----

with st.sidebar:
    st.sidebar.header("Simulation Settings")
    n_trains = st.sidebar.slider("Number of Trains", 2, 20, 10)
    n_tracks = st.sidebar.slider("Number of Tracks", 1, 10, 5)
    n_stations = st.sidebar.slider("Number of Stations", 1, 10, 4)
    episode_time = st.sidebar.slider("Simulation Steps", 100, 1000, 400)
    steps = st.slider("Simulation Episodes", 10, 100, 50)
    
    st.header("Control Panel")
    controller_mode = st.radio("Select Controller Mode", ["Baseline", "AI Model", "Compare AI vs Baseline"])
    run_button = st.button("▶ Run_Simulation", key="run_sim_button1")

    st.header("⚙️ Simulation Controls")
    scenario = st.selectbox("Scenario Strategy", ["Normal", "Reroute", "Holding", "Express Priority"])
    run1_button = st.button("▶ Run Simulation",key="run_sim_button2")


   
    
MAX_TRAINS = 10  # fixed max trains for RL model input size

traj = {"positions": [], "velocities": [], "tracks": [], "passed": [], "violations": [], "fuel": [], "alerts": []}


# ===============================
# 🔥 Load Models
# ===============================
rl_model = None
if controller_mode in ["AI Model", "Compare AI vs Baseline"]:
    try:
        rl_model = PPO.load("models/ppo_multitrain")
    except:
        st.warning("⚠ RL model not found. Falling back to Baseline controller.")
        rl_model = None
        if controller_mode == "AI Model":
            controller_mode = "Baseline"

ml_model = None
try:
    ml_model = joblib.load("models/train_ai_model.pkl")
except:
    st.info("ℹ ML model not found. Delay prediction disabled.")

# ===============================
# ⚙ Baseline Controller
# ===============================
def baseline_controller(env):
    n = env.n_trains
    obs = env._get_obs()
    actions = np.zeros(n, dtype=np.float32)
    target_speed = 0.6 * env.max_speed
    for i in range(n):
        vel = obs[n+i]
        headway = (env.positions[i+1] - env.positions[i]) if i < n-1 else env.L
        dv = target_speed - vel
        a = np.clip(0.2*dv, -env.max_accel, env.max_accel)
        if headway < env.min_gap*1.2:
            a = min(a, -1.0)
        actions[i] = a
    return actions



# ===============================
# ▶ Run Simulation
# ===============================
def run_episode(env, policy=None):
    obs, _ = env.reset()
    terminated = False
    traj = {"positions": [], "velocities": [], "tracks": [], "passed": [], "violations": [], "fuel": [], "alerts": []}

    while not terminated:
        if policy is None:
            action = baseline_controller(env)
        else:
            action, _ = policy.predict(obs, deterministic=True)

        obs, _, terminated, _, info = env.step(action)

        traj["positions"].append(env.positions.copy())
        traj["velocities"].append(env.velocities.copy())
        traj["tracks"].append(env.tracks.copy())
        traj["passed"].append(info["passed_count"])
        traj["violations"].append(info["violations"])
        traj["fuel"].append(info["fuel_proxy"])

        # Use collision_warning for alerts
        alerts = collision_warning(env.positions, env.tracks, threshold=env.min_gap)
        traj["alerts"].append(alerts)

    for k in traj:
        traj[k] = np.array(traj[k], dtype=object)
    return traj

# ===============================
# 📊 Helper Functions
# ===============================
def plot_trains(traj):
    positions = np.array(traj["positions"].tolist())
    tracks = np.array(traj["tracks"].tolist())
    T, n = positions.shape
    fig = go.Figure()
    for i in range(n):
        fig.add_trace(go.Scatter(
            y=positions[:, i],
            x=np.arange(T),
            mode='lines+markers',
            name=f"Train {i+1} (Track {tracks[-1,i]})"
        ))
    fig.update_layout(xaxis_title="Time step", yaxis_title="Position", height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_kpi_curves(traj):
    steps = np.arange(len(traj["fuel"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=traj["passed"], mode="lines+markers", name="🚆 Throughput"))
    fig.add_trace(go.Scatter(x=steps, y=traj["violations"], mode="lines+markers", name="⚠ Safety Violations"))
    fig.add_trace(go.Scatter(x=steps, y=traj["fuel"], mode="lines+markers", name="⛽ Fuel Proxy"))
    fig.update_layout(title="📊 KPI Evolution Over Time", xaxis_title="Time Step", yaxis_title="Value", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def predict_delays(velocities, positions, assigned_tracks):
    if ml_model is None:
        return np.zeros(len(velocities))
    n_trains = len(velocities)
    weights = np.full(n_trains, 50)
    assigned_stations = [(i % n_stations) + 1 for i in range(n_trains)]
    features = pd.DataFrame({
        "Speed": velocities,
        "Weight": weights,
        "Distance": positions,
        "Track": assigned_tracks,
        "Stations": assigned_stations
    })
    predicted_delays = ml_model.predict(features.values)
    return predicted_delays

def collision_warning(positions, tracks, threshold=1500):
    warnings = []
    for track_id in np.unique(tracks):
        idx = np.where(tracks == track_id)[0]
        sorted_idx = idx[np.argsort(positions[idx])]
        for i in range(len(sorted_idx)-1):
            t1, t2 = sorted_idx[i], sorted_idx[i+1]
            if positions[t2] - positions[t1] < threshold:
                warnings.append(f"⚠ Train {t1+1} and Train {t2+1} are too close on Track {track_id}!")
    return warnings

# ===============================
# 🚀 MAIN EXECUTION
# ===============================
if run_button:
    if controller_mode == "Compare AI vs Baseline" and rl_model is not None:
        st.info("🔎 Running both RL Agent and Baseline for comparison...")

        # RL simulation
        env_rl = MultiTrainCorridorEnv(n_trains=n_trains, n_tracks=n_tracks, n_stations=n_stations, episode_time=episode_time, max_trains=MAX_TRAINS)
        traj_rl = run_episode(env_rl, policy=rl_model)

        # Baseline simulation
        env_base = MultiTrainCorridorEnv(n_trains=n_trains, n_tracks=n_tracks, n_stations=n_stations, episode_time=episode_time, max_trains=MAX_TRAINS)
        traj_base = run_episode(env_base, policy=None)

        # Compare KPIs
        col1, col2 = st.columns(2)
        with col1:
           throughput_rl = traj_rl['passed'][-1] / (max(1, len(traj_rl['passed']))/60)
           st.metric("🚆 Throughput (RL)", f"{throughput_rl:.2f} trains/hour")
           st.metric("⚠ Safety Violations (RL)", f"{traj_rl['violations'][-1]}")
           st.metric("⛽ Fuel Proxy (RL)", f"{traj_rl['fuel'][-1]:.2f}")

        with col2:
           throughput_base = traj_base['passed'][-1] / (max(1, len(traj_base['passed']))/60)
           st.metric("🚆 Throughput (Baseline)", f"{throughput_base:.2f} trains/hour")
           st.metric("⚠ Safety Violations (Baseline)", f"{traj_base['violations'][-1]}")
           st.metric("⛽ Fuel Proxy (Baseline)", f"{traj_base['fuel'][-1]:.2f}")

       
        # KPI Comparison Plot
        fig = go.Figure()
        steps = np.arange(len(traj_rl["fuel"]))
        fig.add_trace(go.Scatter(x=steps, y=traj_rl["passed"], mode="lines", name="RL Throughput"))
        fig.add_trace(go.Scatter(x=steps, y=traj_base["passed"], mode="lines", name="Baseline Throughput"))
        fig.add_trace(go.Scatter(x=steps, y=traj_rl["violations"], mode="lines", name="RL Safety Violations"))
        fig.add_trace(go.Scatter(x=steps, y=traj_base["violations"], mode="lines", name="Baseline Safety Violations"))
        fig.add_trace(go.Scatter(x=steps, y=traj_rl["fuel"], mode="lines", name="RL Fuel Proxy"))
        fig.add_trace(go.Scatter(x=steps, y=traj_base["fuel"], mode="lines", name="Baseline Fuel Proxy"))
        fig.update_layout(title="📊 RL vs Baseline KPI Comparison", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        steps = np.arange(len(traj_rl["fuel"]))

        # Throughput per hour
        throughput_rl = np.array(traj_rl["passed"]) / ((steps+1)/60)
        throughput_base = np.array(traj_base["passed"]) / ((steps+1)/60)

        
        radar = go.Figure()

        categories = ["Throughput", "Safety", "Fuel"]
        rl_values = [throughput_rl[-1], -traj_rl["violations"][-1], traj_rl["fuel"][-1]]
        base_values = [throughput_base[-1], -traj_base["violations"][-1], traj_base["fuel"][-1]]

        radar.add_trace(go.Scatterpolar(r=rl_values, theta=categories, fill="toself", name="RL"))
        radar.add_trace(go.Scatterpolar(r=base_values, theta=categories, fill="toself", name="Baseline"))

        radar.update_layout(title="⚖ Performance Profile", polar=dict(radialaxis=dict(visible=True)))
        st.plotly_chart(radar, use_container_width=True)

        rl_score = throughput_rl[-1] - traj_rl["violations"][-1]*2 + traj_rl["fuel"][-1]/50
        base_score = throughput_base[-1] - traj_base["violations"][-1]*2 + traj_base["fuel"][-1]/50

        st.metric("🏆 RL Efficiency Score", f"{rl_score:.2f}")
        st.metric("🏆 Baseline Efficiency Score", f"{base_score:.2f}")




    else:
        # Single simulation
        env = MultiTrainCorridorEnv(n_trains=n_trains, n_tracks=n_tracks, n_stations=n_stations, episode_time=episode_time, max_trains=MAX_TRAINS)
        policy = rl_model if controller_mode == "AI Model" and rl_model is not None else None
        traj = run_episode(env, policy=policy)

        st.success("✅ Simulation Finished!")

        # KPIs
        col1, col2, col3 = st.columns(3)
        # For single simulation
        col1.metric("🚆 Throughput", f"{traj['passed'][-1] / (max(1, len(traj['passed']))/60):.2f} trains/hour")
        col2.metric("⚠ Safety Violations", f"{traj['violations'][-1]}")
        col3.metric("⛽ Fuel Proxy", f"{traj['fuel'][-1]:.2f}")

        # KPI curves
        plot_kpi_curves(traj)

        # Train positions
        plot_trains(traj)

        # Alerts
        st.subheader("🚨 Alerts")
        last_alerts = traj["alerts"][-1]
        if isinstance(last_alerts, np.ndarray):
            last_alerts = last_alerts.tolist()
        if last_alerts and len(last_alerts) > 0:
            for alert in last_alerts:
                st.warning(alert)
        else:
            st.success("No alerts!")

        # 🚉 Track Assignments
        st.subheader("🚉 Track Assignment")

        # Round-robin assignment: each train gets a track (1 → n_tracks) cyclically
        assigned_tracks = [(i % n_tracks) + 1 for i in range(n_trains)]

        track_df = pd.DataFrame({
            "Train": [f"T{i+1}" for i in range(n_trains)],  # Train numbering starts at 1
            "Track": assigned_tracks
        })

        st.table(track_df)

        # Predicted delays
        st.subheader("⏱ Predicted Train Delays")
        predicted_delays = predict_delays(traj["velocities"][-1], env.L - traj["positions"][-1], assigned_tracks)
        delay_df = pd.DataFrame({
            "Train": [f"Train {i+1}" for i in range(n_trains)],
            "Predicted Delay": predicted_delays.astype(int),
            "Current Track": traj["tracks"][-1],
        })
        st.table(delay_df)

        # Collision warnings
        warnings = collision_warning(traj["positions"][-1], traj["tracks"][-1])
        if warnings:
            st.warning("\n".join(warnings))
        else:
            st.info("No collisions predicted ✅")

        # Retrain buttons
        st.subheader("⚡ Retrain Models")
        if st.button("Retrain ML Model"):
            st.info("ML retraining not implemented in demo.")
        if st.button("Retrain RL Agent"):
            st.info("RL retraining not implemented in demo.")


# ===============================
# 🚆 Environment and Simulation
# ===============================
class TrainEnv:
    def __init__(self, n_trains=3, n_tracks=5, steps=50,L=100):
        self.n_trains = n_trains
        self.n_tracks = n_tracks
        self.steps = steps
        self.L=L
    def reset(self):
        traj ={
        "positions" : np.zeros((self.steps, self.n_trains)),
        "fuel" : np.zeros(self.steps),
        "alerts" : [[] for _ in range(self.steps)],
         "tracks": np.zeros((self.steps, self.n_trains)),   # ✅ ensure 'tracks' exists
        }
        return  traj

    def simulate(self):
        traj = self.reset()
        for t in range(self.steps):
            for i in range(self.n_trains):
                traj["positions"][t, i] = (i + t) % self.n_tracks
                traj["tracks"][t, i] = i % self.n_tracks
            traj["fuel"][t] = random.uniform(50, 100)
            if random.random() < 0.1:
                traj["alerts"][t].append(f"Train {random.randint(1,self.n_trains)} too close!")
        return traj

# ===============================
# 📊 KPI Cards Display
# ===============================
def show_kpi_cards(traj):
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='kpi-card'><div class='kpi-value'>{traj['fuel'][-1]:.2f}</div><div class='kpi-label'>⛽ Fuel Proxy</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='kpi-card'><div class='kpi-value'>{len(traj['alerts'][-1])}</div><div class='kpi-label'>🚦 Active Alerts</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='kpi-card'><div class='kpi-value'>{len(traj['positions'][-1])}</div><div class='kpi-label'>🚆 Active Trains</div></div>", unsafe_allow_html=True)

# ===============================
# 🛰 Real-Time Train Control Map (Multiple Trains on Same Track, Spaced Apart)
# ===============================
def live_map(traj, env):
    positions = np.array(traj["positions"])
    tracks = np.array(traj["tracks"])
    steps, n_trains = positions.shape

    # Use final snapshot for map
    last_positions = positions[-1]
    last_tracks = tracks[-1]

    fig = go.Figure()

    # Draw tracks as horizontal corridors
    for t in range(env.n_tracks):
        fig.add_shape(
            type="line",
            x0=0, y0=t, x1=env.L, y1=t,
            line=dict(color="gray", width=2, dash="dot")
        )

    # Keep track of how many trains already placed on each track
    track_counts = {t: 0 for t in range(env.n_tracks)}

    # Large gap to separate trains on same track
    spacing = 100  # increase this value for more distance between trains

    # Place trains on their assigned tracks with spacing
    for i in range(n_trains):
        track_id = int(last_tracks[i])
        # Base position from trajectory + extra spacing
        pos_x = last_positions[i] + track_counts[track_id] * spacing
        track_counts[track_id] += 1

        fig.add_trace(go.Scatter(
            x=[pos_x],
            y=[track_id],
            mode="markers+text",
            name=f"Train {i+1}",
            marker=dict(size=18, color="yellow", symbol="square"),
            text=[f"T{i+1}"],
            textposition="top center"
        ))

    fig.update_layout(
        title="🛰 Real-Time Train Control Map (Multiple Trains per Track)",
        xaxis_title="Distance Along Track",
        yaxis=dict(
            title="Track #",
            tickmode="linear",
            dtick=1,
            range=[-0.5, env.n_tracks - 0.5]  # always show correct number of tracks
        ),
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 🚆 IRCTC-Style Real-Time Train Control Map
#    (Auto-Running Animation, Trains Exit & Replay)
# ===============================
def live_map(traj, env):
    positions = np.array(traj["positions"])   # shape: (steps, n_trains)
    tracks = np.array(traj["tracks"])         # shape: (steps, n_trains)
    steps, n_trains = positions.shape 

    # Colors for each train
    colors = px.colors.qualitative.Set2 * (n_trains // len(px.colors.qualitative.Set2) + 1)

    # Gap between trains on the same track
    spacing = 50  

    # Initialize figure
    fig = go.Figure()

    # Draw static track lines
    for t in range(env.n_tracks):
        fig.add_shape(
            type="line",
            x0=0, y0=t, x1=env.L, y1=t,
            line=dict(color="gray", width=2, dash="dot")
        )

    # --- Frames for animation ---
    frames = []
    for step in range(steps):
        frame_data = []
        track_counts = {t: 0 for t in range(env.n_tracks)}
        for i in range(n_trains):
            track_id = int(tracks[step, i])
            pos_x = positions[step, i] + track_counts[track_id] * spacing
            track_counts[track_id] += 1

            if pos_x <= env.L:  # only show trains within screen
                frame_data.append(go.Scatter(
                    x=[pos_x],
                    y=[track_id],
                    mode="markers+text",
                    marker=dict(size=16, color=colors[i], symbol="square"),
                    text=[f"T{i+1}"],
                    textposition="top center",
                    showlegend=False
                ))
        frames.append(go.Frame(data=frame_data, name=str(step)))

    fig.frames = frames

    # --- Initial state (first frame) ---
    fig.add_traces(frames[0].data)

    # --- Layout & Auto-Play Animation ---
    fig.update_layout(
        title="🚆 IRCTC Real-Time Train Control Map",
        xaxis=dict(title="Distance Along Track (per 1500m)", range=[0, env.L]),
        yaxis=dict(
            title="Track #",
            tickmode="linear",
            dtick=1,
            range=[-0.5, env.n_tracks - 0.5]
        ),
        template="plotly_white",
        height=600,
        updatemenus=[{
            "type": "buttons",
            "buttons": [{
                "args": [None, {"frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate"}],
                "label": "Auto-Play",
                "method": "animate"
            }],
            "showactive": False
        }]
    )

    # Automatically start animation when chart loads
    fig.update_layout(
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Step: "},
            "steps": [{"args": [[str(k)], {"frame": {"duration": 500, "redraw": True},
                                           "mode": "immediate"}],
                       "label": str(k), "method": "animate"} for k in range(len(frames))]
        }]
    )

    # Start animation immediately
    fig.layout.updatemenus[0].buttons[0].args[1]["fromcurrent"] = True

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# 📅 Gantt-Style Scheduler Timeline
# ===============================

# ===============================
# 📅 Conflict-Free Train Schedule (with coordinates)
# ===============================
def conflict_free_scheduler(env):
    return [(f"Train {i+1}", (i+1) * 10) for i in range(env.n_trains)]

def show_scheduler(env):
    df = pd.DataFrame(conflict_free_scheduler(env), columns=["Train", "Position"])
    fig = go.Figure()
    for idx, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Position"]],
            y=[row["Train"]],
            orientation="h",
            name=row["Train"],
            marker=dict(color="#38bdf8"),
            text=f"Pos={row['Position']}",
            textposition="inside"
        ))
    fig.update_layout(
        title="📅 Conflict-Free Train Schedule",
        barmode="stack",
        xaxis_title="Track Position (Distance Units)",
        yaxis_title="Train",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
# ===============================
# 🚨 Alert Panel
# ===============================
def show_alerts(alerts):
    st.subheader("🚨 Signal Alerts")
    if not alerts or len(alerts) == 0:
        st.markdown("<div class='alert-signal safe'>✅ No Alerts – All Clear</div>", unsafe_allow_html=True)
    else:
        for alert in alerts:
            if "too close" in alert:
                st.markdown(f"<div class='alert-signal danger'>{alert}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert-signal warn'>{alert}</div>", unsafe_allow_html=True)

# ===============================
# 📜 Audit Trail Styled Logs
# ===============================
def log_run(params, traj):
    return {"params": params, "fuel": traj["fuel"][-1], "alerts": traj["alerts"][-1]}

def show_audit_trail(audit_entries):
    st.subheader("📜 Audit Log")
    for entry in audit_entries:
        st.markdown(f"<div style='background:#1e293b; padding:8px; border-radius:8px; margin:5px;'>🕒 {entry['params']} ➝ Fuel={entry['fuel']:.2f}, Alerts={entry['alerts']}</div>", unsafe_allow_html=True)

# ===============================
# 🌍 Congestion Heatmap
# ===============================
def show_congestion(traj, env):
    usage = np.zeros((env.steps, env.n_tracks))
    for t in range(env.steps):
        for pos in traj["positions"][t]:
            usage[t, int(pos)] += 1
    fig = go.Figure(data=go.Heatmap(z=usage.T, colorscale="YlOrRd"))
    fig.update_layout(title="🚦 Track Congestion Heatmap", xaxis_title="Time Step", yaxis_title="Track")
    st.plotly_chart(fig, use_container_width=True)
    if n_tracks==5 and n_trains==10:
        st.markdown("<div class='alert-signal danger'>❌ Too Much Rush - Just Descrease no. of Trains or Increase no. of Tracks</div>", unsafe_allow_html=True)
    

# ===============================
# 🚂 Main Control Room App
# ===============================
#=====================================================
#   scenario = st.selectbox("Scenario Strategy", ["Normal", "Reroute", "Holding", "Express Priority"])
#==============================================================

if run1_button:
    env = TrainEnv(n_trains, n_tracks, steps)
    traj = env.simulate()

    # KPIs
    st.subheader("📊 Key Performance Indicators")
    show_kpi_cards(traj)

    # Live Map
    live_map(traj, env)

    # Scheduler
    show_scheduler(env)

    # Congestion Heatmap
    show_congestion(traj, env)

    # Alerts
    show_alerts(traj["alerts"][-1])

    # ===============================
# 🧪 Scenario Tester
# ===============================
def scenario_tester(env):
    st.subheader("🧪 Scenario Tester")
    speed = st.slider("Train Speed Multiplier", 0.5, 2.0, 1.0)
    holding = st.slider("Holding Probability", 0.0, 0.5, 0.1)
    st.write(f"Testing with speed={speed}, holding={holding}...")
    st.success("✅ Scenario simulated successfully (mock run)")



    # Audit Trail
    audit_entry = log_run({"n_trains": n_trains, "n_tracks": n_tracks, "scenario": scenario}, traj)
    st.session_state.setdefault("audit", []).append(audit_entry)
    show_audit_trail(st.session_state["audit"])



# ===============================
# 🚉 Environment for Train Simulation
# ===============================
class TrainEnv:
    def __init__(self, n_trains=5, n_tracks=3, steps=50, L=100):
        self.n_trains = n_trains
        self.n_tracks = n_tracks
        self.steps = steps
        self.L = L

    def reset(self):
        traj = {
            "positions": np.zeros((self.steps, self.n_trains)),
            "tracks": np.zeros((self.steps, self.n_trains)),
            "fuel": np.zeros(self.steps),
            "alerts": [[] for _ in range(self.steps)]
        }
        return traj

    def simulate(self):
        traj = self.reset()
        for t in range(self.steps):
            for i in range(self.n_trains):
                traj["positions"][t, i] = ((i * 20) + t * random.randint(1, 3)) % self.L
                traj["tracks"][t, i] = i % self.n_tracks
            traj["fuel"][t] = random.uniform(50, 100)
            if random.random() < 0.1:
                traj["alerts"][t].append(f"⚠️ Train {random.randint(1, self.n_trains)} too close!")
        return traj






 

