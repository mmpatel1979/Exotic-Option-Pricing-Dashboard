import streamlit as st
import numpy as np
import plotly.graph_objects as go

# === Exotic Option Pricing Functions ===
def price_asian_option(paths, K, r, T, option_type="call", averaging="arithmetic"):
    if averaging == "arithmetic":
        average_price = np.mean(paths[:, 1:], axis=1)
    elif averaging == "geometric":
        average_price = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    else:
        raise ValueError("averaging must be 'arithmetic' or 'geometric'")

    if option_type == "call":
        payoff = np.maximum(average_price - K, 0)
    else:
        payoff = np.maximum(K - average_price, 0)

    return np.exp(-r * T) * np.mean(payoff), payoff, average_price


def price_barrier_option(paths, K, r, T, option_type="call", barrier=100, barrier_type="up-and-out"):
    ST = paths[:, -1]

    if "up" in barrier_type:
        hit_barrier = np.max(paths, axis=1) >= barrier
    elif "down" in barrier_type:
        hit_barrier = np.min(paths, axis=1) <= barrier
    else:
        raise ValueError("barrier_type must contain 'up' or 'down'")

    payoff = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)

    if "out" in barrier_type:
        payoff = np.where(hit_barrier, 0, payoff)
    elif "in" in barrier_type:
        payoff = np.where(hit_barrier, payoff, 0)

    return np.exp(-r * T) * np.mean(payoff), payoff, hit_barrier


def price_cliquet_option(paths, r, T, option_type="call", local_floor=0.0, local_cap=0.1):
    local_returns = paths[:, 1:] / paths[:, :-1] - 1.0
    capped_returns = np.clip(local_returns, local_floor, local_cap)
    ratchet_payoff = np.sum(capped_returns, axis=1)

    if option_type == "call":
        payoff = np.maximum(ratchet_payoff, 0)
    else:
        payoff = np.maximum(-ratchet_payoff, 0)

    return np.exp(-r * T) * np.mean(payoff), payoff, ratchet_payoff


# === Heston Path Simulator ===
def simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, steps, n_paths, seed=None, antithetic=True):
    rng = np.random.default_rng(seed)
    dt = T / steps
    sqrt_dt = np.sqrt(dt)

    n_eff = n_paths * (2 if antithetic else 1)
    S = np.empty((n_eff, steps + 1))
    v = np.empty((n_eff, steps + 1))

    Z = rng.standard_normal(size=(n_paths, steps, 2))
    Z_s_base = Z[:, :, 0]
    Z_v_base = rho * Z_s_base + np.sqrt(max(0.0, 1 - rho**2)) * Z[:, :, 1]

    if antithetic:
        Z_s = np.vstack([Z_s_base, -Z_s_base])
        Z_v = np.vstack([Z_v_base, -Z_v_base])
    else:
        Z_s = Z_s_base
        Z_v = Z_v_base

    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(steps):
        v_prev = np.maximum(v[:, t], 0.0)
        dv = kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev) * sqrt_dt * Z_v[:, t]
        v[:, t+1] = np.maximum(v_prev + dv, 0.0)
        diffusion = np.sqrt(v_prev * dt) * Z_s[:, t]
        S[:, t+1] = S[:, t] * np.exp((r - 0.5 * v_prev) * dt + diffusion)

    return S, v


# === Streamlit App ===
st.set_page_config(page_title="Exotic Option Pricing Dashboard", layout="wide")
st.title("Exotic Option Pricing with Heston Model")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")
S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
T = st.sidebar.number_input("Time to Maturity (T, years)", value=1.0)
v0 = st.sidebar.number_input("Initial Variance (v0)", value=0.04)
kappa = st.sidebar.number_input("Mean Reversion (kappa)", value=2.0)
theta = st.sidebar.number_input("Long-term Variance (theta)", value=0.04)
xi = st.sidebar.number_input("Vol of Vol (xi)", value=0.5)
rho = st.sidebar.number_input("Correlation (rho)", value=-0.7)
steps = st.sidebar.slider("Steps", 50, 500, 252)
n_paths = st.sidebar.slider("Number of Paths", 100, 5000, 1000)

st.sidebar.header("Option Parameters")
option_type = st.sidebar.radio("Option Type", ["call", "put"])

exotic_choice = st.sidebar.selectbox("Exotic Option Type", ["Asian", "Barrier", "Cliquet"])

if exotic_choice == "Asian":
    averaging = st.sidebar.radio("Averaging", ["arithmetic", "geometric"])
elif exotic_choice == "Barrier":
    barrier = st.sidebar.number_input("Barrier Level", value=120.0)
    barrier_type = st.sidebar.selectbox("Barrier Type", ["up-and-out", "up-and-in", "down-and-out", "down-and-in"])
elif exotic_choice == "Cliquet":
    local_floor = st.sidebar.number_input("Local Floor", value=0.0)
    local_cap = st.sidebar.number_input("Local Cap", value=0.1)

# Run simulation
S_paths, v_paths = simulate_heston_paths(S0, v0, r, kappa, theta, xi, rho, T, steps, n_paths, seed=42)

# Price the chosen exotic option
if exotic_choice == "Asian":
    price, payoff, avg_prices = price_asian_option(S_paths, K, r, T, option_type, averaging)
elif exotic_choice == "Barrier":
    price, payoff, hit_barrier = price_barrier_option(S_paths, K, r, T, option_type, barrier, barrier_type)
elif exotic_choice == "Cliquet":
    price, payoff, ratchet = price_cliquet_option(S_paths, r, T, option_type, local_floor, local_cap)

st.subheader(f"{exotic_choice} Option Price: {price:.4f}")

# Plot paths
fig = go.Figure()
t = np.linspace(0, T, steps+1)

if exotic_choice == "Barrier":
    for i in range(min(50, S_paths.shape[0])):
        color = "red" if hit_barrier[i] and ("out" in barrier_type) else "green"
        fig.add_trace(go.Scatter(x=t, y=S_paths[i], mode='lines', line=dict(width=1, color=color), opacity=0.5))
    fig.add_hline(y=barrier, line_dash="dash", line_color="red", annotation_text="Barrier")

elif exotic_choice == "Asian":
    for i in range(min(50, S_paths.shape[0])):
        fig.add_trace(go.Scatter(x=t, y=S_paths[i], mode='lines', line=dict(width=1), opacity=0.5))
        # Add final average as a horizontal marker
        fig.add_trace(go.Scatter(x=[T], y=[avg_prices[i]], mode='markers', marker=dict(color='orange', size=6)))

elif exotic_choice == "Cliquet":
    for i in range(min(50, S_paths.shape[0])):
        fig.add_trace(go.Scatter(x=t, y=S_paths[i], mode='lines', line=dict(width=1), opacity=0.5))
    # Show histogram-like final payoff marker
    fig.add_trace(go.Scatter(x=[T]*min(50, len(ratchet)), y=S_paths[:min(50, len(ratchet)), -1],
                             mode='markers', marker=dict(color='purple', size=6),
                             name="Final ST vs ratchet payoff"))

fig.update_layout(title="Simulated Stock Price Paths (Heston)", xaxis_title="Time (Years)", yaxis_title="Stock Price", height=500)
st.plotly_chart(fig, use_container_width=True)

# Plot payoff distribution
fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=payoff, nbinsx=50, name="Payoff Distribution"))
fig2.update_layout(title="Payoff Distribution", xaxis_title="Payoff", yaxis_title="Frequency", height=400)
st.plotly_chart(fig2, use_container_width=True)
