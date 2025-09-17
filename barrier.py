import numpy as np

def price_barrier_option(paths, K, r, T, option_type="call", barrier=100, barrier_type="up-and-out"):
    """
    Monte Carlo pricer for Barrier options.

    paths : np.ndarray
        Shape (n_paths, steps+1). Simulated price paths.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    option_type : str
        "call" or "put".
    barrier : float
        Barrier level.
    barrier_type : str
        "up-and-out", "down-and-out", "up-and-in", "down-and-in".
    """
    ST = paths[:, -1]
    hit_barrier = None

    if "up" in barrier_type:
        hit_barrier = np.max(paths, axis=1) >= barrier
    elif "down" in barrier_type:
        hit_barrier = np.min(paths, axis=1) <= barrier
    else:
        raise ValueError("barrier_type must contain 'up' or 'down'")

    # Vanilla payoff
    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Apply barrier condition
    if "out" in barrier_type:
        payoff = np.where(hit_barrier, 0, payoff)
    elif "in" in barrier_type:
        payoff = np.where(hit_barrier, payoff, 0)

    price = np.exp(-r * T) * np.mean(payoff)
    return price
