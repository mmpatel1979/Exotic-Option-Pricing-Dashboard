import numpy as np

def price_cliquet_option(paths, r, T, option_type="call", local_floor=0.0, local_cap=0.1):
    """
    Monte Carlo pricer for Cliquet (ratchet) options.
    A series of forward-start options with capped/floored local returns.

    paths : np.ndarray
        Shape (n_paths, steps+1). Simulated price paths.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    option_type : str
        "call" or "put" (treats payoff symmetrically).
    local_floor : float
        Minimum local return (e.g., 0%).
    local_cap : float
        Maximum local return (e.g., 10%).
    """
    # Compute local returns step by step
    local_returns = paths[:, 1:] / paths[:, :-1] - 1.0
    capped_returns = np.clip(local_returns, local_floor, local_cap)

    # Cliquet payoff is sum of capped local returns (ratcheted each step)
    ratchet_payoff = np.sum(capped_returns, axis=1)

    if option_type == "call":
        payoff = np.maximum(ratchet_payoff, 0)
    elif option_type == "put":
        payoff = np.maximum(-ratchet_payoff, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    price = np.exp(-r * T) * np.mean(payoff)
    return price
