import numpy as np

def price_asian_option(paths, K, r, T, option_type="call", averaging="arithmetic"):
    """
    Monte Carlo pricer for Asian options.
    
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
    averaging : str
        "arithmetic" or "geometric".
    """
    if averaging == "arithmetic":
        average_price = np.mean(paths[:, 1:], axis=1)  # exclude S0
    elif averaging == "geometric":
        average_price = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    else:
        raise ValueError("averaging must be 'arithmetic' or 'geometric'")

    if option_type == "call":
        payoff = np.maximum(average_price - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - average_price, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    price = np.exp(-r * T) * np.mean(payoff)
    return price
