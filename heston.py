import numpy as np
import matplotlib.pyplot as plt

def simulate_heston_paths(
    S0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    steps: int,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = True,
    return_all_paths: bool = False,
):
    rng = np.random.default_rng(seed)
    dt = T / steps
    sqrt_dt = np.sqrt(dt)

    # Effective number of Monte Carlo paths (accounting for antithetic)
    n_eff = n_paths * (2 if antithetic else 1)

    # Pre-allocate arrays (we will keep full paths only if requested)
    if return_all_paths:
        S = np.empty((n_eff, steps + 1), dtype=float)
        v = np.empty((n_eff, steps + 1), dtype=float)
    else:
        S = np.empty((n_eff, ), dtype=float)  # will store only final S_T
        v = None

    # Prepare Z draws: shape (n_paths, steps, 2) for independent normals
    Z = rng.standard_normal(size=(n_paths, steps, 2))
    Z_s_base = Z[:, :, 0]
    Z_indep = Z[:, :, 1]
    Z_v_base = rho * Z_s_base + np.sqrt(max(0.0, 1 - rho**2)) * Z_indep

    # Prepare antithetic arrays if needed
    if antithetic:
        Z_s = np.vstack([Z_s_base, -Z_s_base])
        Z_v = np.vstack([Z_v_base, -Z_v_base])
    else:
        Z_s = Z_s_base
        Z_v = Z_v_base

    # Initialize paths
    if return_all_paths:
        S[:, 0] = S0
        v[:, 0] = v0
    else:
        S_current = np.full((n_eff,), S0, dtype=float)
        v_current = np.full((n_eff,), v0, dtype=float)

    # Vectorized time stepping
    for t in range(steps):
        zs = Z_s[:, t]
        zv = Z_v[:, t]

        if return_all_paths:
            v_prev = v[:, t]
            v_prev_pos = np.maximum(v_prev, 0.0)
            dv = kappa * (theta - v_prev_pos) * dt + xi * np.sqrt(v_prev_pos) * sqrt_dt * zv
            v_next = np.maximum(v_prev + dv, 0.0)
            v[:, t+1] = v_next

            diffusion = np.sqrt(v_prev_pos * dt) * zs
            S[:, t+1] = S[:, t] * np.exp((r - 0.5 * v_prev_pos) * dt + diffusion)
        else:
            v_prev = v_current
            v_prev_pos = np.maximum(v_prev, 0.0)
            dv = kappa * (theta - v_prev_pos) * dt + xi * np.sqrt(v_prev_pos) * sqrt_dt * zv
            v_next = np.maximum(v_prev + dv, 0.0)

            diffusion = np.sqrt(v_prev_pos * dt) * zs
            S_next = S_current * np.exp((r - 0.5 * v_prev_pos) * dt + diffusion)

            v_current = v_next
            S_current = S_next

    if return_all_paths:
        return {
            'price_paths': S,        
            'var_paths': v,
            'Zs': (Z_s, Z_v),
        }
    else:
        return {
            'ST': S_current,
            'vT': v_current,
            'Zs': (Z_s, Z_v),
        }

# Plotting 
if __name__ == "__main__":
    res = simulate_heston_paths(
        S0=100,
        v0=0.04,       
        r=0.01,
        kappa=2.0,   
        theta=0.04,    
        xi=0.3,       
        rho=-0.7,      
        T=1.0,
        steps=252,
        n_paths=5,
        seed=42,
        antithetic=False,
        return_all_paths=True
    )

    # Extract results
    S_paths = res['price_paths']
    v_paths = res['var_paths']
    t = np.linspace(0, 1, S_paths.shape[1])

    # Plot stock price paths
    plt.figure(figsize=(10,5))
    for i in range(min(5, S_paths.shape[0])):
        plt.plot(t, S_paths[i], lw=1)
    plt.title("Heston Stock Price Paths")
    plt.xlabel("Time (Years)")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.show()

    # Plot variance paths
    plt.figure(figsize=(10,5))
    for i in range(min(5, v_paths.shape[0])):
        plt.plot(t, v_paths[i], lw=1)
    plt.title("Heston Variance Paths")
    plt.xlabel("Time (Years)")
    plt.ylabel("Variance")
    plt.grid(True)
    plt.show()
