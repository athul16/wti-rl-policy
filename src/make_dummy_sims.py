from pathlib import Path
import numpy as np

def main():
    out = Path("data/sims/wti_simulated_paths.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    N = 8000
    T = 600
    rng = np.random.default_rng(0)

    sigma = np.empty((N, T), dtype=np.float32)
    r = np.empty((N, T), dtype=np.float32)

    sigma[:, 0] = 0.02
    r[:, 0] = rng.normal(0.0, sigma[:, 0])

    for t in range(1, T):
        shock = rng.normal(0.0, 0.005, size=N).astype(np.float32)
        sigma[:, t] = np.clip(0.97 * sigma[:, t - 1] + 0.03 * (0.02 + np.abs(shock)), 0.005, 0.12)
        z = rng.standard_t(df=5, size=N).astype(np.float32)
        r[:, t] = 0.0001 + sigma[:, t] * z

    np.savez_compressed(out, paths=r.astype(np.float32))
    print(f"saved {out} with paths shape {r.shape}")

if __name__ == "__main__":
    main()
