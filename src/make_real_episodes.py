from pathlib import Path
import numpy as np

def main():
    src = Path("data/real/wti_returns.npz")
    out = Path("data/sims/wti_real_episodes.npz")

    obj = np.load(src, allow_pickle=True)
    r = np.asarray(obj["returns"], dtype=np.float32)

    r = r[np.isfinite(r)]

    T = 600
    stride = 5

    episodes = []
    for start in range(0, len(r) - T, stride):
        ep = r[start:start+T]
        if np.isfinite(ep).all():
            episodes.append(ep)

    paths = np.stack(episodes).astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, paths=paths)
    print("saved", out, "paths shape", paths.shape)

if __name__ == "__main__":
    main()
