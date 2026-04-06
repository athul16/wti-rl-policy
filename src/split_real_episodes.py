from pathlib import Path
import numpy as np

def make_episodes(r: np.ndarray, T: int, stride: int) -> np.ndarray:
    episodes = []
    for start in range(0, len(r) - T, stride):
        ep = r[start:start+T]
        if np.isfinite(ep).all():
            episodes.append(ep.astype(np.float32))
    return np.stack(episodes)

def main():
    src = Path("data/real/wti_returns.npz")
    obj = np.load(src, allow_pickle=True)
    r = np.asarray(obj["returns"], dtype=np.float32)
    r = r[np.isfinite(r)]

    T = 600
    stride = 5

    split = int(0.8 * len(r))
    r_train = r[:split]
    r_test  = r[split:]

    train_paths = make_episodes(r_train, T, stride)
    test_paths  = make_episodes(r_test,  T, stride)

    out_train = Path("data/sims/wti_real_train.npz")
    out_test  = Path("data/sims/wti_real_test.npz")
    out_train.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_train, paths=train_paths)
    np.savez_compressed(out_test, paths=test_paths)

    print("saved", out_train, "shape", train_paths.shape)
    print("saved", out_test,  "shape", test_paths.shape)

if __name__ == "__main__":
    main()
