from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

def main():
    out = Path("data/real/wti_returns.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = yf.download("CL=F", start="1986-01-01", auto_adjust=True)
    df = df.dropna()

    df["logp"] = np.log(df["Close"])
    df["ret"] = df["logp"].diff()
    df = df.dropna()

    rets = df["ret"].to_numpy(dtype=np.float32)
    dates = df.index.astype(str).to_numpy()

    np.savez_compressed(out, returns=rets, dates=dates)
    print("saved", out, "len", len(rets), "from", dates[0], "to", dates[-1])

if __name__ == "__main__":
    main()
