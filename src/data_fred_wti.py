from pathlib import Path
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

def main():
    out = Path("data/real/wti_fred_returns.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    s = pdr.DataReader("DCOILWTICO", "fred", start="1986-01-02")
    s = s.rename(columns={"DCOILWTICO": "price"}).dropna()
    s["price"] = s["price"].astype(float)

    s["logp"] = np.log(s["price"])
    s["ret"] = s["logp"].diff()
    s = s.dropna()

    rets = s["ret"].to_numpy(dtype=np.float32)
    dates = s.index.astype(str).to_numpy()

    np.savez_compressed(out, returns=rets, dates=dates)
    print("saved", out, "len", len(rets), "from", dates[0], "to", dates[-1])

if __name__ == "__main__":
    main()
