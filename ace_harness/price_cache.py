"""One snapshot of downloaded prices per campaign, so every cell and every
replicate reads identical data.

Yahoo recomputes adjusted prices on every request (and rescales history after
each new dividend), so in RUN 1 no two cells received the same frames: every
manifest's data_hash differed, even between cells that ran minutes apart. With
ACE_PRICE_CACHE=<dir> (run_run1_refresh.sh sets OUT_ROOT/price_cache) the first
cleaned download of a (source, ticker, start, end) is pickled there and every
later cell loads it instead of downloading again; pickle round-trips a
DataFrame exactly. Without the variable nothing changes.

The ACE engine and the three ReAct baselines each keep their own entries
(source), because each script cleans the raw download its own way.
"""
import os
import socket

import pandas as pd

ENV = "ACE_PRICE_CACHE"


def _path(source, ticker, start, end):
    root = os.environ.get(ENV)
    return os.path.join(root, f"{source}__{ticker}__{start}__{end}.pkl") if root else None


def fetch(source, ticker, start, end, download):
    """download() the first time a campaign needs these prices, the stored frame after that."""
    path = _path(source, ticker, start, end)
    if path and os.path.exists(path):
        return pd.read_pickle(path)
    df = download()
    if not path:
        return df
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{socket.gethostname()}.{os.getpid()}.tmp"
    try:
        df.to_pickle(tmp)
        try:
            os.link(tmp, path)  # atomic, and the first writer wins if two cells race
        except FileExistsError:
            pass
        except OSError:  # filesystem without hard links
            if not os.path.exists(path):
                os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return pd.read_pickle(path)  # even the cell that downloaded uses the stored snapshot
