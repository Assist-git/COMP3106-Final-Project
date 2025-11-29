import numpy as np
import pickle
from typing import Any

def normalize_reward(reward) -> float:
    # convert wrapper reward (dict/array/list/scalar) to a float for single agent
    if isinstance(reward, dict):
        return float(next(iter(reward.values())))
    if isinstance(reward, np.ndarray):
        return float(reward.item()) if reward.size == 1 else float(reward[0])
    if isinstance(reward, (list, tuple)):
        return float(reward[0])
    return float(reward)

def to_bool(x) -> bool:
    # convert terminated/truncated (dict/array/list/scalar) to a single bool
    if isinstance(x, dict):
        return bool(next(iter(x.values())))
    if isinstance(x, np.ndarray):
        return bool(x.item()) if x.size == 1 else bool(x[0])
    if isinstance(x, (list, tuple)):
        return bool(x[0])
    return bool(x)

def save_q_table(Q: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(dict(Q), f)

def load_q_table(path: str) -> Any:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data
