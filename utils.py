import numpy as np
import pickle
from typing import Any

def normalize_reward(reward, agent_index: int = 0) -> float:
    # rlgym returns { 'blue-0': value }
    if isinstance(reward, dict):
        if len(reward) == 0:
            return 0.0
        try:
            # handle both int and str keys
            if agent_index in reward:
                return float(reward[agent_index])
            else:
                return float(list(reward.values())[0])
        except Exception:
            return 0.0

    # numpy array case
    if isinstance(reward, np.ndarray):
        if reward.size == 0:
            return 0.0
        return float(reward.item()) if reward.size == 1 else float(reward[agent_index])

    # list/tuple case
    if isinstance(reward, (list, tuple)):
        return float(reward[agent_index]) if len(reward) > agent_index else 0.0

    # scalar case
    try:
        return float(reward)
    except Exception:
        return 0.0

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
