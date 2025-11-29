import numpy as np

def obs_to_scalar(obs_agent: np.ndarray) -> float:
    # reduces full observation vector to a single scalar
    return float(np.linalg.norm(obs_agent))

def discretize_state(obs_agent: np.ndarray, bucket_size: float = 1.0):
    # returns a tuple key for Q-table
    val = obs_to_scalar(obs_agent)
    return (int(val // bucket_size),)
