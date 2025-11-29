import numpy as np

ACTIONS = [0, 1, 2, 3, 4, 5]

# maps from action index to 8-element control vector for RocketSimVis
ACTION_CONTROL_MAP = {
    0: np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),   # idle
    1: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),   # forward
    2: np.array([1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32),  # forward+left
    3: np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),   # forward+right
    4: np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),   # jump
    5: np.array([1, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),   # boost
}