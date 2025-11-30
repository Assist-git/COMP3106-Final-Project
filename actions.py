import numpy as np

# simple action index for now
ACTIONS = [8, 16, 12, 13, 18]

# maps from action index to 8-element control vector for RocketSimVis
ACTION_CONTROL_MAP = {
    8: np.array([0,0,0,0,0,0,0,0], dtype=np.float32),   # idle
    16: np.array([1,0,0,0,0,0,0,0], dtype=np.float32),  # forward
    12: np.array([1,-1,0,-1,0,0,0,0], dtype=np.float32),# forward+left
    13: np.array([1,-1,0,-1,0,0,0,1], dtype=np.float32),# forward+left+handbrake
    18: np.array([1,0,0,0,0,0,1,0], dtype=np.float32),  # forward+boost
}