import numpy as np

def discretize_state(obs_agent: np.ndarray,
                    dist_bucket: float = 500.0,
                    speed_bucket: float = 250.0) -> tuple:

    # replaces indices with correct slices after inspecting obs_agent
    car_pos = obs_agent[0:3] # car position (x,y,z)
    ball_pos = obs_agent[6:9] # ball position (x,y,z)
    car_lin_vel = obs_agent[12:15] # car linear velocity (x,y,z)

    dist = float(np.linalg.norm(ball_pos - car_pos))
    speed = float(np.linalg.norm(car_lin_vel))

    return (int(dist // dist_bucket), int(speed // speed_bucket))
