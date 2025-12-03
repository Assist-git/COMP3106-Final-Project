import numpy as np
from rlgym.rocket_league import common_values

# discretize normalized obs into dist / speed buckets
def discretize_state(obs_agent: np.ndarray,
                    dist_bucket: float = 500.0,
                    speed_bucket: float = 250.0) -> tuple:

    car_pos = obs_agent[0:3] # car position (normalized)
    ball_pos = obs_agent[6:9] # ball position (normalized)
    car_lin_vel = obs_agent[12:15] # car linear velocity (normalized)

    # un-normalize positions
    pos_scale = np.array([common_values.SIDE_WALL_X, 
                          common_values.BACK_NET_Y, 
                          common_values.CEILING_Z])
    car_pos_real = car_pos * pos_scale
    ball_pos_real = ball_pos * pos_scale
    
    # un-normalize velocity
    vel_scale = common_values.CAR_MAX_SPEED
    car_lin_vel_real = car_lin_vel * vel_scale

    dist = float(np.linalg.norm(ball_pos_real - car_pos_real))
    speed = float(np.linalg.norm(car_lin_vel_real))

    return (int(dist // dist_bucket), int(speed // speed_bucket))
