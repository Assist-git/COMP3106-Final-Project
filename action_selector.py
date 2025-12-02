import random
import numpy as np

def select_action(
    s,
    Q,
    epsilon,
    env_wrapper,
    raw_rlgym_env,
    tbl,
    allowed_env_actions,
    reduced_to_env,
    env_to_reduced,
    n_reduced,
    turn_bias=0.65,
    angle_threshold=0.3,
    close_thresh=3.0,
    hysteresis_steps=8
):
    left_reduced = []
    right_reduced = []
    forward_reduced = []
    if tbl is not None:
        for env_idx in allowed_env_actions:
            if env_idx not in env_to_reduced:
                continue
            steer_val = float(tbl[env_idx][1])
            throttle_val = float(tbl[env_idx][0])
            if steer_val < -0.5:
                left_reduced.append(env_to_reduced[env_idx])
            if steer_val > 0.5:
                right_reduced.append(env_to_reduced[env_idx])
            if throttle_val >= 0.9 and abs(steer_val) < 0.5:
                forward_reduced.append(env_to_reduced[env_idx])

    # compute angle to ball, side, and distance (convert to meters)
    agent_key = env_wrapper.agent_map[0]
    try:
        car = raw_rlgym_env.state.cars[agent_key]
        to_ball = raw_rlgym_env.state.ball.position - car.physics.position
        dist = float(np.linalg.norm(to_ball))        # world units (uu / cm)
        dist_m = dist / 100.0                        # convert to meters
        to_ball_norm = to_ball / (dist + 1e-8)
        forward_vec = car.physics.forward
        dot = float(np.clip(np.dot(forward_vec, to_ball_norm), -1.0, 1.0))
        angle_to_ball = float(np.arccos(dot))
        cross_z = float(np.cross(forward_vec, to_ball_norm)[2])
    except Exception:
        angle_to_ball = 0.0
        cross_z = 0.0
        dist_m = 999.0

    if not hasattr(select_action, "_hyst"):
        select_action._hyst = {}
    hyst = select_action._hyst.get(agent_key, 0)

    # CLOSE-GATE: when very close, prefer forward and set hysteresis
    if dist_m < close_thresh and forward_reduced:
        # almost always pick forward when close
        if random.random() < 0.95:
            a_reduced = random.choice(forward_reduced)
        else:
            a_reduced = random.randrange(n_reduced)
        hyst = hysteresis_steps
    elif hyst > 0 and forward_reduced:
        # keep preferring forward for a few steps after a close encounter
        if random.random() < 0.9:
            a_reduced = random.choice(forward_reduced)
        else:
            a_reduced = random.randrange(n_reduced)
        hyst -= 1
    else:
        # normal epsilon-greedy with biased exploration
        if random.random() < epsilon or s not in Q:
            if angle_to_ball > angle_threshold:
                if cross_z > 0 and left_reduced:
                    if random.random() < turn_bias:
                        a_reduced = random.choice(left_reduced)
                    else:
                        a_reduced = random.randrange(n_reduced)
                elif cross_z < 0 and right_reduced:
                    if random.random() < turn_bias:
                        a_reduced = random.choice(right_reduced)
                    else:
                        a_reduced = random.randrange(n_reduced)
                else:
                    if forward_reduced and random.random() < 0.75:
                        a_reduced = random.choice(forward_reduced)
                    else:
                        a_reduced = random.randrange(n_reduced)
            else:
                # favor forward driving when roughly aligned with ball
                if forward_reduced and random.random() < 0.95:
                    a_reduced = random.choice(forward_reduced)
                else:
                    a_reduced = random.randrange(n_reduced)
        else:
            a_reduced = int(np.argmax(Q[s]))

    select_action._hyst[agent_key] = hyst
    env_action = int(reduced_to_env[a_reduced])
    return a_reduced, env_action