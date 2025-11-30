import numpy as np
import random
from collections import defaultdict
from actions import ACTIONS, ACTION_CONTROL_MAP
from discretizer import discretize_state
from utils import normalize_reward, to_bool
import time

_prev_control = np.zeros(8, dtype=np.float32)

def smooth_control(new, alpha=0.4):
    """
    Blend new control vector with previous one for smoother visualization.
    alpha=0.4 means 40% new, 60% old.
    """
    global _prev_control
    new = np.array(new, dtype=np.float32)
    _prev_control = alpha * new + (1 - alpha) * _prev_control
    return _prev_control

def train_qlearning(env_wrapper, raw_rlgym_env, config):
    alpha = config.get("alpha", 0.1)
    gamma = config.get("gamma", 0.99)
    epsilon = config.get("epsilon", 1.0)
    epsilon_min = config.get("epsilon_min", 0.05)
    epsilon_decay = config.get("epsilon_decay", 0.995)
    num_episodes = config.get("num_episodes", 200)
    bucket_size = config.get("bucket_size", 1.0)
    max_steps = config.get("max_steps", 2000)

    Q = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=np.float32))

    # print checks for debugging
    obs = env_wrapper.reset()
    print("Agent map after reset:", env_wrapper.agent_map)
    print("Obs shape:", obs.shape)

    for ep in range(1, num_episodes + 1):
        obs = env_wrapper.reset()
        obs_agent = obs[0]
        s = discretize_state(obs_agent, dist_bucket=500.0, speed_bucket=500.0)

        done = False
        ep_return = 0.0
        steps = 0

        while not done and steps < max_steps:
            # epsilon-greedy
            if random.random() < epsilon or s not in Q:
                a = random.choice(ACTIONS)
            else:
                a = ACTIONS[int(np.argmax(Q[s]))]

            actions = np.array([[int(a)]], dtype=np.int32)
            next_obs, reward, terminated, truncated, info = env_wrapper.step(actions)

            # normalize reward and done
            r = normalize_reward(reward)
            done = to_bool(terminated) or to_bool(truncated)
            
            # slower steps to see what bot is doing in visualizer
            if steps % 4 == 0:
                try:
                    raw_vec = ACTION_CONTROL_MAP[int(a)]
                    control_vec = smooth_control(raw_vec)
                    shared_info = {"controls": {0: control_vec}}
                    raw_rlgym_env.renderer.render(raw_rlgym_env.state, shared_info)

                    time.sleep(0.02)
                except Exception as e:
                    print("Renderer error (non-fatal):", e)

            # send controls to RocketSimVis for visualization
            try:
                control_vec = ACTION_CONTROL_MAP[int(a)]
                shared_info = {"controls": {0: control_vec}}
                raw_rlgym_env.renderer.render(raw_rlgym_env.state, shared_info)
            except Exception:
                pass

            next_obs_agent = next_obs[0]
            s_next = discretize_state(next_obs_agent, dist_bucket=500.0, speed_bucket=500.0)

            best_next = float(np.max(Q[s_next])) if s_next in Q else 0.0
            a_idx = ACTIONS.index(a)
            Q[s][a_idx] += alpha * (r + gamma * best_next - Q[s][a_idx])

            s = s_next
            ep_return += r
            steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if ep % 5 == 0:
            print(f"Episode {ep}: return={ep_return:.3f}, steps={steps}, epsilon={epsilon:.3f}")

    return Q

