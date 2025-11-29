import numpy as np
import random
from collections import defaultdict
from actions import ACTIONS, ACTION_CONTROL_MAP
from discretizer import discretize_state
from utils import normalize_reward, to_bool
import time

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
        s = discretize_state(obs_agent, bucket_size=bucket_size)

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

            # send controls to RocketSimVis for visualization
            try:
                control_vec = ACTION_CONTROL_MAP[int(a)]
                shared_info = {"controls": {0: control_vec}}
                raw_rlgym_env.renderer.render(raw_rlgym_env.state, shared_info)
            except Exception:
                pass

            next_obs_agent = next_obs[0]
            s_next = discretize_state(next_obs_agent, bucket_size=bucket_size)

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

