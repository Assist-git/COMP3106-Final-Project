import numpy as np
import random
from collections import defaultdict
from discretizer import discretize_state
from utils import normalize_reward, to_bool
import time
import csv
import matplotlib.pyplot as plt
from action_selector import select_action

_prev_control = np.zeros(8, dtype=np.float32)

def smooth_control(new, alpha=0.4):
    global _prev_control
    new = np.array(new, dtype=np.float32)
    _prev_control = alpha * new + (1 - alpha) * _prev_control
    return _prev_control

# lookup table parser helper function
def get_lookup_table_from_parser(parser):
    # direct attributes
    for attr in ("lookup_table", "table", "_lookup_table"):
        if hasattr(parser, attr):
            return getattr(parser, attr)

    # wrapper name used by RepeatAction
    inner = getattr(parser, "inner", None)
    if inner is not None:
        for attr in ("lookup_table", "table", "_lookup_table"):
            if hasattr(inner, attr):
                return getattr(inner, attr)

    # other plausible nested names
    for name in ("action_parser", "parser", "inner_parser"):
        inner = getattr(parser, name, None)
        if inner is not None:
            for attr in ("lookup_table", "table", "_lookup_table"):
                if hasattr(inner, attr):
                    return getattr(inner, attr)
    return None

def train_qlearning(env_wrapper, raw_rlgym_env, config, lookup_table=None):
    alpha = config.get("alpha", 0.1)
    gamma = config.get("gamma", 1.0)
    epsilon = config.get("epsilon", 1.0)
    epsilon_min = config.get("epsilon_min", 0.1)
    epsilon_decay = config.get("epsilon_decay", 0.999)
    num_episodes = config.get("num_episodes", 1000)
    max_steps = config.get("max_steps", 5000)

    tbl = lookup_table
    if tbl is None:
        tbl = get_lookup_table_from_parser(raw_rlgym_env.action_parser)

    if tbl is None:
        tbl = None
    else:
        tbl = np.asarray(tbl)

    allowed_env_actions = [12, 14, 16, 18]

    # reduced env mappings
    reduced_to_env = {i: env_idx for i, env_idx in enumerate(allowed_env_actions)}
    env_to_reduced = {env_idx: i for i, env_idx in reduced_to_env.items()}
    n_reduced = len(allowed_env_actions)

    # Q uses reduced action space
    Q = defaultdict(lambda: np.zeros(n_reduced, dtype=np.float32))

    # Debug prints
    obs = env_wrapper.reset()

    episode_stats = []

    try:
        for ep in range(1, num_episodes + 1):
            obs = env_wrapper.reset()
            obs_agent = obs[0]
            s = discretize_state(obs_agent, dist_bucket=500.0, speed_bucket=500.0)

            # track cumulative ball_touches so we only count new touches
            agent_key = env_wrapper.agent_map[0]
            try:
                car = raw_rlgym_env.state.cars[agent_key]
                prev_ball_touches = int(car.ball_touches)
            except Exception:
                prev_ball_touches = 0

            done = False
            ep_return = 0.0
            steps = 0
            touches = 0

            while not done and steps < max_steps:
                # use the steering-aware selector
                a_reduced, env_action = select_action(
                    s=s,
                    Q=Q,
                    epsilon=epsilon,
                    env_wrapper=env_wrapper,
                    raw_rlgym_env=raw_rlgym_env,
                    tbl=tbl,
                    allowed_env_actions=allowed_env_actions,
                    reduced_to_env=reduced_to_env,
                    env_to_reduced=env_to_reduced,
                    n_reduced=n_reduced,
                    turn_bias=0.85,      
                    angle_threshold=0.12
                )

                # step with env action index returned by selector
                actions = np.array([[env_action]], dtype=np.int32)
                next_obs, reward, terminated, truncated, info = env_wrapper.step(actions)


                # normalize reward and done
                r = normalize_reward(reward)
                done = to_bool(terminated) or to_bool(truncated)

                # visualize using env's lookup table
                if tbl is not None and steps % 4 == 0:
                    try:
                        env_vec = tbl[env_action]
                        control_vec = smooth_control(env_vec)
                        shared_info = {"controls": {agent_key: control_vec}}
                        raw_rlgym_env.renderer.render(raw_rlgym_env.state, shared_info)
                    except Exception as e:
                        print("Renderer error (non-fatal):", e)

                # update state and Q
                next_obs_agent = next_obs[0]
                s_next = discretize_state(next_obs_agent, dist_bucket=500.0, speed_bucket=500.0)

                best_next = float(np.max(Q[s_next])) if s_next in Q else 0.0
                Q[s][a_reduced] += alpha * (r + gamma * best_next - Q[s][a_reduced])

                s = s_next
                ep_return += r

                # count only real new ball touches using the car's cumulative counter
                try:
                    car = raw_rlgym_env.state.cars[agent_key]
                    current_touches = int(car.ball_touches)
                except Exception:
                    current_touches = prev_ball_touches

                new_touches = current_touches - prev_ball_touches
                if new_touches > 0:
                    touches += new_touches
                    print(f"Episode {ep}, Step {steps}, new_touches={new_touches}, total_touches={touches}, ep_return={ep_return:.3f}")

                prev_ball_touches = current_touches
                steps += 1

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            episode_stats.append((ep, ep_return, touches))

            if ep % 10 == 0:
                print(f"Episode {ep}: return={ep_return:.3f}, touches={touches}, steps={steps}, epsilon={epsilon:.3f}")
                avg_touches = np.mean([t for _, _, t in episode_stats[-10:]])
                print(f"last 10 episodes: avg touches={avg_touches: .2f}")

    except KeyboardInterrupt:
        print("training interrupt, saving q table")
        return Q

    # save stats
    with open("data/episode_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "touches"])
        writer.writerows(episode_stats)

    episodes = [ep for ep, ret, t in episode_stats]
    touch_counts = [t for ep, ret, t in episode_stats]

    window = 20
    if len(touch_counts) >= window:
        moving_avg = np.convolve(touch_counts, np.ones(window)/window, mode="valid")
        plt.figure(figsize=(10,5))
        plt.plot(episodes, touch_counts, label="Touches per episode", alpha=0.5)
        plt.plot(episodes[window-1:], moving_avg, label=f"{window}-episode moving average", color="red", linewidth=2)
    else:
        plt.figure(figsize=(10,5))
        plt.plot(episodes, touch_counts, label="Touches per episode", alpha=0.5)

    plt.xlabel("Episode")
    plt.ylabel("Touches")
    plt.title("Ball touches over training")
    plt.legend()
    plt.grid(True)
    plt.show()

    return Q

