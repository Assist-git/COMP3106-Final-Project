import numpy as np
import random
from collections import defaultdict
from discretizer import discretize_state
from utils import normalize_reward, to_bool
import time
import csv
import matplotlib.pyplot as plt
from action_selector import select_action
import os as os

# Global state for rendering control smoothing
_prev_control = np.zeros(8, dtype=np.float32)

# smooth control output for rendering using exponential moving average
# this is to be able to see the smaller movements the car makes
def smooth_control(new, alpha=0.4):
    global _prev_control
    new = np.array(new, dtype=np.float32)
    _prev_control = alpha * new + (1 - alpha) * _prev_control
    return _prev_control

# extracts the action lookup table from various parser wrappers used by RLGym
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
    # extract hyperparameters from config
    alpha = config.get("alpha", 0.2)
    gamma = config.get("gamma", 0.99)
    epsilon = config.get("epsilon", 1.0)
    epsilon_min = config.get("epsilon_min", 0.02)
    epsilon_decay = config.get("epsilon_decay", 0.9965)
    num_episodes = config.get("num_episodes", 2000)
    max_steps = config.get("max_steps", 5000)
    dist_bucket = config.get("dist_bucket", 100.0)
    speed_bucket = config.get("speed_bucket", 100.0)

    # acquire lookup table from RLGym
    tbl = lookup_table
    if tbl is None:
        tbl = get_lookup_table_from_parser(raw_rlgym_env.action_parser)

    if tbl is None:
        tbl = None
    else:
        tbl = np.asarray(tbl)

    # allowed reduced action set
    allowed_env_actions = [12, 14, 16, 18] # forward left, forward, forward right, boost forward

    # bidirectional mapping between reduced action space and RLGym environment actions
    reduced_to_env = {i: env_idx for i, env_idx in enumerate(allowed_env_actions)}
    env_to_reduced = {env_idx: i for i, env_idx in reduced_to_env.items()}
    n_reduced = len(allowed_env_actions)

    # initialize Q-table: state -> [Q-values for each reduced action]
    Q = defaultdict(lambda: np.zeros(n_reduced, dtype=np.float32))

    obs = env_wrapper.reset()

    episode_stats = []

    try:
        for ep in range(1, num_episodes + 1):
            # reset environment and get initial state
            obs = env_wrapper.reset()
            obs_agent = obs[0]
            s = discretize_state(obs_agent, dist_bucket=dist_bucket, speed_bucket=speed_bucket)

            # track ball touches to detect new contact events
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
            dist_sum = 0.0
            align_sum = 0.0
            forward_action_count = 0
            time_to_first_touch = None
            had_terminated = False
            states_visited = set([s])  # Track unique states in this episode

            while not done and steps < max_steps:
                # select action using steering-aware ε-greedy policy
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

                # step environment with the selected action
                actions = np.array([[env_action]], dtype=np.int32)
                next_obs, reward, terminated, truncated, info = env_wrapper.step(actions)

                # normalize reward and detect episode termination
                r = normalize_reward(reward)
                this_terminated = to_bool(terminated)
                this_truncated = to_bool(truncated)
                done = this_terminated or this_truncated
                had_terminated = had_terminated or this_terminated

                # render the environment using the action lookup table
                if tbl is not None and steps % 4 == 0:
                    try:
                        env_vec = tbl[env_action]
                        control_vec = smooth_control(env_vec)
                        shared_info = {"controls": {agent_key: control_vec}}
                        raw_rlgym_env.renderer.render(raw_rlgym_env.state, shared_info)
                        #time.sleep(0.02)
                    except Exception as e:
                        print("Renderer error (non-fatal):", e)

                # collect diagnostic metrics: distance to ball and alignment with ball direction
                try:
                    car = raw_rlgym_env.state.cars[agent_key]
                    to_ball = raw_rlgym_env.state.ball.position - car.physics.position
                    dist = float(np.linalg.norm(to_ball))
                    to_ball_norm = to_ball / (dist + 1e-8)
                    forward_vec = car.physics.forward
                    alignment = float(np.clip(np.dot(forward_vec, to_ball_norm), -1.0, 1.0))
                except Exception:
                    dist = 0.0
                    alignment = 0.0

                dist_sum += dist
                align_sum += alignment

                # count forward actions to measure straight driving
                try:
                    if tbl is not None:
                        env_vec = tbl[env_action]
                        throttle_val = float(env_vec[0])
                        if throttle_val >= 0.9:
                            forward_action_count += 1
                except Exception:
                    pass

                # compute next state
                next_obs_agent = next_obs[0]
                s_next = discretize_state(next_obs_agent, dist_bucket=dist_bucket, speed_bucket=speed_bucket)

                # bellman update: Q[s][a] += alpha * (r + gamma * max(Q[s_next]) - Q[s][a])
                best_next = float(np.max(Q[s_next])) if s_next in Q else 0.0
                Q[s][a_reduced] += alpha * (r + gamma * best_next - Q[s][a_reduced])

                s = s_next
                states_visited.add(s_next)  # track visited states
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
                    if time_to_first_touch is None:
                        time_to_first_touch = steps

                prev_ball_touches = current_touches
                steps += 1

            # decay epsilon for next episode
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # per episode data
            avg_dist = float(dist_sum / steps) if steps > 0 else 0.0
            avg_align = float(align_sum / steps) if steps > 0 else 0.0
            pct_forward = float(forward_action_count / steps) if steps > 0 else 0.0

            # store episode stats
            episode_stats.append({
                "episode": ep,
                "return": ep_return,
                "touches": touches,
                "avg_dist": avg_dist,
                "avg_align": avg_align,
                "pct_forward": pct_forward,
                "time_to_first_touch": (time_to_first_touch if time_to_first_touch is not None else -1),
                "terminated": bool(had_terminated),
            })

            if ep % 10 == 0:
                print(f"Episode {ep}: return={ep_return:.3f}, touches={touches}, steps={steps}, epsilon={epsilon:.3f}, states_visited={len(states_visited)}")
                recent = episode_stats[-10:]
                if len(recent) > 0:
                    avg_touches = np.mean([e["touches"] for e in recent])
                else:
                    avg_touches = 0.0
                print(f"last 10 episodes: avg touches={avg_touches: .2f}")

    except KeyboardInterrupt:
        print("training interrupt, saving q table")
        return Q

    # save stats CSV file
    if len(episode_stats) > 0:
        keys = list(episode_stats[0].keys())
        with open("data/episode_stats.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(episode_stats)

    # for plotting
    episodes = [e["episode"] for e in episode_stats]
    touch_counts = [e["touches"] for e in episode_stats]
    avg_aligns = [e["avg_align"] for e in episode_stats]

    window = 20
    plt.figure(figsize=(12,6))
    ax1 = plt.gca()
    ax1.plot(episodes, touch_counts, label="Touches per episode", alpha=0.6)
    if len(touch_counts) >= window:
        moving_avg = np.convolve(touch_counts, np.ones(window)/window, mode="valid")
        ax1.plot(episodes[window-1:], moving_avg, label=f"{window}-episode moving avg", color="red", linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Touches")

    ax2 = ax1.twinx()
    ax2.plot(episodes, avg_aligns, label="Avg alignment", color="green", alpha=0.6)
    ax2.set_ylabel("Avg alignment (dot)")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("Training diagnostics: touches and average alignment")
    plt.grid(True)
    plt.show()

    return Q

# visualise learned q-table
def visualize_q_table(Q_dict_like, grid_h=None, grid_w=None, show=True, save_path=None):
    Q_dict = dict(Q_dict_like)
    if len(Q_dict) == 0:
        print("Q-table empty, nothing to visualize.")
        return

    sample = next(iter(Q_dict.values()))
    n_actions = len(sample)
    n_states = len(Q_dict)

    # grid dimensions from state space
    max_d = 0
    max_s = 0
    for key in Q_dict.keys():
        if isinstance(key, tuple) and len(key) >= 2:
            d_bucket, s_bucket = key[0], key[1]
            if isinstance(d_bucket, int) and isinstance(s_bucket, int):
                max_d = max(max_d, d_bucket)
                max_s = max(max_s, s_bucket)
    
    grid_h = (max_d + 1) if max_d >= 0 else 1
    grid_w = (max_s + 1) if max_s >= 0 else 1
    
    # grids for max-Q, avg-Q, and visitation count
    q_max_grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    q_avg_grid = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    visited_grid = np.zeros((grid_h, grid_w), dtype=np.int32)

    # populate grids from Q-table
    for (d_bucket, s_bucket), qvals in Q_dict.items():
        if not (isinstance(d_bucket, int) and isinstance(s_bucket, int)):
            continue
        if 0 <= d_bucket < grid_h and 0 <= s_bucket < grid_w:
            q_max_grid[d_bucket, s_bucket] = float(np.max(qvals))
            q_avg_grid[d_bucket, s_bucket] = float(np.mean(qvals))
            visited_grid[d_bucket, s_bucket] = 1

    q_max_plot = np.nan_to_num(q_max_grid, nan=0.0)
    q_avg_plot = np.nan_to_num(q_avg_grid, nan=0.0)
    
    # summary
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel 1: Max Q-value heatmap (shows best learned action value per state)
    im1 = axes[0].imshow(q_max_plot, cmap='viridis', interpolation='nearest', aspect='auto')
    axes[0].set_title(f'Max Q-value per State\n(Grid: {grid_h}x{grid_w}, States: {n_states})')
    axes[0].set_xlabel('Speed Bucket')
    axes[0].set_ylabel('Distance Bucket')
    plt.colorbar(im1, ax=axes[0], label='Q-value')
    
    # Panel 2: Average Q-value heatmap (shows mean Q-value across actions per state)
    im2 = axes[1].imshow(q_avg_plot, cmap='plasma', interpolation='nearest', aspect='auto')
    axes[1].set_title('Average Q-value per State')
    axes[1].set_xlabel('Speed Bucket')
    axes[1].set_ylabel('Distance Bucket')
    plt.colorbar(im2, ax=axes[1], label='Q-value')
    
    # Panel 3: State visitation heatmap (shows which states were discovered)
    im3 = axes[2].imshow(visited_grid, cmap='Greys', interpolation='nearest', aspect='auto')
    axes[2].set_title('Discovered States (1=visited, 0=never seen)')
    axes[2].set_xlabel('Speed Bucket')
    axes[2].set_ylabel('Distance Bucket')
    plt.colorbar(im3, ax=axes[2], label='Visited')

    plt.tight_layout()
    
    # save figure
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        ts = int(time.time())
        filename = f"qtable_{ts}.png"
        outpath = os.path.join(save_path, filename)
        fig.savefig(outpath, bbox_inches='tight', dpi=100)
        print(f"Saved Q-table visualization to {outpath}")

    if show:
        plt.show()

    # summary statistics
    print(f"\n=== Q-Table Summary Statistics ===")
    print(f"Total unique states discovered: {n_states}")
    print(f"Grid dimensions: {grid_h} (distance) × {grid_w} (speed)")
    print(f"Max Q-value: {np.max(q_max_plot):.4f}")
    print(f"Mean Q-value: {np.mean(q_max_plot[q_max_plot > 0]):.4f}" if np.any(q_max_plot > 0) else "Mean Q-value: 0.0000")
    print(f"Sparsity: {np.sum(visited_grid) / (grid_h * grid_w) * 100:.1f}% states explored")
    
    # Print per-distance statistics
    print(f"\nPer-distance bucket analysis:")
    for d in range(min(grid_h, 10)):
        visited_count = np.sum(visited_grid[d, :])
        if visited_count > 0:
            max_q_vals = q_max_plot[d, visited_grid[d, :] > 0]
            print(f"  Distance {d*100}-{(d+1)*100}: {visited_count} speed states, avg max-Q={np.mean(max_q_vals):.4f}")
    if grid_h > 10:
        print(f"  ... ({grid_h - 10} more distance buckets)")
