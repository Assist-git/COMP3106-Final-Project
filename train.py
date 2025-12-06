from env_builder import build_rlgym_v2_env
from qlearning import train_qlearning, visualize_q_table
from utils import save_q_table
import csv
import os
from datetime import datetime

if __name__ == "__main__":
    # build both wrapper and raw env
    env_wrapper, raw_rlgym_env, lookup_table = build_rlgym_v2_env()

    config = {
        "alpha": 0.2, # learning rate
        "gamma": 0.99, # discount factor
        "epsilon": 1.0, # initial exploration
        "epsilon_min": 0.02, # min exploration
        "epsilon_decay": 0.9965, # decay per episode for exploration
        "num_episodes": 2000, # total training eps
        "max_steps": 5000, # max steps per ep
        "dist_bucket": 100.0, # dist bucket size
        "speed_bucket": 100.0, # speed bucket size
    }

    Q = train_qlearning(env_wrapper, raw_rlgym_env, config, lookup_table)

    # save Q-table
    save_q_table(Q, "data/q_table.pkl")
    print("Training finished and Q-table saved.")

    # learned q table
    print(f"Q-table has {len(Q)} unique states: {list(Q.keys())[:10]}...")
    
    try:
        # auto size based on explored states
        visualize_q_table(Q, show=True, save_path="data/qtable-vis")
    except Exception as e:
        print("Failed to visualize Q-table:", e)
    
    # training metrics to CSV to track progress
    import numpy as np
    metrics_file = "data/training_metrics.csv"
    
    # compute q table stats
    max_d = max([k[0] for k in Q.keys()]) if Q else 0
    max_s = max([k[1] for k in Q.keys()]) if Q else 0
    grid_h = max_d + 1
    grid_w = max_s + 1
    
    q_max_values = [float(np.max(qvals)) for qvals in Q.values()]
    q_avg_values = [float(np.mean(qvals)) for qvals in Q.values()]
    
    # stats for close range states
    close_range_states = {k: v for k, v in Q.items() if k[0] == 0}
    close_range_max_q = np.mean([float(np.max(v)) for v in close_range_states.values()]) if close_range_states else 0.0
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "total_episodes": config["num_episodes"],
        "total_states_discovered": len(Q),
        "grid_h": grid_h,
        "grid_w": grid_w,
        "sparsity_percent": (len(Q) / (grid_h * grid_w) * 100) if (grid_h * grid_w) > 0 else 0.0,
        "max_q_value": np.max(q_max_values) if q_max_values else 0.0,
        "mean_q_value": np.mean(q_max_values) if q_max_values else 0.0,
        "close_range_avg_q": float(close_range_max_q),
    }
    
    # write to CSV
    file_exists = os.path.exists(metrics_file)
    with open(metrics_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    
    print(f"\nTraining metrics logged to {metrics_file}")
