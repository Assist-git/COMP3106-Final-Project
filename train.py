from env_builder import build_rlgym_v2_env
from qlearning import train_qlearning
from utils import save_q_table

if __name__ == "__main__":
    # build both wrapper and raw env
    env_wrapper, raw_rlgym_env = build_rlgym_v2_env()

    config = {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "num_episodes": 200,
        "bucket_size": 1.0,
        "max_steps": 2000,
    }

    Q = train_qlearning(env_wrapper, raw_rlgym_env, config)

    # save Q-table
    save_q_table(Q, "q_table.pkl")
    print("Training finished and Q-table saved.")
