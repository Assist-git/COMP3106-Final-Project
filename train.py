from env_builder import build_rlgym_v2_env
from qlearning import train_qlearning
from utils import save_q_table

# lookup table mapping info to see actions
from rlgym.rocket_league.action_parsers import LookupTableAction
parser = LookupTableAction()
for i, act in enumerate(parser._lookup_table[:20]):
    print(i, act)

if __name__ == "__main__":
    # build both wrapper and raw env
    env_wrapper, raw_rlgym_env, lookup_table = build_rlgym_v2_env()

    config = {
        "alpha": 0.15,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.9975,
        "num_episodes": 2000,  
        "bucket_size": 1.0,
        "max_steps": 5000,
    }

    Q = train_qlearning(env_wrapper, raw_rlgym_env, config, lookup_table)

    # save Q-table
    save_q_table(Q, "data/q_table.pkl")
    print("Training finished and Q-table saved.")
