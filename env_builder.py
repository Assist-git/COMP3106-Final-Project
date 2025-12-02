def build_rlgym_v2_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper

    from rewards import TouchReward, BoostAlignmentReward, SpeedTowardBallReward, FacingBallReward, SteeringTowardBallReward, BallToGoalReward
    from renderer import RocketSimVisRenderer

    spawn_opponents = False
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 1
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )

    reward_fn = CombinedReward(
        (SpeedTowardBallReward(), 0.2),
        (TouchReward(), 50),
        (BallToGoalReward(), 0.5),
        (BoostAlignmentReward(), 0.1),
        (FacingBallReward(), 0.05),
        (SteeringTowardBallReward(coef=2.0, clip=0.5, min_dist=3.0), 1.0)
    )

    obs_builder = DefaultObs(zero_padding=3,
                           pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 
                                              1 / common_values.BACK_NET_Y, 
                                              1 / common_values.CEILING_Z]),
                           ang_coef=1 / np.pi,
                           lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
                           ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
                           boost_coef=1 / 100.0)

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=0),
        KickoffMutator()
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RocketSimVisRenderer(),
    )

    # extract the lookup table from the lookup parser instance
    try:
        if hasattr(LookupTableAction(), "lookup_table"):
            table = LookupTableAction().lookup_table
        elif hasattr(LookupTableAction(), "table"):
            table = LookupTableAction().table
        else:
            table = None
    except Exception:
        table = None

    rlgym_env.reset()
    wrapped = RLGymV2GymWrapper(rlgym_env)

    return wrapped, rlgym_env, table
