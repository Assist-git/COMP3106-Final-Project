from typing import List, Dict, Any
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
import numpy as np

# reward when car moves toward ball
class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents, initial_state, shared_info):
        return

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        # compute reward based on cars speed toward ball
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # car physics based on team (from RLGym)
            car_phys = car.physics if car.is_orange else car.inverted_physics
            ball_phys = state.ball if car.is_orange else state.inverted_ball

            # vector from car to ball
            pos_diff = ball_phys.position - car_phys.position
            dist = float(np.linalg.norm(pos_diff))
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue

            # direction unit vector toward ball
            dir_to_ball = pos_diff / dist
            # project car velocity onto direction toward ball
            speed_toward_ball = float(np.dot(car_phys.linear_velocity, dir_to_ball))
            
            # how good the car is aligned to ball
            alignment = float(np.dot(car_phys.forward, dir_to_ball))
            
            shaped = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)
            
            # apply distance-based scaling: reward faster movement when close to ball
            if dist > 500:  # far from ball: lower reward
                base_reward = shaped * 0.3
            else:  # close to ball: higher reward
                base_reward = shaped * 0.2
            
            # apply misalignment penalty: penalize moving toward ball while facing wrong direction
            if speed_toward_ball > 0 and alignment < 0.7:  # moving toward but misaligned
                misalignment_penalty = (1.0 - alignment) * 0.15
                rewards[agent] = base_reward - misalignment_penalty
            else:
                rewards[agent] = base_reward
                
        return rewards

# TouchReward taken from RLGym starter code
# is made for multiple agents but works fine for single agent
class TouchReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self):
        # track previous touch count for each agent to detect new touches
        self._prev_touches = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        # initializes previous touch counts at ep start
        self._prev_touches.clear()
        for agent in agents:
            self._prev_touches[agent] = int(initial_state.cars[agent].ball_touches)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated, is_truncated, shared_info):
        # 1.0 reward for each new touch
        rewards = {}
        for agent in agents:
            prev = self._prev_touches.get(agent, 0)
            cur = int(state.cars[agent].ball_touches)
            delta = cur - prev
            rewards[agent] = 1.0 if delta > 0 else 0.0
            self._prev_touches[agent] = cur
        return rewards

# reward when using boost when the car is aligned with ball
class BoostAlignmentReward(RewardFunction[AgentID, GameState, float]):

    def __init__(self, angle_threshold: float = 0.7,
                 reward_value: float = 0.1,
                 penalty_value: float = -0.05):
        self.angle_threshold = angle_threshold
        self.reward_value = reward_value
        self.penalty_value = penalty_value

    def reset(self, agents: List[AgentID], initial_state: GameState,
              shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState,
                    is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool],
                    shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        # computes rewward for all agents but only one agent implemented
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        ball = state.ball

        # vector from car to ball
        to_ball = ball.position - car.physics.position
        if np.linalg.norm(to_ball) == 0:
            return 0.0
        to_ball /= np.linalg.norm(to_ball)

        # car forward vector
        forward = car.physics.forward
        # dot product: 1.0 = perfect alignment, 0.0 = perpendicular, -1.0 = backward
        alignment = np.dot(forward, to_ball)

        if car.is_boosting and alignment > self.angle_threshold:
            # boosting while aligned toward ball = reward
            return self.reward_value
        elif car.is_boosting and alignment < 0.0:
            # boosting while facing away from ball = penalty
            return self.penalty_value
        return 0.0

# dense reward for orienting the car toward the ball
class FacingBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            # vector from car to ball
            to_ball = ball.position - car.physics.position
            if np.linalg.norm(to_ball) < 1e-6:
                rewards[agent] = 0.0
                continue
            to_ball /= np.linalg.norm(to_ball)
            # car forward direction
            forward = car.physics.forward
            # alignment: dot product in [-1, 1] range. Clip to [0, 1] to reward only forward-facing
            alignment = np.dot(forward, to_ball)
            rewards[agent] = max(0.0, alignment) * 0.05
        return rewards

# Reward when ball is pushed towards goal
class BallToGoalReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            
            # Determine goal position based on team
            if car.is_orange: # only this works since only 1 agent
                goal_y = -common_values.BACK_NET_Y
            else:
                goal_y = common_values.BACK_NET_Y
            
            # vector from ball to goal
            ball_to_goal = np.array([0, goal_y, 0]) - ball.position
            dist_to_goal = float(np.linalg.norm(ball_to_goal))
            
            if dist_to_goal < 1e-6:
                rewards[agent] = 0.0
                continue
            
            # distance reward the closer it is the higher reward is
            goal_dist_normalized = 1.0 / (1.0 + dist_to_goal / 1000.0)
            rewards[agent] = goal_dist_normalized * 0.05
        
        return rewards

# reward for steering towards ball
class SteeringTowardBallReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, coef: float = 2.0, clip: float = 0.5, min_dist: float = 3.0):
        self.coef = float(coef)
        self.clip = float(clip)
        self.min_dist = float(min_dist)
        # track previous angle for each agent to compute angular delta
        self._prev_angle = {}

    def reset(self, agents, initial_state, shared_info):
        self._prev_angle.clear()
        for agent in agents:
            car = initial_state.cars[agent]
            to_ball = initial_state.ball.position - car.physics.position
            dist = float(np.linalg.norm(to_ball))
            if dist < 1e-6:
                self._prev_angle[agent] = 0.0
            else:
                to_ball /= (dist + 1e-8)
                # compute angle between car forward and ball direction
                dot = float(np.clip(np.dot(car.physics.forward, to_ball), -1.0, 1.0))
                self._prev_angle[agent] = float(np.arccos(dot))

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            to_ball = state.ball.position - car.physics.position
            dist = float(np.linalg.norm(to_ball))
            
            # skip reward computation if too close to ball
            if dist < self.min_dist:
                if dist < 1e-6:
                    angle_now = 0.0
                else:
                    to_ball /= (dist + 1e-8)
                    dot = float(np.clip(np.dot(car.physics.forward, to_ball), -1.0, 1.0))
                    angle_now = float(np.arccos(dot))
                self._prev_angle[agent] = angle_now
                rewards[agent] = 0.0
                continue

            # normalize direction to ball
            to_ball /= (dist + 1e-8)
            # compute current angle between car forward and ball direction
            dot = float(np.clip(np.dot(car.physics.forward, to_ball), -1.0, 1.0))
            angle_now = float(np.arccos(dot))
            
            # get previous angle and compute change
            prev = self._prev_angle.get(agent, angle_now)
            angle_delta = prev - angle_now  # positive if angle decreased
            self._prev_angle[agent] = angle_now

            # scale angle delta by coefficient
            shaped = self.coef * angle_delta
            # clip to prevent extreme rewards
            if self.clip is not None:
                shaped = float(np.clip(shaped, -self.clip, self.clip))
            rewards[agent] = shaped
        return rewards