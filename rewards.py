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
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            car_phys = car.physics if car.is_orange else car.inverted_physics
            ball_phys = state.ball if car.is_orange else state.inverted_ball

            pos_diff = ball_phys.position - car_phys.position
            dist = float(np.linalg.norm(pos_diff))
            if dist < 1e-6:
                rewards[agent] = 0.0
                continue

            dir_to_ball = pos_diff / dist
            speed_toward_ball = float(np.dot(car_phys.linear_velocity, dir_to_ball))
            
            # reward speed toward ball
            alignment = float(np.dot(car_phys.forward, dir_to_ball))
            
            shaped = max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0)
            
            # distance based scaling
            if dist > 500:  # far from ball
                base_reward = shaped * 0.3
            else:  # close to ball
                base_reward = shaped * 0.2
            
            # penalize if circling around ball
            if speed_toward_ball > 0 and alignment < 0.7:  # moving toward ball but misaligned
                misalignment_penalty = (1.0 - alignment) * 0.15
                rewards[agent] = base_reward - misalignment_penalty
            else:
                rewards[agent] = base_reward
                
        return rewards

# reward when ball is touched
class TouchReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self):
        self._prev_touches = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._prev_touches.clear()
        for agent in agents:
            self._prev_touches[agent] = int(initial_state.cars[agent].ball_touches)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated, is_truncated, shared_info):
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
        alignment = np.dot(forward, to_ball)

        if car.is_boosting and alignment > self.angle_threshold:
            return self.reward_value
        elif car.is_boosting and alignment < 0.0:
            return self.penalty_value
        return 0.0


# reward when car is facing ball
class FacingBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            ball = state.ball
            to_ball = ball.position - car.physics.position
            if np.linalg.norm(to_ball) < 1e-6:
                rewards[agent] = 0.0
                continue
            to_ball /= np.linalg.norm(to_ball)
            forward = car.physics.forward
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
            if car.is_orange:
                goal_y = -common_values.BACK_NET_Y
            else:
                goal_y = common_values.BACK_NET_Y
            
            ball_to_goal = np.array([0, goal_y, 0]) - ball.position
            dist_to_goal = float(np.linalg.norm(ball_to_goal))
            
            if dist_to_goal < 1e-6:
                rewards[agent] = 0.0
                continue
            
            # distance reward the closer it is the higher reward is
            goal_dist_normalized = 1.0 / (1.0 + dist_to_goal / 1000.0)
            rewards[agent] = goal_dist_normalized * 0.05
        
        return rewards

class SteeringTowardBallReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, coef: float = 2.0, clip: float = 0.5, min_dist: float = 3.0):
        self.coef = float(coef)
        self.clip = float(clip)
        self.min_dist = float(min_dist)
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
                dot = float(np.clip(np.dot(car.physics.forward, to_ball), -1.0, 1.0))
                self._prev_angle[agent] = float(np.arccos(dot))

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            to_ball = state.ball.position - car.physics.position
            dist = float(np.linalg.norm(to_ball))
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

            to_ball /= (dist + 1e-8)
            dot = float(np.clip(np.dot(car.physics.forward, to_ball), -1.0, 1.0))
            angle_now = float(np.arccos(dot))
            prev = self._prev_angle.get(agent, angle_now)
            angle_delta = prev - angle_now
            self._prev_angle[agent] = angle_now

            shaped = self.coef * angle_delta
            if self.clip is not None:
                shaped = float(np.clip(shaped, -self.clip, self.clip))
            rewards[agent] = shaped
        return rewards