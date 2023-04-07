# MODEL: Univewrsal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 04/10/2022
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering
from typing import Any, Dict, Union

import numpy as np

from gym_envs.envs.core import Task
from gym_envs.utils import distance

class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range_low=np.array([-0.1, 0.3, 0.885]),
        goal_range_high=np.array([0.1, 0.38, 0.955]),
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = goal_range_low
        self.goal_range_high = goal_range_high
        # self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        # self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])

    def get_obs(self) -> np.ndarray:
        return np.array([]) 
    
    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position("eef"))
        # print("eef:", ee_position)
        return ee_position
    
    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_mocap_pos(mocap="box", pos=self.goal)
    
    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal
    
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)
    
    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d