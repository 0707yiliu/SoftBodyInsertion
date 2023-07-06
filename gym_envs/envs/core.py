# MODEL: Universal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 04/10/2022
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import gym
import gym.spaces
from gym import utils
from gym import spaces
# from gym.envs.mujoco import mujoco_env
import gym.utils.seeding
import gym_robotics
import numpy as np

# import mujoco_py as mj
# from gym_envs.mujoco_func2 import Mujoco_Func
# from ur3_kdl_func import inverse

class MujocoRobot(ABC):
    def __init__(
        self,
        sim,
        action_space: gym.spaces.Space,
        joint_index: np.ndarray,
        joint_forces: np.ndarray,
        joint_list: list,
        sensor_list: list,
    ) -> None:
        self.sim = sim
        self.setup()
        self.action_space = action_space
        self.joint_index = joint_index
        self.joint_forces = joint_forces
        self.joint_list = joint_list
        self.sensor_list = sensor_list
    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be called just before sim.step().

        Args:
            action (np.ndarray): The action.
        """
    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: The observation.
        """
    @abstractmethod
    def reset(self) -> None:
        """Reset the robot and return the observation."""
    
    def setup(self) -> None:
        """Called after robot loading."""
        pass

    def set_forwad(self) -> None:
        self.sim.set_forward()

    def get_body_position(self, body: str) -> np.ndarray:
        return self.sim.get_body_position(body=body)

    def get_body_velocity(self, body: str) -> np.ndarray:
        return self.sim.get_body_velocity(body=body)
    
    def get_touch_sensor(self, sensor: str) -> np.ndarray:
        return self.sim.get_touch_sensor(sensor=sensor)
    
    def get_joint_angle(self, joint: str) -> float:
        return self.sim.get_joint_angle(joint=joint)
    
    def get_joint_velocity(self, joint: str) -> float:
        return self.sim.get_joint_velocity(joint=joint)
    
    def control_joints(self, target_angles: np.ndarray) -> None:
        self.sim.control_joints(target_angles=target_angles)
    
    def set_joint_angles(self, angles: np.ndarray) -> None:
        self.sim.set_joint_angles(angles=angles)

    def inverse_kinematics(self, current_joint: np.ndarray, target_position: np.ndarray, target_orientation: np.ndarray) -> np.ndarray:
        inverse_kinematics = self.sim.inverse_kinematics(
            current_joint, 
            target_position, 
            target_orientation)
        return inverse_kinematics
    
    def forward_kinematics(self, qpos) -> np.ndarray:
        f_pos = self.sim.forward_kinematics(qpos=qpos)
        return f_pos

class Task(ABC):
    def __init__(
        self, 
        sim,
        ) -> None:
        self.sim = sim
        self.goal = None

    @abstractmethod
    def reset(self) -> None:
        """Reset the task: sample a new goal."""

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    @abstractmethod
    def is_success(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        """Returns whether the achieved goal match the desired goal."""

    @abstractmethod
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        """Compute reward associated to the achieved and the desired goal."""

class RobotTaskEnv(gym_robotics.GoalEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    def __init__(
        self, 
        robot: MujocoRobot, 
        task: Task,
        init_grasping: bool,
        render: bool,
        num_goal: int = 1,
        ur_gen: int = 5,
        ) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.num_goal = num_goal
        self._num_goal = 0
        self.sim = robot.sim
        self.robot = robot
        self.task = task
        self.init_grasping = init_grasping
        self.render = render
        # self._reset_first = False
        self.one_episode = 0
        obs = self._get_obs()  # required for init; seed can be changed later
        # self._reset_first = True
        # observation_shape = obs["observation"].shape
        # achieved_goal_shape = obs["achieved_goal"].shape
        # desired_goal_shape = obs["achieved_goal"].shape
        # self.observation_space = gym.spaces.Dict(
        #     dict(
        #         observation=gym.spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
        #         desired_goal=gym.spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
        #         achieved_goal=gym.spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
        #     )
        # )
        
        observation_shape = obs.shape
        # self.obs_record = [np.zeros(observation_shape)]
        # print("observation shape:", observation_shape)
        # print(self.obs_record)
        # print(observation_shape)
        self.observation_space = spaces.Box(-1, 1.0, shape=observation_shape, dtype=np.float32)
        print("obs space:",self.observation_space)
        self.action_space = self.robot.action_space

        # print(self.observation_space, self.action_space)
        self.compute_reward = self.task.compute_reward
        # self._saved_goal = dict()
        # print("over here")


    def reset(self, seed: Optional[int] = None, options={}) -> tuple:
        # self.task.np_random, seed = gym.utils.seeding.np_random(seed)
        # with self.sim.no_rendering():
        #     self.robot.reset()
        #     self.task.reset()
        self.one_episode = 0
        self.robot.reset()
        self.task.reset()
        self.sim.set_forward() # update env statement
        info = dict()
        # if self.init_grasping:
        #     self.robot._init_grasping()
        return self._get_obs(), info
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs()  # robot state
        task_obs = self.task.get_obs()  # object position, velococity, etc...
        # print("robot obs:",robot_obs)
        # print("task obs:",task_obs)
        observation = np.concatenate([robot_obs, task_obs])
        
        # if self._reset_first is True:
        #     self.obs_record = np.r_[self.obs_record, [observation]]
            # print(self.obs_record)

        achieved_goal = self.task.get_achieved_goal()
        # return {
        #     "observation": observation,
        #     "achieved_goal": achieved_goal,
        #     "desired_goal": self.task.get_goal(),
        # }
        return observation

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        # self.sim.step()
        obs = self._get_obs()
        done = False
        # info = {"is_success": self.task.is_success(obs["achieved_goal"], self.task.get_goal())}
        # reward = self.task.compute_reward(obs["achieved_goal"], self.task.get_goal(), info)
        info = {"is_success": self.task.is_success(self.task.get_achieved_goal(), self.task.get_goal())}
        reward = self.task.compute_reward(self.task.get_achieved_goal(), self.task.get_goal(), info) # TODO: designed for only tactile sensor mode.
        assert isinstance(reward, float)  # needed for pytype cheking

        # print(reward)
        # if reward > -0.01:
        #     box_ee = self.task.get_achieved_goal()
        #     hole_center = self.task.get_goal()
        #     box_hole_disx = abs(box_ee[0] - hole_center[0])
        #     box_hole_disy = abs(box_ee[1] - hole_center[1])
        #     box_hole_dis = box_hole_disx + box_hole_disy
        #     box_dis_z = abs(box_ee[2] - hole_center[2])
        #     # print(box_hole_dis)
        #     if box_hole_dis < 0.006 and box_dis_z < 0.005:
        #         self._num_goal += 1
        #         if self._num_goal >= self.num_goal:
        #             done = True
        #             self._num_goal = 0
        #             # print('done is Ture-------------------------------------')
        #         else:
        #             done = False
        #     else:
        #         self._num_goal = 0
        #         done = False
        self.one_episode += 1
        # print(self.one_episode)
        if self.render is False:
            if self.task.is_success(self.task.get_achieved_goal(), self.task.get_goal()):
                self._num_goal += 1
                if self._num_goal >= self.num_goal:
                    done = True
                    self._num_goal = 0
                else:
                    done = False
            else:
                self._num_goal = 0
                done = False
        else:
            if self.task.is_success(self.task.get_achieved_goal(), self.task.get_goal()):
                self._num_goal += 1
                if self._num_goal >= self.num_goal:
                    done = False
                    print("success")
                    self._num_goal = 0
                else:
                    done = False
            else:
                self._num_goal = 0
                done = False
            if self.one_episode >= 400:
                done = True
        truncated = False
        return obs, reward, done, truncated, info
    
