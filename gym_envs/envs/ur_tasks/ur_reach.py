# MODEL: Univewrsal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 04/10/2022
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

import numpy as np

from gym_envs.envs.core import RobotTaskEnv
from gym_envs.envs.robots.ur import UR3
from gym_envs.envs.tasks.reach import Reach
from gym_envs.mujoco_func import Mujoco_Func

class URReachEnv(RobotTaskEnv):
    def __init__(
        self, 
        render: bool = False, 
        reward_type: str = "sparse", 
        control_type: str = "ee",
        ) -> None:
        
        sim = Mujoco_Func(render=render)
        robot = UR3(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Reach(sim, reward_type=reward_type, get_ee_position=robot.get_body_position)
        super().__init__(
            robot, 
            task,
            init_grasping=False,
            )