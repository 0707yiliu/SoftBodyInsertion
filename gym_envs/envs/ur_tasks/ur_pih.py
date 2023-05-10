# MODEL: Universal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 04/10/2022
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

import numpy as np

from gym_envs.envs.core import RobotTaskEnv
from gym_envs.envs.robots.ur5 import UR
from gym_envs.envs.tasks.pih import PeginHole
from gym_envs.mujoco_func2 import Mujoco_Func
# from gym_envs.envs.apriltagDetection import AprilTag

class URPeginHoleEnv(RobotTaskEnv):
    def __init__(
        self, 
        render: bool = False, 
        reward_type: str = "sparse", 
        control_type: str = "ee",
        vision_touch: str = "vision",
        normalizeObs: bool = False,
        hole_size: str = '2cm',
        match_shape: bool = True,
        real_robot: bool = False,
        dsl: bool = False,
        domain_randomization: bool = False,
        ) -> None:
        # print("--------")
        sim = Mujoco_Func(
            render=render,
            vision_touch=vision_touch,
            hole_size=hole_size,
            real_robot=real_robot,
            domain_randomization=domain_randomization,
            )
        robot = UR(
            sim=sim,
            block_gripper=True, 
            base_position=np.array([-0.6, 0.0, 0.0]), 
            control_type=control_type, 
            vision_touch=vision_touch,
            _normalize=normalizeObs,
            match_shape=match_shape,
            real_robot=real_robot,
            dsl=dsl,
            )
        task = PeginHole(
            sim=sim,
            reward_type=reward_type, 
            get_ee_position=robot.get_body_position, 
            vision_touch=vision_touch,
            _normalize=normalizeObs,
            real_robot=real_robot,
            )
        super().__init__(
            robot, 
            task,
            init_grasping=True,
            render=render,
            )

