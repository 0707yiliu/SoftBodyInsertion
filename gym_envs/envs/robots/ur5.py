# MODEL: Universal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 04/10/2022
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

from array import array
from email.mime import base
from typing import Optional

import numpy as np
import math

from gym import spaces
from gym_envs.envs.core import MujocoRobot
# import mujoco_py as mj
# from gym_envs.mujoco_func import Mujoco_Func
from gym_envs.utils import distance, normalizeVector, euler_to_quaternion
from gym_envs.envs.impedance_controller import FT_controller
import time
from datetime import datetime

from ctypes import util
import signal
import socket
import serial
import struct
import sys
import time
import numpy as np
import copy
import argparse
import threading
import os
from scipy.spatial.transform import Rotation as R
import rtde_control
import rtde_receive
import math
import gym_envs.envs.robots.robotiq_gripper as robotiq_gripper

class UR(MujocoRobot):
    def __init__(
        self,
        sim,
        block_gripper: bool = False,
        base_position: Optional[np.array] = None,
        control_type: str = "ee",
        d2r: float = math.pi/180,
        planner_time: float = 1, # planner used time: union(s)
        ee_positon_low: Optional[np.array] = None,
        ee_positon_high: Optional[np.array] = None,
        gripper_joint_low: Optional[float] = None,
        gripper_joint_high: Optional[float] = None,
        ee_dis_ratio: float = 0.05,
        ee_rot_ratio: float = 0.1,
        joint_dis_ratio: float = 0.003,
        gripper_action_ratio: float = 0.001,
        gripper_max_joint: float = 0.67,
        vision_touch: str = "vision",
        _normalize: bool = False,
        ee_init: Optional[np.array] = np.array([-0.3, 0.31, 0.89]),
        z_offset: float = 0.16,
        match_shape: bool = True,
        real_robot: bool = False,
        admittance_control: bool = True,
        dsl: bool = False,
        ee_rot_enable: bool = True,
    ) -> None:
        self.sim_ft_threshold = 50
        self.admittance_control = admittance_control
        self.real_robot = real_robot
        self.z_press = False
        self._init = False
        self.dsl = dsl
        self.matching_shape = match_shape # add z-axis rotation
        self.ee_init_pos = ee_init
        self.ee_init_pos[2] += z_offset
        self.init_ft_sensor = np.array([0.34328045,  2.12207036, -9.31005002,  0.27880092,  0.04346038,  0.01519237])
        self.ft_sensor_z = 0.0
        # self.ee_pos = ee_init
        # self.ee_pos[2] += z_offset
        self._normalize_obs = _normalize
        self.gripper_max_joint = gripper_max_joint
        self.vision_touch = vision_touch
        self.ee_rot_ratio = ee_rot_ratio
        self.ee_dis_ratio = ee_dis_ratio / 10 # sim time = 0.001s, so the ee maximum movement is ee_dis_ratio(/m) between 1s.
        self.gripper_action_ratio = gripper_action_ratio/40 if self.vision_touch =='vision' else gripper_action_ratio/20
        self.random_high = 0.045 if self.vision_touch == 'vision' else 0.045
        self.random_low = -0.045 if self.vision_touch == 'vision' else -0.045 # 0.014
        self.random_lim_high = 0.025 if self.vision_touch == 'vision' else 0.025
        self.random_lim_low = -0.025 if self.vision_touch == 'vision' else -0.025
        self._z_up = 0.01
        self._z_down = 0.002
        # self._z_up = 0
        # self.random_high = 0.014 if self.vision_touch == 'vision' else 0.014
        # self.random_low = -0.014 if self.vision_touch == 'vision' else -0.014
        # self.random_lim_high = 0.004 if self.vision_touch == 'vision' else 0.004
        # self.random_lim_low = -0.004 if self.vision_touch == 'vision' else -0.004
        self.joint_dis_ratio = joint_dis_ratio
        self.ee_rot_low = np.array([-180, -56, 12])
        self.ee_rot_high = np.array([180, -116, 72])
        self.ee_position_low = ee_positon_low if ee_positon_low is not None else np.array([-0.25, 0.1, 0.8])
        self.ee_position_high = ee_positon_high if ee_positon_high is not None else np.array([0.25, 1, 1.6])
        self.gripper_joint_low = gripper_joint_low if gripper_joint_low is not None else 0.3
        self.gripper_joint_high = gripper_joint_high if gripper_joint_high is not None else 0.47
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.planner_time = planner_time
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 6 # ur-ee control (x,y) / ur-joint control 
        # n_action += 0 if self.block_gripper else 1 # replace by self.vision_touch
        n_action += 3 if ee_rot_enable is True else 0
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        print("action space:",action_space)
        norm_max = 1
        norm_min = -1
        self.ee_scale = self.ee_position_high - self.ee_position_low
        self.norm_scale = (norm_max - norm_min) / self.ee_scale
        self.ee_mean = (self.ee_position_high + self.ee_position_low) / 2
        self.norm_mean = (norm_max + norm_min) / 2 * np.array([1, 1, 1])
        # ------------ for real robot ------------------
        j_acc = 0.03
        j_vel = 0.015
        HOST = "10.42.0.163"
        PORT = 30002
        super().__init__(
            sim,
            action_space=action_space,
            joint_index=np.array([0, 1, 2, 3, 4, 5, 6]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 170.0]),
            joint_list=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            sensor_list=["touchsensor_r1", "touchsensor_r2", "touchsensor_r3", "touchsensor_r4", "touchsensor_r5", "touchsensor_r6", 
                        "touchsensor_l1", "touchsensor_l2", "touchsensor_l3", "touchsensor_l4", "touchsensor_l5", "touchsensor_l6"],
        )
        #!TODO: to be uniformed --------------
        self.d2r = d2r
        self.sensor_num = len(self.sensor_list)
        self.fingers_index = np.array([6])
        # self.neutral_joint_values = np.array((90, -90, 90, -90, -90, 0, 0))*d2r
        self.neutral_joint_values = np.array([1.53094203, -1.50108324,  1.75714794, -1.78607852, -1.52985284,  0])
        self.ee_body = "eef"
        self.finger1 = "right_driver_1"
        self.finger2 = "left_driver_1"
        if self.matching_shape is True:
            self.init_z_rot = 0.0
            self.z_rot_high = 160.0
            self.z_rot_low = -160.0
            # print(n_action)
        #! -----------------------------------
    def log_info(self):
        print(f"Pos: {str(self.gripper.get_current_position()): >3}  "
            f"Open: {self.gripper.is_open(): <2}  "
            f"Closed: {self.gripper.is_closed(): <2}  ")

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()
        # print(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # ----------- ee / joint controlling type -------------
        if self.control_type == "ee":
            ee_displacement = np.copy(action) # 3-xyz + 3-rot = 6-dof
            # print(ee_displacement)
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
            current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
            if np.isnan(target_arm_angles).any() is True or np.isfinite(target_arm_angles).all() is False:
                target_arm_angles = np.copy(current_joint_arm)
            # elif (np.absolute(target_arm_angles - current_joint_arm) > 0.5).any():
            #     target_arm_angles = np.copy(current_joint_arm)
        else:
            arm_joint_ctrl = action[:7] # 6 arm joints + 1 gripper joints
            target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)
        # ---------------------------------------------------------
        current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        current_joint_gripper = np.array(self.get_fingers_width()/2)
        # current_joint_gripper = np.clip(current_joint_gripper, self.gripper_joint_low, self.gripper_joint_high)
        current_joint = np.concatenate((current_joint_arm, [current_joint_gripper]))
        self.joint_planner(grasping=True, _time=self.planner_time/50, 
                            current_joint=current_joint_arm,
                            target_joint=target_arm_angles)
       
    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        xyz = 3
        ee_displacement[:xyz] = ee_displacement[:xyz] * self.ee_dis_ratio
        ee_displacement[xyz:] = ee_displacement[xyz:] * self.ee_rot_ratio
        # print(ee_displacement)
        # target_ee_position = np.clip(ee_displacement[:xyz], self.ee_position_low, self.ee_position_high)
        current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        current_ee_pos = self.sim.get_site_position('obj_bottom')
        current_ee_rot = R.from_matrix(self.sim.get_site_mat('obj_bottom').reshape(3, 3)).as_euler('xyz', degrees=True)
        target_ee_pos = current_ee_pos + ee_displacement[:xyz]
        target_ee_rot = current_ee_rot + ee_displacement[xyz:]
        target_ee_rot = np.clip(target_ee_rot, self.ee_rot_low, self.ee_rot_high)
        # print(current_ee_rot)
        target_ee_pos = np.clip(target_ee_pos, self.ee_position_low, self.ee_position_high)
        target_ee_quat = R.from_euler('xyz', target_ee_rot, degrees=True).as_quat()
        target_arm_angles = self.inverse_kinematics(
                current_joint=current_joint, 
                target_position=target_ee_pos, 
                target_orientation=target_ee_quat
            )
        # print(target_arm_angles)
        return target_arm_angles
    
    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        arm_joint_ctrl = arm_joint_ctrl * self.joint_dis_ratio
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_fingers_width(self) -> float:
        finger1 = self.sim.get_joint_angle(self.finger1)
        finger2 = self.sim.get_joint_angle(self.finger2)
        return finger1 + finger2

    def get_obs(self) -> np.ndarray:
        # ee_position = np.copy(self.get_ee_position())
        ee_position = self.sim.get_body_position('base_mount')
        ee_rot = R.from_quat(np.roll(self.sim.get_body_quaternion('base_mount'), -1)).as_euler('xyz')
        # ee_position = (ee_position - self.ee_mean) * self.norm_scale + self.norm_mean
        # print("sim-ee-position:", ee_position)
        # ee_velocity = np.array(self.get_ee_velocity())
        # joint_angles = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        # print(joint_angles)
        # joint_vels = np.array([self.sim.get_joint_velocity(joint=self.joint_list[i]) for i in range(6)])
        ft_sensor = self.sim.get_ft_sensor(force_site="ee_force_sensor", torque_site="ee_torque_sensor")
        obs = np.concatenate((ee_position, ee_rot))
        return obs

    def reset(self) -> None:
        self.sim.reset()
        self.init_z_rot = 0.0
        self.set_joint_neutral()

    def log_info(self, gripper):
        print(f"Pos: {str(gripper.get_current_position()): >3}  "
            f"Open: {gripper.is_open(): <2}  "
            f"Closed: {gripper.is_closed(): <2}  ")

    def set_joint_neutral(self) -> None:
        self.set_joint_angles(self.neutral_joint_values)

    def joint_planner(self, grasping: bool, _time: float, current_joint: np.ndarray, target_joint: np.ndarray) -> None:
        delta_joint = target_joint - current_joint
        planner_steps = int(_time / self.sim.timestep)
        if grasping is True:
            finger_joint = np.array([200])
            target_full_joint = np.concatenate((target_joint, finger_joint))
            self.control_joints(target_full_joint)
            self.sim.step()
            # for i in range(planner_steps):
            #     planned_delta_joint = delta_joint * math.sin((math.pi/2)*(i/planner_steps))
            #     target_joint = planned_delta_joint + current_joint
            #     target_full_joint = np.concatenate((target_joint, finger_joint))
            #     self.control_joints(target_full_joint)
            #     self.sim.step()
        else:
            for i in range(planner_steps):
                planned_delta_joint = delta_joint * math.sin((math.pi/2)*(i/planner_steps))
                target_joint = planned_delta_joint + current_joint
                self.control_joints(target_joint)
                self.sim.step()