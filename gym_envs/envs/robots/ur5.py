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

import rtde_control
import rtde_receive
import math
import gym_envs.envs.robots.robotiq_gripper as robotiq_gripper

# this func defines the action which contains x and y without z, z would be used by force detection.
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
        self.ee_dis_ratio = ee_dis_ratio/60 if self.vision_touch == 'vision' or 'vision-touch' else ee_dis_ratio/40
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
        self.ee_position_low = ee_positon_low if ee_positon_low is not None else np.array([-0.2, 0.25, 0.8])
        self.ee_position_high = ee_positon_high if ee_positon_high is not None else np.array([0.05, 0.40, 1.6])
        self.gripper_joint_low = gripper_joint_low if gripper_joint_low is not None else 0.3
        self.gripper_joint_high = gripper_joint_high if gripper_joint_high is not None else 0.47
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.planner_time = planner_time
        self.control_type = control_type
        n_action = 2 if self.control_type == "ee" else 6 # ur-ee control (x,y) / ur-joint control 
        # n_action += 0 if self.block_gripper else 1 # replace by self.vision_touch
        n_action += 0 if self.vision_touch == 'vision' else 0
        if self.vision_touch == "vision" and self.dsl is True:
            n_action += 1
        if self.dsl is False:
            n_action = 3
        n_action += 1 if self.matching_shape is True else 0
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
        if self.real_robot is True:
            # self.tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # self.tcp_client.settimeout(10)
            # # self.tcp_client.connect((HOST, PORT))
            # self.back_log = 5
            # self.buffer_size = 1060
            self._reset = False
            self.z_press_to_hole = False
            self.z_press_suc_items = 0
            self.z_low_fornodsl = -1
            # rtde-process
            self.rtde_r = rtde_receive.RTDEReceiveInterface(HOST)
            self.rtde_c = rtde_control.RTDEControlInterface(HOST)
            actual_q = np.array(self.rtde_r.getActualQ())
            print("current qpos:", actual_q)
            # self.j_init = np.array([1.37637351, -1.51406425,  1.25493969, -1.31167174, -1.57079633, -0.10715635])
            self.j_init = np.array([1.31423293, -1.55386663,  1.32749743, -1.34477619, -1.57079633, -0.10715635]) 
            print(self.j_init)
            self.vel = 0.3
            self.acc = 0.3
            self.z_rot_offset = 0
            # gripper setting-----------
            print("Creating gripper...")
            self.gripper = robotiq_gripper.RobotiqGripper()
            print("Connecting to gripper...")
            self.gripper.connect(HOST, 63352)
            print("Activating gripper...")
            self.gripper.activate()
            self.ft_threashold = 10
            self.z_up_offset = 0.002
            self.z_down_offset = 0.003
            self._init_xy = np.array([-0.05, 0.32])
            _items = 500
            ft_record = np.zeros((_items, 6))
            for i in range(_items):
                ft_record[i] = self.rtde_r.getActualTCPForce()
                time.sleep(0.01)
            self.real_init_ft_sensor = np.mean(ft_record, axis=0)
        if self.admittance_control is True:
            # self.admittance_controller = FT_controller(0.008, 0.05, 0, 1/400) # admittance control with current state (hard to compliance)
            self.admittance_controller = FT_controller(1, 500, 1200, 1/500) # admittance control with compliance state (testing...)
            # self.admittance_controller = FT_controller(50, 100, 200, 1/10) # for sim
            # self.admittance_controller = FT_controller(2000, 1, 5, 1/20) # for real with 0.2s
            self.admittance_params = np.zeros((3, 3)) # contains acc, vel and pos in xyz derictions
            self.admittance_paramsT = np.zeros((3, 3))
            # --------------------------
        # ----------------------------------------------
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
        self.neutral_joint_values = np.array((90, -90, 90, -90, -90, 0, 0))*d2r
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
        self.z_press = True
        action = action.copy()
        # print(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.real_robot is False:
            # ----------- ee / joint controlling type -------------
            if self.control_type == "ee":
                ee_displacement = np.copy(action)
                # print(ee_displacement)
                target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
                current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
                if np.isnan(target_arm_angles).any() is True or np.isfinite(target_arm_angles).all() is False:
                    target_arm_angles = np.copy(current_joint_arm)
                # elif (np.absolute(target_arm_angles - current_joint_arm) > 0.5).any():
                #     target_arm_angles = np.copy(current_joint_arm)
            else:
                arm_joint_ctrl = action[:7]
                target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)
            # ---------------------------------------------------------
            # -------------- sim / real robot ---------------------------
        
            if self.vision_touch == 'vision':
                target_fingers_width = 0.1
            else:
                # fingers_ctrl = action[-1] * self.gripper_action_ratio
                fingers_ctrl = 0.1
                fingers_width = self.get_fingers_width()
                target_fingers_width = fingers_width / 2 + fingers_ctrl
                target_fingers_width = np.clip(target_fingers_width, self.gripper_joint_low, self.gripper_joint_high)
            target_angles = np.concatenate((target_arm_angles, [target_fingers_width]))
            # ------------------- record the first contact z position ------------------------
            # if self.dsl is True and self.vision_touch != "vision":
            if self.vision_touch != "vision":
                _items = 0
                while self._init is True:
                    current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
                    current_joint_gripper = np.array(self.get_fingers_width()/2)
                    # current_joint_gripper = np.clip(current_joint_gripper, self.gripper_joint_low, self.gripper_joint_high)
                    current_joint = np.concatenate((current_joint_arm, [current_joint_gripper]))
                    self.joint_planner(grasping=True, _time=self.planner_time/50, current_joint=current_joint_arm, target_joint=target_arm_angles)
                    ft_sensor = self.sim.get_ft_sensor(force_site='ee_force_sensor', torque_site='ee_torque_sensor').copy()
                    ft_sensor_current = self.init_ft_sensor - ft_sensor
                    for i in range(6):
                        if ft_sensor_current[i] >= self.sim_ft_threshold:
                            ft_sensor_current[i] = self.sim_ft_threshold
                        elif ft_sensor_current[i] <= -self.sim_ft_threshold:
                            ft_sensor_current[i] = -self.sim_ft_threshold
                        ft_sensor_current[i] = ft_sensor_current[i] / (self.sim_ft_threshold /2)
                    self.ft_sensor_z = np.copy(ft_sensor_current[2])
                    _items += 1
                    # print(self.sim.get_site_position("obj_bottom"), self.get_ee_position())
                    if _items > 5000:
                        print("cannot break the WHILE loop:", self.sim.get_body_position("box"), self.get_ee_position(), self.ft_sensor_z)
                        print(self.sim.get_site_position("obj_bottom")[-1] - self.sim.get_body_position("box")[-1])
                    if self.ft_sensor_z < -1:
                        self._init = False
                        self._z_contact_record = np.copy(self.get_ee_position()[-1]) + 0.01
                        # print(self._z_contact_record)
                    elif self.sim.get_site_position("obj_bottom")[-1] - self.sim.get_body_position("box")[-1] < 0:
                        self._z_contact_record = np.copy(self.sim.get_body_position("box")[-1]) + 0.03
                        self._init = False
                    else:
                        target_arm_angles = self.ee_displacement_to_target_arm_angles(np.array([0, 0, 0, 0]))
                        target_angles = np.concatenate((target_arm_angles, [target_fingers_width]))
                    # self._init = False
                    # self._z_contact_record = 1.063
            # -------------------------------------------------------------------------------
            current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
            current_joint_gripper = np.array(self.get_fingers_width()/2)
            # current_joint_gripper = np.clip(current_joint_gripper, self.gripper_joint_low, self.gripper_joint_high)
            current_joint = np.concatenate((current_joint_arm, [current_joint_gripper]))
            self.joint_planner(grasping=True, _time=self.planner_time/50, 
                                current_joint=current_joint_arm,
                                target_joint=target_arm_angles)
            # if self.vision_touch == 'vision':
            #     # self.control_joints(target_angles=target_angles)
            #     current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
            #     current_joint_gripper = np.array(self.get_fingers_width()/2)
            #     # current_joint_gripper = np.clip(current_joint_gripper, self.gripper_joint_low, self.gripper_joint_high)
            #     current_joint = np.concatenate((current_joint_arm, [current_joint_gripper]))
            #     self.joint_planner(grasping=False, _time=self.planner_time/50, current_joint=current_joint, target_joint=target_angles)
            # else:
            #     current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
            #     current_joint_gripper = np.array(self.get_fingers_width()/2)
            #     # current_joint_gripper = np.clip(current_joint_gripper, self.gripper_joint_low, self.gripper_joint_high)
            #     current_joint = np.concatenate((current_joint_arm, [current_joint_gripper]))
            #     self.joint_planner(grasping=False, _time=self.planner_time/50, current_joint=current_joint, target_joint=target_angles)
        else:
            # ----------- ee / joint controlling type -------------
            ee_displacement = np.copy(action)
            real_angles, target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
            # print("target angles:", target_arm_angles)
            # time.sleep(10)
            # ---------------------------------------------------------
            # ------------------- record the first contact z position ------------------------
            # if self.dsl is True and self.vision_touch != "vision":
            if self.vision_touch != "vision":
                _items = 0
                while self._init is True:
                    current_joint_arm = np.array(self.rtde_r.getActualQ())
                    for i in range(6):
                        while current_joint_arm[i] > 3.14:
                            current_joint_arm[i] -= 3.14
                        while current_joint_arm[i] < -3.14:
                            current_joint_arm[i] += 3.14
                    # print("current jpos:(type)", type(current_joint_arm), current_joint_arm)
                    forward_pos = self.forward_kinematics(qpos=current_joint_arm)
                    # print("forward_pos", forward_pos)
                    # print("target jpos:(type)", type(target_arm_angles), real_angles)
                    
                    self.real_joint_planner(grasping=True, 
                                        _time=self.planner_time/50, 
                                        current_joint=current_joint_arm, 
                                        target_joint=target_arm_angles)
                    time.sleep(0.1)
                    ft_sensor = np.array(self.rtde_r.getActualTCPForce())
                    ft_sensor_current = self.real_init_ft_sensor - ft_sensor
                    # print("ft_current:", ft_sensor_current)
                    # print("ft_init:", self.init_ft_sensor)
                    ft_threshold = self.ft_threashold
                    for i in range(6):
                        if ft_sensor_current[i] >= ft_threshold:
                            ft_sensor_current[i] = ft_threshold
                        elif ft_sensor_current[i] <= -ft_threshold:
                            ft_sensor_current[i] = -ft_threshold
                        ft_sensor_current[i] = ft_sensor_current[i] / (ft_threshold/2)
                    self.ft_sensor_z = np.copy(ft_sensor_current[2])
                    # print(self.init_ft_sensor, self.ft_sensor_z)
                    # _items += 1
                    # # print(self.sim.get_site_position("obj_bottom"), self.get_ee_position())
                    # if _items > 500:
                    #     print("cannot break the WHILE loop:", self.sim.get_body_position("box"), self.get_ee_position(), self.ft_sensor_z)
                    #     print(self.sim.get_site_position("obj_bottom")[-1] - self.sim.get_body_position("box")[-1])
                    if self.ft_sensor_z < -1:
                        self._init = False
                        # self._z_contact_record = np.copy(self.get_ee_position()[-1]) + 0.01
                        self._z_contact_record = np.copy(np.array(self.rtde_r.getActualTCPPose())[2]) + 0.86 + self.z_up_offset
                        self.z_low_fornodsl = np.copy(np.array(self.rtde_r.getActualTCPPose())[2]) - 0.008
                        # print(self.z_low_fornodsl)

                        # print("force get:",self._z_contact_record)
                    # elif self.sim.get_site_position("obj_bottom")[-1] - self.sim.get_body_position("box")[-1] < 0:
                    #     self._z_contact_record = 1.07
                    #     self._init = False
                    else:
                        _, target_arm_angles = self.ee_displacement_to_target_arm_angles(np.array([0, 0, 0, 0]))
                        # print("loop target arm angles:", target_arm_angles)
                        # target_angles = np.concatenate((target_arm_angles, [target_fingers_width]))
                    # self._init = False
                    # self._z_contact_record = 1.063


                    # z_lim_Test = np.array(self.rtde_r.getActualTCPPose())[2]
                    # print(z_lim_Test)
                    # if z_lim_Test < 0.20:
                    #     self._z_contact_record = z_lim_Test + 0.86
                        
                    #     self._init = False
            # -------------------------------------------------------------------------------
            current_joint_arm = np.array(self.rtde_r.getActualQ())
            # print("current jpos:(type)", type(current_joint_arm), current_joint_arm)
            # print("target jpos:(type)", type(target_arm_angles), target_arm_angles)
            self.real_joint_planner(grasping=True,
                                    _time=self.planner_time/50, 
                                    current_joint=current_joint_arm, 
                                    target_joint=target_arm_angles)
            if self._init is True:
                time.sleep(0.1)
            else:
                time.sleep(0.3)
                # if self.dsl is False:
                #     time.sleep(0.2)
                # else:
                #     

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        if self.real_robot is False:
            if self.dsl is False or self.vision_touch == "vision":
                xyz = 3
            else:
                xyz = 2

            ee_displacement[:xyz] = ee_displacement[:xyz] * self.ee_dis_ratio
            # print(ee_displacement)
            self.ee_init_pos[:xyz] = self.ee_init_pos[:xyz] + ee_displacement[:xyz] # used for orignal program
            # print(self.ft_sensor_z)
            # if self.dsl is True and self.vision_touch != "vision":
            if self.dsl is True and self.vision_touch != "vision":
                ft_sensor = self.sim.get_ft_sensor(force_site='ee_force_sensor', torque_site='ee_torque_sensor').copy()
                ft_sensor_current = self.init_ft_sensor - ft_sensor
                ft_threshold = self.sim_ft_threshold
                for i in range(6):
                    if ft_sensor_current[i] >= ft_threshold:
                        ft_sensor_current[i] = ft_threshold
                    elif ft_sensor_current[i] <= -ft_threshold:
                        ft_sensor_current[i] = -ft_threshold
                    ft_sensor_current[i] = ft_sensor_current[i] / (ft_threshold/2)
                self.ft_sensor_z = np.copy(ft_sensor_current[2])
                if self.ft_sensor_z < -1:
                    self.ee_init_pos[2] += self._z_up
                else:
                    self.ee_init_pos[2] -= self._z_down
                if self._init is False:
                    if self.z_press is True:
                        self.ee_init_pos[2] = np.copy(self._z_contact_record - 0.0135)
                        self.z_press = False
                    else:
                        self.ee_init_pos[2] = np.copy(self._z_contact_record)
                else:
                    ee_displacement[:2] = np.array([-0.05, 0.32])
                    self.ee_init_pos[:2] = np.array([-0.05, 0.32])
            if self.dsl is False and self.vision_touch != "vision" and self._init is True:
                ft_sensor = self.sim.get_ft_sensor(force_site='ee_force_sensor', torque_site='ee_torque_sensor').copy()
                ft_sensor_current = self.init_ft_sensor - ft_sensor
                ft_threshold = self.sim_ft_threshold
                for i in range(6):
                    if ft_sensor_current[i] >= ft_threshold:
                        ft_sensor_current[i] = ft_threshold
                    elif ft_sensor_current[i] <= -ft_threshold:
                        ft_sensor_current[i] = -ft_threshold
                    ft_sensor_current[i] = ft_sensor_current[i] / (ft_threshold/2)
                self.ft_sensor_z = np.copy(ft_sensor_current[2])
                if self.ft_sensor_z < -1:
                    self.ee_init_pos[2] += self._z_up
                else:
                    self.ee_init_pos[2] -= self._z_down
                if self._init is False:
                    if self.z_press is True:
                        self.ee_init_pos[2] = np.copy(self._z_contact_record - 0.02)
                        self.z_press = False
                    else:
                        self.ee_init_pos[2] = np.copy(self._z_contact_record)
                else:
                    ee_displacement[:2] = np.array([-0.05, 0.32])
                    self.ee_init_pos[:2] = np.array([-0.05, 0.32])
            # ---------- testing kinematic function --------------
            # self.ee_init_pos = self.ee_init_pos
            # target_ee_position = np.clip(self.ee_init_pos, self.ee_position_low, self.ee_position_high)
            # current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
            # -----------------------------------------------------
            # target_ee_position = np.array([-0.35, 0.3, 1]) # for testing pid controller
            # print(self.get_body_position("eef")) # for testing pid controller
            #! the constrain of ee position
            target_ee_position = np.clip(self.ee_init_pos, self.ee_position_low, self.ee_position_high) # use for orignal program
            # print("target ee pos:",target_ee_position)
            current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)]) # use for orignal program
            # print("current joint:", current_joint)
            # print("-----------------------:", current_joint)
            # print(current_joint)
            # print(ee_displacement)
            if self.matching_shape is True:
                ee_displacement[-1] = ee_displacement[-1] * 5
                # ------ change the eef joint (z-axis) --------
                # # self.init_z_rot = self.init_z_rot + ee_displacement[3]
                # self.init_z_rot = np.copy(ee_displacement[3]) 
                # target_z_rot = np.clip(self.init_z_rot, self.z_rot_low, self.z_rot_high)
                # target_z_rot = target_z_rot * self.d2r
                # target_quat = np.array([-1.0, 0.0, 0.0, 0.0])
                # target_arm_angles = self.inverse_kinematics(
                #     current_joint=current_joint, target_position=target_ee_position, target_orientation=target_quat
                # )
                # target_arm_angles[-1] = current_joint[-1] + target_z_rot
                # print(target_arm_angles)
                # -------- change eef orientation ---------
                # print(ee_displacement)
                self.init_z_rot = self.init_z_rot + ee_displacement[-1]
                # target_z_rot = np.copy(self.init_z_rot)
                target_z_rot = np.clip(self.init_z_rot, self.z_rot_low, self.z_rot_high)
                target_quat = euler_to_quaternion(-180 * self.d2r, 0, target_z_rot * self.d2r)
                desired_rotation = np.array([-180 * self.d2r, 0, target_z_rot * self.d2r])
                target_quat = np.roll(target_quat, 3)
                # print(current_joint, target_z_rot)
                # target_quat = np.array([-1.0, 0.0, 0.0, 0.0])
                # FT_data = np.array([-.1, -.1, -.1, -.1, -.1, -.1])
                # !admittance controller-----------
                # FT_data = -ft_sensor_current
                # # print(FT_data)
                # updated_pos, updated_rot, self.admittance_params, self.admittance_paramsT = self.admittance_controller.admittance_control(target_ee_position, 
                #                                                                                                                         desired_rotation, 
                #                                                                                                                         FT_data, 
                #                                                                                                                         self.admittance_params, 
                #                                                                                                                         self.admittance_paramsT)
                # target_ee_position = np.copy(updated_pos)
                # target_quat = euler_to_quaternion(updated_rot[0], updated_rot[1], updated_rot[2])
                # target_quat = np.roll(target_quat, 3)
                # # print(updated_pos, updated_rot)
                # -------------------
                target_arm_angles = self.inverse_kinematics(
                    current_joint=current_joint, 
                    target_position=target_ee_position, 
                    target_orientation=target_quat
                )
                # print("ik target angle:", target_arm_angles)
                # ---------- eef orientation testing -------------
                # self.init_z_rot = self.init_z_rot + ee_displacement[3]
                # target_z_rot = np.clip(self.init_z_rot, self.z_rot_low, self.z_rot_high)
                # target_quat = euler_to_quaternion(180 * self.d2r, 0, 30 * self.d2r)
                # target_quat = np.roll(target_quat, 3)
                # # target_quat = np.array([-1.0, 0.0, 0.0, 0.0])
                # # print(current_joint)
                # target_arm_angles = self.inverse_kinematics(
                #     current_joint=current_joint, target_position=target_ee_position, target_orientation=target_quat
                # )
                # # print(np.array(self.sim.get_body_quaternion(self.ee_body)))
            else:
                target_quat = np.array([-1.0, 0.0, 0.0, 0.0])
                target_arm_angles = self.inverse_kinematics(
                    current_joint=current_joint, 
                    target_position=target_ee_position, 
                    target_orientation=target_quat
                )
                
            # print('target_angles:', target_arm_angles)
            return target_arm_angles
        else:
            current_ee_pos = np.around(np.array(self.rtde_r.getActualTCPPose()), 4)
            # print(current_ee_pos[:3])
            if self.dsl is False or self.vision_touch == "vision":
                xyz = 3
                # ee_displacement[:2] = ee_displacement[:2] * self.ee_dis_ratio
                ee_displacement[0] = -ee_displacement[0] * self.ee_dis_ratio * 0.05
                ee_displacement[1] = -ee_displacement[1] * self.ee_dis_ratio * 1
                ee_displacement[2] = ee_displacement[2] * self.ee_dis_ratio
                # for admittance control ---------------
                if self.admittance_control is True:
                    self.real_ee_init_pos[:2] = self.real_ee_init_pos[:2] + ee_displacement[:2]
                    self.real_ee_init_pos[2] = self.real_ee_init_pos[2] + ee_displacement[2]
                    if self.dsl is False and self.vision_touch != "vision" and self._init is True:
                        _ft_sensor = np.array(self.rtde_r.getActualTCPForce())
                        ft_sensor_current = self.real_init_ft_sensor - _ft_sensor
                        ft_threshold = self.ft_threashold
                        for i in range(6):
                            if ft_sensor_current[i] >= ft_threshold:
                                ft_sensor_current[i] = ft_threshold
                            elif ft_sensor_current[i] <= -ft_threshold:
                                ft_sensor_current[i] = -ft_threshold
                            ft_sensor_current[i] = ft_sensor_current[i] / (ft_threshold/2)
                        self.ft_sensor_z = np.copy(ft_sensor_current[2])
                        if self.ft_sensor_z < -1:
                            self.real_ee_init_pos[2] += self._z_up
                            # print("up")
                        else:
                            self.real_ee_init_pos[2] -= self._z_down
                            # print(_target_ee_pos_z)
                            # print("down")
                        # print(self._init)
                    # print(self.real_ee_init_pos)
                    self.real_ee_init_pos[2] = np.clip(self.real_ee_init_pos[2], self.z_low_fornodsl, 1)

                    # print(self.real_ee_init_pos)
                    desired_pos = np.copy(self.real_ee_init_pos)
                    # desired_pos[:2] = np.copy([0.06505750256153474, -0.325988738975769]) # for ik testing
                    ee_displacement[-1] = ee_displacement[-1] * 3
                    # !theoretical data method
                    self.real_init_z_rot = self.real_init_z_rot + ee_displacement[-1]
                    ee_jpos_target = np.copy(self.real_init_z_rot)
                    # !raw data method
                    # print(ee_displacement)
                    # ee_jpos_current = np.array(self.rtde_r.getActualQ())[-1]
                    # while ee_jpos_current > 3.14:
                    #     ee_jpos_current -= 3.14
                    # while ee_jpos_current < -3.14:
                    #     ee_jpos_current += 3.14
                    # ee_jpos_target = ee_jpos_current / self.d2r + ee_displacement[-1]

                    target_z_rot = np.clip(ee_jpos_target, self.real_z_rot_low, self.real_z_rot_high)
                    desired_rotation = np.array([-180 * self.d2r, 0, target_z_rot * self.d2r])
                    FT_data = self.rtde_r.getActualTCPForce() - self.real_init_ft_sensor
                    updated_pos, updated_rot, self.admittance_params, self.admittance_paramsT = self.admittance_controller.admittance_control(desired_pos, desired_rotation, FT_data, self.admittance_params, self.admittance_paramsT)
                    updated_pos = np.around(updated_pos, 5)
                    _target_ee_pos_xy = -updated_pos[:2]
                    _target_ee_pos_z = np.copy(updated_pos[2]) + 0.86
                    current_joint = np.array(self.rtde_r.getActualQ())
                    target_ee_pos = np.concatenate((_target_ee_pos_xy, [_target_ee_pos_z]))
                    # print(target_ee_pos, desired_pos)
                    target_quat = euler_to_quaternion(updated_rot[0], updated_rot[1], updated_rot[2])
                    # _target_ee_pos_xy = np.array([-0.0555, 0.32])
                    # print("current tcp pos:", current_ee_pos[:3])
                    target_quat = np.roll(target_quat, 3)
                
                    target_arm_angles = self.inverse_kinematics(
                        current_joint=current_joint, 
                        target_position=target_ee_pos, 
                        target_orientation=target_quat
                    )
                    target_arm_angles_original = np.copy(target_arm_angles)
                    # print("current arm:", self.rtde_r.getActualQ())
                    # print("target arm:", target_arm_angles)
                    return target_arm_angles_original, target_arm_angles
                # --------------------------------------
                else:
                    # vision or nodsl without admittance control, need to be modified ...
                    # print("current_ee_pos:", current_ee_pos)
                    _target_ee_pos_xy = -(current_ee_pos[:2]) + ee_displacement[:2]
                    _target_ee_pos_z = np.copy(current_ee_pos[2]) + 0.86 + ee_displacement[2]
            else:
                xyz = 2
                if self.z_press_suc_items > 0:
                    ee_displacement = np.zeros(3)
                if self.vision_touch == "vision-touch":
                    ee_displacement[0] = -ee_displacement[0] * self.ee_dis_ratio * 0.5
                    ee_displacement[1] = -ee_displacement[1] * self.ee_dis_ratio * 1
                elif self.vision_touch == "touch":
                    ee_displacement[0] = -ee_displacement[0] * self.ee_dis_ratio * 0.8
                    ee_displacement[1] = -ee_displacement[1] * self.ee_dis_ratio * 1
                self.real_ee_init_pos[:2] = self.real_ee_init_pos[:2] + ee_displacement[:2] 
                # print("real ee pos trajectory:",self.real_ee_init_pos)
                _target_ee_pos_xy = -np.copy(self.real_ee_init_pos[:2])
                # print("current_ee_pos:", current_ee_pos)
                # _target_ee_pos_xy = -(current_ee_pos[:2]) + ee_displacement[:2]
                _target_ee_pos_z = np.copy(current_ee_pos[2]) + 0.86
            current_joint = np.array(self.rtde_r.getActualQ())
            # z direction ---------------
            if self.dsl is True:
                _ft_sensor = np.array(self.rtde_r.getActualTCPForce())
                ft_sensor_current = self.real_init_ft_sensor - _ft_sensor
                ft_threshold = self.ft_threashold
                for i in range(6):
                    if ft_sensor_current[i] >= ft_threshold:
                        ft_sensor_current[i] = ft_threshold
                    elif ft_sensor_current[i] <= -ft_threshold:
                        ft_sensor_current[i] = -ft_threshold
                    ft_sensor_current[i] = ft_sensor_current[i] / (ft_threshold/2)
                self.ft_sensor_z = np.copy(ft_sensor_current[2])
                if self.ft_sensor_z < -1:
                    _target_ee_pos_z += self._z_up
                    # print("up")
                else:
                    _target_ee_pos_z -= self._z_down
                    # print(_target_ee_pos_z)
                    # print("down")
                if self._init is False:
                    if self.z_press is True:
                        _target_ee_pos_z = np.copy(self._z_contact_record) - self.z_down_offset
                        self.z_press = False
                    else:
                        _target_ee_pos_z = np.copy(self._z_contact_record)
                        # self.z_press_suc_items = 0
                else:
                    pass
                if self.z_press_to_hole is True:
                    _target_ee_pos_z  = np.copy(self._z_contact_record) - self.z_down_offset - 0.0005 * self.z_press_suc_items
                    self.z_press_to_hole = False
            # -----------------------------
            target_ee_pos = np.concatenate((_target_ee_pos_xy, [_target_ee_pos_z]))
            # print("target_ee_pos:", target_ee_pos)
            # print("current_ee_pos:", current_ee_pos[:3])
            for i in range(6):
                while current_joint[i] > 3.14:
                    current_joint[i] -= 3.14
                while current_joint[i] < -3.14:
                    current_joint[i] += 3.14
            if self.matching_shape is True:
                ee_displacement[-1] = ee_displacement[-1] * 5.5
                # theoretical method ---------
                self.real_init_z_rot = self.real_init_z_rot + ee_displacement[-1]
                ee_jpos_target = np.copy(self.real_init_z_rot)
                target_z_rot = np.clip(ee_jpos_target, self.real_z_rot_low, self.real_z_rot_high)
                desired_rotation = np.array([-180 * self.d2r, 0, target_z_rot * self.d2r])
                target_quat = euler_to_quaternion(desired_rotation[0], desired_rotation[1], desired_rotation[2])
                target_quat = np.roll(target_quat, 3)
                # -------------------
                # current state method ----------
                # ee_jpos_current = np.array(self.rtde_r.getActualQ())[-1]
                # while ee_jpos_current > 3.14:
                #     ee_jpos_current -= 3.14
                # while ee_jpos_current < -3.14:
                #     ee_jpos_current += 3.14
                # ee_jpos_target = ee_jpos_current / self.d2r + ee_displacement[-1]
                # target_z_rot = np.clip(ee_jpos_target, self.z_rot_low, self.z_rot_high)
                # target_quat = euler_to_quaternion(-180 * self.d2r, 0, target_z_rot * self.d2r)
                # target_quat = np.roll(target_quat, 3)
                # ----------------
                target_arm_angles = self.inverse_kinematics(
                    current_joint=current_joint, 
                    target_position=target_ee_pos, 
                    target_orientation=target_quat
                )
                target_arm_angles_original = np.copy(target_arm_angles)
                # target_arm_angles = np.copy(target_arm_angles_original)
            else:
                print("use matching shape mode in real robot now, pls.")
                target_arm_angles = current_joint
            # target_arm_angles = current_joint
            return target_arm_angles_original, target_arm_angles

    def get_ee_position(self) -> np.ndarray:
        return self.get_body_position(self.ee_body)

    def get_ee_velocity(self) -> np.ndarray:
        return self.get_body_velocity(self.ee_body)
    
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
        ee_position = np.copy(self.get_ee_position())
        ee_position = (ee_position - self.ee_mean) * self.norm_scale + self.norm_mean
        # print("sim-ee-position:", ee_position)
        ee_velocity = np.array(self.get_ee_velocity())
        joint_angles = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        # print(joint_angles)
        joint_vels = np.array([self.sim.get_joint_velocity(joint=self.joint_list[i]) for i in range(6)])
        # ee_quat = self.sim.get_body_quaternion(self.ee_body)
        # print(ee_quat)
        ft_sensor = self.sim.get_ft_sensor(force_site='ee_force_sensor', torque_site='ee_torque_sensor').copy()
        ft_sensor_current = self.init_ft_sensor - ft_sensor
        # print(ft_sensor_current)
        for i in range(6):
            if ft_sensor_current[i] >= self.sim_ft_threshold:
                ft_sensor_current[i] = self.sim_ft_threshold
            elif ft_sensor_current[i] <= -self.sim_ft_threshold:
                ft_sensor_current[i] = -self.sim_ft_threshold
            ft_sensor_current[i] = ft_sensor_current[i] / (self.sim_ft_threshold / 2)
        if self._normalize_obs is True:
            # ee_position = normalizeVector(data=ee_position)
            # ee_velocity = normalizeVector(data=ee_velocity)
            joint_vels = normalizeVector(data=joint_vels)
        if self.real_robot is True:
            _ee_pos = np.array(self.rtde_r.getActualTCPPose())[:3]
            # print("ee pos:", _ee_pos)
            _ee_vel = np.array(self.rtde_r.getActualTCPSpeed())[:3]
            _ee_jpos = np.array(self.rtde_r.getActualQ())[-1]
            _ee_jpos -= self.z_rot_offset * self.d2r
            _ee_pos[:2] = -_ee_pos[:2]
            _ee_vel[0] = -_ee_vel[0]
            _ee_vel[2] = -_ee_vel[2]
            _ee_pos[2] += 0.86
            _ee_pos = (_ee_pos - self.ee_mean) * self.norm_scale + self.norm_mean
            while _ee_jpos > 3.14:
                _ee_jpos -= 3.14
            while _ee_jpos < -3.14:
                _ee_jpos += 3.14
            _ft_sensor = np.array(self.rtde_r.getActualTCPForce())
            ft_sensor_current = self.real_init_ft_sensor - _ft_sensor
            # print(ft_sensor_current, self.real_init_ft_sensor, _ft_sensor)
            ft_threshold = self.ft_threashold
            for i in range(6):
                if ft_sensor_current[i] >= ft_threshold:
                    ft_sensor_current[i] = ft_threshold
                elif ft_sensor_current[i] <= -ft_threshold:
                    ft_sensor_current[i] = -ft_threshold
                if i == 0 or i == 1:
                    ft_sensor_current[i] = (ft_sensor_current[i] / (ft_threshold/2)) * 5
                if i == 3 or i == 4:
                    ft_sensor_current[i] = (ft_sensor_current[i] / (ft_threshold/2)) * 5
                else:
                    ft_sensor_current[i] = ft_sensor_current[i] / (ft_threshold/2) 
        if self.vision_touch == 'vision':
            # obs = np.concatenate((ee_position, ee_velocity, [joint_angles[-1]]))
            if self.real_robot is True:
                # print(_ee_pos, _ee_vel, _ee_jpos)
                obs = np.concatenate((_ee_pos, [_ee_jpos]))
                # print(_ee_pos, _ee_jpos)
            else:
                # obs = np.concatenate((ee_position, ee_velocity, [joint_angles[-1]]))
                obs = np.concatenate((ee_position, [joint_angles[-1]]))
                # print(ee_position[:2])
                # print(ee_position)
            # obs = np.copy(joint_vels)
            # print("ee_force_sensor:", self.sim.get_ft_sensor())
            # # print("ee_torque_sensor:", self.sim.get_touch_sensor('ee_torque_sensor'))
            # print("-----------------------------")
        elif self.vision_touch == 'vision-touch':
            if self.real_robot is True:
                if self.dsl is True:
                    # ft_sensor_current_exchanged = np.array([ft_sensor_current[0], ft_sensor_current[1], ft_sensor_current[3], ft_sensor_current[4]])
                    # ft_sensor_current[0] = -np.copy(ft_sensor_current_exchanged[1])
                    # ft_sensor_current[1] = -np.copy(ft_sensor_current_exchanged[0])
                    # ft_sensor_current[3] = -np.copy(ft_sensor_current_exchanged[3])
                    # ft_sensor_current[4] = -np.copy(ft_sensor_current_exchanged[2])
                    # ft_sensor_current[0] = -np.copy(ft_sensor_current[1])
                    # ft_sensor_current[1] = -np.copy(ft_sensor_current[0])
                    # ft_sensor_current[3] = -np.copy(ft_sensor_current[4])
                    # ft_sensor_current[4] = -np.copy(ft_sensor_current[3])
                    ft_sensor_current[0] = -ft_sensor_current[0]
                    ft_sensor_current[1] = -ft_sensor_current[1]
                    ft_sensor_current[3] = -ft_sensor_current[3]
                    ft_sensor_current[4] = -ft_sensor_current[4]
                    obs = np.concatenate((_ee_pos[:2], ft_sensor_current, [_ee_jpos]))
                    # print(_ee_pos[:2])
                else:
                    obs = np.concatenate((_ee_pos, ft_sensor_current, [_ee_jpos]))
                # print(_ee_pos[:2])
            else:
                fingers_width = self.get_fingers_width() / 2 / self.gripper_max_joint
                # touch_sensors = np.array([self.sim.get_touch_sensor(sensor=self.sensor_list[i]) for i in range(self.sensor_num)])
                # if self._normalize_obs is True:
                    # touch_sensors = normalizeVector(data=touch_sensors, min=0, max=1)
                # obs = np.concatenate((ee_position, ee_velocity, joint_angles, touch_sensors, [fingers_width]))
                # obs = np.concatenate((joint_vels, touch_sensors, [fingers_width]))
                # obs = np.concatenate((ee_position, ee_velocity, [fingers_width], ft_sensor_current))
                # obs = np.concatenate((ee_position, ee_velocity, ft_sensor_current, [joint_angles[-1]]))
                # print(joint_angles[-1])
                if self.dsl is True:
                    obs = np.concatenate((ee_position[:2], ft_sensor_current, [joint_angles[-1]]))
                else:
                    obs = np.concatenate((ee_position, ft_sensor_current, [joint_angles[-1]]))
                self.ft_sensor_z = np.copy(ft_sensor_current[2])
                # print(ee_position[:2])
                # print(obs)
            # fingers_width = self.get_fingers_width() / 2 / self.gripper_max_joint
            # # touch_sensors = np.array([self.sim.get_touch_sensor(sensor=self.sensor_list[i]) for i in range(self.sensor_num)])
            # # if self._normalize_obs is True:
            #     # touch_sensors = normalizeVector(data=touch_sensors, min=0, max=1)
            # # obs = np.concatenate((ee_position, ee_velocity, joint_angles, touch_sensors, [fingers_width]))
            # # obs = np.concatenate((joint_vels, touch_sensors, [fingers_width]))
            # # obs = np.concatenate((ee_position, ee_velocity, [fingers_width], ft_sensor_current))
            # obs = np.concatenate((ee_position, ee_velocity, ft_sensor_current, [joint_angles[-1]]))
            # # print(joint_angles[-1])
            # self.ft_sensor_z = np.copy(ft_sensor_current[2])
        # recording_ur_obs = np.r_[[recording_ur_obs], [obs]]
        elif self.vision_touch == 'touch':
            if self.real_robot is True:
                ft_sensor_current[0] = -ft_sensor_current[0]
                ft_sensor_current[3] = -ft_sensor_current[3]
                obs = np.concatenate((_ee_pos, ft_sensor_current, [_ee_jpos]))
            else:
                fingers_width = self.get_fingers_width() / 2 / self.gripper_max_joint
                # touch_sensors = np.array([self.sim.get_touch_sensor(sensor=self.sensor_list[i]) for i in range(self.sensor_num)])
                # if self._normalize_obs is True:
                    # touch_sensors = normalizeVector(data=touch_sensors, min=0, max=1)
                # obs = np.concatenate((ee_position, ee_velocity, joint_angles, touch_sensors, [fingers_width]))
                # obs = np.concatenate((joint_vels, touch_sensors, [fingers_width]))
                # obs = np.concatenate((ee_position, ee_velocity, [fingers_width], ft_sensor_current))
                obs = np.concatenate((ee_position, ft_sensor_current, [joint_angles[-1]]))
        # # -------------------------- DSL -----------------------------
        if self.dsl is False and self.real_robot is True:
            if self._init is False:
                if self._reset is True:
                    while True:
                        if ft_sensor_current[2] > -0.6:
                            self.z_press_to_hole = True
                            _, target_arm_angles = self.ee_displacement_to_target_arm_angles(np.array([0, 0, -0.001, 0]))
                            current_joint_arm = np.array(self.rtde_r.getActualQ())
                            self.real_joint_planner(grasping=True,
                                                    _time=self.planner_time/50, 
                                                    current_joint=current_joint_arm, 
                                                    target_joint=target_arm_angles)
                            self.z_press_suc_items += 1
                            if self.z_press_suc_items > 10:
                                self.gripper.move_and_wait_for_pos(130, 255, 255)
                                self.rtde_c.moveJ(self.j_init, self.vel, self.acc)
                                sys.exit(0)

                            time.sleep(0.3)
                        else:
                            self.z_press_suc_items = 0
                            break
        if self.dsl is True and self.vision_touch != "vision":
            if self._init is False:
                if self.real_robot is False:
                    if ft_sensor_current[2] < -1:
                        self.z_press = False
                        target_arm_angles = self.ee_displacement_to_target_arm_angles(np.array([0, 0, 0]))
                        current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
                        # current_joint_gripper = np.array(self.get_fingers_width()/2)
                        # # current_joint_gripper = np.clip(current_joint_gripper, self.gripper_joint_low, self.gripper_joint_high)
                        # current_joint = np.concatenate((current_joint_arm, [current_joint_gripper]))
                        self.joint_planner(grasping=True, _time=self.planner_time/50, current_joint=current_joint_arm, target_joint=target_arm_angles)
                    else:
                        self.z_press = True
                else:
                    if self._reset is True:
                        if ft_sensor_current[2] < -1:
                            # print(ft_sensor_current)
                            self.z_press = False
                            _, target_arm_angles = self.ee_displacement_to_target_arm_angles(np.array([0, 0, 0]))
                            current_joint_arm = np.array(self.rtde_r.getActualQ())
                            self.real_joint_planner(grasping=True,
                                                    _time=self.planner_time/50, 
                                                    current_joint=current_joint_arm, 
                                                    target_joint=target_arm_angles)
                            self.z_press_suc_items = 0
                            time.sleep(0.3)
                        else:
                            self.z_press_to_hole = True
                            _, target_arm_angles = self.ee_displacement_to_target_arm_angles(np.array([0, 0, 0]))
                            current_joint_arm = np.array(self.rtde_r.getActualQ())
                            self.real_joint_planner(grasping=True,
                                                    _time=self.planner_time/50, 
                                                    current_joint=current_joint_arm, 
                                                    target_joint=target_arm_angles)
                            self.z_press_suc_items += 1
                            if self.z_press_suc_items > 8:
                                self.gripper.move_and_wait_for_pos(130, 255, 255)
                                self.rtde_c.moveJ(self.j_init, self.vel, self.acc)
                                sys.exit(0)

                            time.sleep(0.3)
                    else:
                        pass
        # # --------------------------------------------------------
            # print(target_arm_angles)
        # print(ee_position)
        return obs

    def reset(self) -> None:
        self.sim.reset()
        self.init_z_rot = 0.0
        self.set_joint_neutral()
        # print("reset---")
        # init_qpos = np.array([ 1.2279318 , -1.56335969 , 1.30385035 ,-1.27012857 ,-1.40113781, -0.23923021])
        if self.real_robot is True:
            _items = 500
            ft_record = np.zeros((_items, 6))
            acc_record = np.zeros((_items, 3))
            print("reset real robot pos and init the grasping for insertion.")
            self._init_grasping_for_realRobot()
            print("moveing ee to init pos ...")
            time.sleep(2)
            self.rtde_c.moveJ(self.j_init, self.vel, self.acc)
            self.real_ee_init_pos = self.rtde_r.getActualTCPPose()[:3]
            self.z_rot_offset = np.array(self.rtde_r.getActualQ())[-1] / self.d2r
            self.real_init_z_rot = np.array(self.rtde_r.getActualQ())[-1] / self.d2r
            self.real_z_rot_low = self.z_rot_offset + self.z_rot_low
            self.real_z_rot_high = self.z_rot_offset + self.z_rot_high
            for i in range(_items):
                ft_record[i] = self.rtde_r.getActualTCPForce()
                acc_record[i] = self.rtde_r.getActualToolAccelerometer()
                time.sleep(0.01)
            print(self.rtde_r.getActualTCPPose()[:3])
            self.real_init_ft_sensor = np.mean(ft_record, axis=0)
            print("init_ft:", self.real_init_ft_sensor)
            print("init over.")
            self._reset = True
            time.sleep(1)
            # #  used for testing adimittance controller --------------
            # desired_position = self.rtde_r.getActualTCPPose()[:3]
            # ee_jpos_current = np.array(self.rtde_r.getActualQ())[-1]
            # while ee_jpos_current > 3.14:
            #     ee_jpos_current -= 3.14
            # while ee_jpos_current < -3.14:
            #     ee_jpos_current += 3.14
            # ee_jpos_target = ee_jpos_current / self.d2r
            # target_z_rot = np.clip(ee_jpos_target, self.z_rot_low, self.z_rot_high)
            # desired_rotation = np.array([-180 * self.d2r, 0, target_z_rot * self.d2r])
            # init_acc = np.mean(acc_record, axis=0)
            # init_vel = self.rtde_r.getActualTCPSpeed()[:3]
            # # print(init_acc)
            # # print(init_vel)
            # ik_pos = np.zeros(3)
            # # ----------------- compliance state ------------
            # _mat = np.zeros((3, 3))
            # T_mat = np.zeros((3, 3))
            # # -------------------------------------------------
            # items = 5
            # i = 0
            # ft_record = [np.zeros(6)]
            # while True:
            #     i += 1
            #     # begin = datetime.now()
            #     # current_pos = self.rtde_r.getActualTCPPose()[:3]
            #     # current_vel = self.rtde_r.getActualTCPSpeed()[:3]
            #     # current_acc = self.rtde_r.getActualToolAccelerometer()
            #     current_FT = self.rtde_r.getActualTCPForce()
            #     # print("current F/T:",current_FT)
            #     # # # print(current_acc)
            #     # print("current TCP pos:",current_pos)
            #     FT_data = current_FT - self.real_init_ft_sensor
            #     ft_record = np.r_[ft_record, [-FT_data]]
            #     np.save('/home/yi/peg_in_hole/ur3_rl_sim2real/src/recording/' + 'forcetest_real' +'.npy', ft_record)
            #     # # print(FT_data)
            #     # --------------- current state --------
            #     # ddx_e = init_acc[0] - current_acc[0]
            #     # ddy_e = init_acc[1] - current_acc[1]
            #     # ddz_e = init_acc[2] - current_acc[2]
            #     # # ddx_e, ddy_e, ddz_e = init_acc - current_acc
            #     # # dx_e, dy_e, dz_e = init_vel - current_vel
            #     # # x_e, y_e, z_e = desired_position - current_pos
            #     # # print(ddx_e, ddy_e, ddz_e)
            #     # dx_e = init_vel[0] - current_vel[0]
            #     # dy_e = init_vel[1] - current_vel[1]
            #     # dz_e = init_vel[2] - current_vel[2]
            #     # x_e = desired_position[0] - current_pos[0]
            #     # y_e = desired_position[1] - current_pos[1]
            #     # z_e = desired_position[2] - current_pos[2]
            #     # ----------------------
            #     # print(ddx_e, ddy_e, ddz_e)
            #     # print(ddx_e, dx_e, x_e)
            #     # print(FT_data)
            #     # time.sleep(10)
            #     updated_pos, update_rotation, _mat, T_mat = self.admittance_controller.admittance_control(desired_position, desired_rotation, FT_data, _mat, T_mat)
            #     updated_pos = np.around(updated_pos, 5)
            #     # ------------------------------
            #     # print("updated TCP pos:",updated_pos)
            #     # eeposFK = self.sim.forward_kinematics(qpos=self.j_init)
            #     # eeposFK[-1] -= 0.86
            #     # print("ee pos FK:", eeposFK)
            #     # print("-------------")
            #     # time.sleep(10)
            #     ik_pos[2] = np.copy(updated_pos[2]) + 0.86
            #     ik_pos[:2] = -np.copy(updated_pos[:2])
            #     ee_jpos_current = np.array(self.rtde_r.getActualQ())[-1]
            #     while ee_jpos_current > 3.14:
            #         ee_jpos_current -= 3.14
            #     while ee_jpos_current < -3.14:
            #         ee_jpos_current += 3.14
            #     ee_jpos_target = ee_jpos_current / self.d2r
            #     target_z_rot = np.clip(ee_jpos_target, self.z_rot_low, self.z_rot_high)
            #     # target_quat = euler_to_quaternion(-180 * self.d2r, 0*self.d2r, target_z_rot * self.d2r)
            #     target_quat = euler_to_quaternion(update_rotation[0], update_rotation[1], update_rotation[2])
            #     # print(180 * self.d2r, 0, target_z_rot * self.d2r)
            #     target_quat = np.roll(target_quat, 3)
            #     current_joint = np.array(self.rtde_r.getActualQ())
            #     target_arm_angles = self.inverse_kinematics(
            #         current_joint=current_joint, 
            #         target_position=ik_pos,
            #         target_orientation=target_quat
            #     )
            #     # print("init qpos:", self.j_init)
            #     # print("target arm qpos:",target_arm_angles)
            #     self.real_joint_planner(grasping=True,
            #                         _time=self.planner_time/50, 
            #                         current_joint=current_joint, 
            #                         target_joint=target_arm_angles)
            #     # print(desired_position[:2], desired_position[2] + 0.86)
            #     # print(ik_pos)
            #     # # print(FT_data)
            #     # # print(current_joint)
            #     # # print(target_arm_angles)
            #     # print("--------------------------")
                
            #     time.sleep(0.3)
            #     # if i > items:
            #     #     desired_position[-1] -= 0.001
            #     #     if FT_data[2] > 10:
            #     #         desired_position[-1] += 0.006
            #     #     i = 0
            #     # print("time(millseconds):", ((datetime.now() - begin).microseconds) / 1000)
            # # --------------------------------------------------------------
                


    def log_info(self, gripper):
        print(f"Pos: {str(gripper.get_current_position()): >3}  "
            f"Open: {gripper.is_open(): <2}  "
            f"Closed: {gripper.is_closed(): <2}  ")

    def _init_grasping_for_realRobot(self):
        print("grapsing for real robot ...")
        target1joint = np.array([2.28441358, -1.6466042,  1.94246418, -2.05400958, -1.55255968, -0.07506973])
        target2joint = np.array([2.28441358, -1.41584693,  1.94246418, -2.05400958, -1.55255968, -0.07506973])
        self.rtde_c.moveJ(target1joint, self.vel, self.acc)
        self.rtde_c.moveJ(target2joint, self.vel, self.acc)
        self.gripper.move_and_wait_for_pos(203, 255, 255)
        # self.gripper.move_and_wait_for_pos(250, 255, 255)

        
    def set_joint_neutral(self) -> None:
        self.set_joint_angles(self.neutral_joint_values)
        
    def _init_grasping(self) -> None:
        # #! fixed object programme ------------------------------------------
        # if self.vision_touch == 'touch':
        #     random_done = False
        #     # # hole_top_position = np.copy(self.get_body_position("box"))
        #     # hole_top_position = np.copy(self.ee_init_pos)
        #     # hole_top_position[2] += 0.13
        #     # current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        #     # target_arm_angles = self.inverse_kinematics(
        #     #     current_joint=current_joint, target_position=hole_top_position, target_orientation=np.array([-1.0, 0.0, 0.0, 0.0])
        #     # )
        #     # self.set_joint_angles(angles=target_arm_angles)
        #     # self.set_forwad()
        #     _goal = self.sim.get_site_position('box_surface')
        #     x_random = np.random.uniform(self.random_low, self.random_high)
        #     y_random = np.random.uniform(self.random_low, self.random_high)
        #     while random_done is False:
        #         if (x_random > self.random_lim_low and x_random < self.random_lim_high) | ((y_random > self.random_lim_low and y_random < self.random_lim_high)):
        #             x_random = np.random.uniform(self.random_low, self.random_high)
        #             y_random = np.random.uniform(self.random_low, self.random_high)
        #         else:
        #             random_done = True
        #     self.ee_init_pos[0] = x_random + _goal[0]
        #     self.ee_init_pos[1] = y_random + _goal[1]
        #     self.ee_init_pos[2] = 0.89 + 0.16
        #     # hole_top_position[0] = x_random
        #     # hole_top_position[1] = y_random
        #     current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        #     target_arm_angles = self.inverse_kinematics(
        #         current_joint=current_joint, target_position=self.ee_init_pos, target_orientation=np.array([-1.0, 0.0, 0.0, 0.0])
        #     )
        #     # target_arm_angles = np.array([2.06492469 ,-0.99604245 , 0.73810484, -1.31285872, -1.57079633 , 0.49412837])
        #     # target_arm_angles = np.copy(self.neutral_joint_values)
        #     # print("init target_arm_angles:", target_arm_angles)
        #     self.set_joint_angles(angles=target_arm_angles)
        #     self.set_forwad()
        # #! -----------------------------------------------------------------
        # else:
        # # #! moveable object initial programme. ----------------------------
        # #     #! step 1: get object position
        # #     object_position = self.get_body_position("cylinder_obj")
        # #     #! step 2: grasping
        # #     current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        # #     target_arm_angles = self.inverse_kinematics(
        # #         current_joint=current_joint, target_position=object_position, target_orientation=np.array([-1.0, 0.0, 0.0, 0.0])
        # #     )
        # #     self.set_joint_angles(angles=target_arm_angles)
        # #     self.set_forwad()
        # #     finger_joint = np.array([0.4])
        # #     grasping_joint = np.concatenate((target_arm_angles, finger_joint))
        # #     while self.get_joint_angle("right_driver_1") < 0.2:
        # #         self.control_joints(grasping_joint)
        # #         # print(self.get_joint_angle("right_driver_1"))
        # #         self.sim.step()
        # #     #! step 3: lifting
        # #     object_position[2] += 0.1
        # #     current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        # #     target_arm_angles = self.inverse_kinematics(
        # #         current_joint=current_joint, target_position=object_position, target_orientation=np.array([-1.0, 0.0, 0.0, 0.0])
        # #     )
        # #     print('lifting...')
        # #     self.joint_planner(grasping=True, _time=self.planner_time, current_joint=current_joint, target_joint=target_arm_angles)
        # #     #! step 4: moving object to top of the hole
        # #     print('moving...')
        # #     hole_top_position = np.copy(self.get_body_position("box"))
        # #     hole_top_position[2] += 0.11
        # #     current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        # #     target_arm_angles = self.inverse_kinematics(
        # #         current_joint=current_joint, target_position=hole_top_position, target_orientation=np.array([-1.0, 0.0, 0.0, 0.0])
        # #     )
        # #     self.joint_planner(grasping=True, _time=self.planner_time, current_joint=current_joint, target_joint=target_arm_angles)
        # #     #! -----------------------------------------------------------------
        #     # hole_top_position = np.copy(self.get_body_position("box"))

        #     # hole_top_position = np.copy(self.ee_init_pos)
        #     # hole_top_position[2] += 0.13
        #     # current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        #     # target_arm_angles = self.inverse_kinematics(
        #     #     current_joint=current_joint, target_position=hole_top_position, target_orientation=np.array([-1.0, 0.0, 0.0, 0.0])
        #     # )
        #     # self.set_joint_angles(angles=target_arm_angles)
        #     # self.set_forwad()

        #     # hole_top_position = np.copy(self.ee_init_pos)
        #     # hole_top_position[2] += 0.16
        #     random_done = False
        #     _goal = self.sim.get_site_position('box_surface')
        #     x_random = np.random.uniform(-0.075, 0.075)
        #     y_random = np.random.uniform(-0.075, 0.075)
        #     while random_done is False:
        #         if (x_random > -0.02 and x_random < 0.02) | ((y_random > -0.02 and y_random < 0.02)):
        #             x_random = np.random.uniform(-0.075, 0.075)
        #             y_random = np.random.uniform(-0.075, 0.075)
        #         else:
        #             random_done = True
        #     self.ee_init_pos[0] = x_random + _goal[0]
        #     self.ee_init_pos[1] = y_random + _goal[1]
        #     self.ee_init_pos[2] = 0.89 + 0.16
        #     # hole_top_position[0] = x_random
        #     # hole_top_position[1] = y_random
        #     current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        #     target_arm_angles = self.inverse_kinematics(
        #         current_joint=current_joint, target_position=self.ee_init_pos, target_orientation=np.array([-1.0, 0.0, 0.0, 0.0])
        #     )
        #     # target_arm_angles = np.array([2.06492469 ,-0.99604245 , 0.73810484, -1.31285872, -1.57079633 , 0.49412837])
        #     # target_arm_angles = np.copy(self.neutral_joint_values)
        #     # print("init target_arm_angles:", target_arm_angles)
        #     self.set_joint_angles(angles=target_arm_angles)
        #     self.set_forwad()
        #     # self.sim.step()
        #     # current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        #     # print("over current angles:", current_joint)
        #     # time.sleep(2)

        #----------------------------------------------------
        random_done = False
        # _goal = self.sim.get_site_position('box_surface') # 1-method: randomize the init eef pos 
        _goal = np.array([-0.05, 0.32, 0.93]) # 2-method: fixed the init eef pos and randomize the hole in pih.py
        x_random = np.random.uniform(self.random_low, self.random_high)
        y_random = np.random.uniform(self.random_low, self.random_high)
        while random_done is False:
            if (x_random > self.random_lim_low and x_random < self.random_lim_high) | ((y_random > self.random_lim_low and y_random < self.random_lim_high)):
                x_random = np.random.uniform(self.random_low, self.random_high)
                y_random = np.random.uniform(self.random_low, self.random_high)
            else:
                random_done = True
        # self.ee_init_pos[0] = x_random + _goal[0]
        # self.ee_init_pos[1] = y_random + _goal[1]
        self.ee_init_pos[0] = _goal[0]
        self.ee_init_pos[1] = _goal[1]
        self.ee_init_pos[2] = 0.89 + 0.16
        current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        target_arm_angles = self.inverse_kinematics(
            current_joint=current_joint, 
            target_position=self.ee_init_pos, 
            target_orientation=np.array([-1.0, 0.0, 0.0, 0.0])
        )
        # print(target_arm_angles)
        self.set_joint_angles(angles=target_arm_angles)
        self.set_forwad()
        self._init = True
        # for i in range(10):
        # ft_record = [np.zeros(6)]
        # while True:
        #     # self.init_ft_sensor = self.sim.get_ft_sensor().copy()
        #     last_time = time.time()
        #     ft_sensor = self.sim.get_ft_sensor().copy()
        #     ft_current = self.init_ft_sensor - ft_sensor
        #     target_arm_angles = self.ee_displacement_to_target_arm_angles(np.array([0.0, 0.0, 0.0, 0]))
        #     current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        #     current_joint_gripper = np.array(self.get_fingers_width()/2)
        #     # current_joint_gripper = np.clip(current_joint_gripper, self.gripper_joint_low, self.gripper_joint_high)
        #     current_joint = np.concatenate((current_joint_arm, [current_joint_gripper]))
            
        #     self.joint_planner(grasping=True, _time=self.planner_time/50, 
        #                         current_joint=current_joint_arm,
        #                         target_joint=target_arm_angles)
        #     self.sim.step()
        #     # print(self.ee_init_pos, self.sim.get_site_position('ee_tool'))
        #     ft_record = np.r_[ft_record, [ft_current]]
        #     # print('time:%s' % ((time.time() - last_time)*1000))
        #     np.save('/home/yi/peg_in_hole/ur3_rl_sim2real/src/recording/' + 'forcetest' +'.npy', ft_record)

        # # # time.sleep(5)


    def joint_planner(self, grasping: bool, _time: float, current_joint: np.ndarray, target_joint: np.ndarray) -> None:
        delta_joint = target_joint - current_joint
        planner_steps = int(_time / self.sim.timestep)
        if grasping is True:
            finger_joint = np.array([0.42])
            for i in range(planner_steps):
                planned_delta_joint = delta_joint * math.sin((math.pi/2)*(i/planner_steps))
                target_joint = planned_delta_joint + current_joint
                target_full_joint = np.concatenate((target_joint, finger_joint))
                self.control_joints(target_full_joint)
                self.sim.step()
                # time.sleep(2)
        else:
            for i in range(planner_steps):
                planned_delta_joint = delta_joint * math.sin((math.pi/2)*(i/planner_steps))
                target_joint = planned_delta_joint + current_joint
                self.control_joints(target_joint)
                self.sim.step()
                # time.sleep(2)

    def real_joint_planner(self, grasping: bool, _time: float, current_joint: np.ndarray, target_joint: np.ndarray) -> None:
        delta_joint = target_joint - current_joint
        abs_delta_joint = np.absolute(delta_joint)
        if np.any(abs_delta_joint > 0.3):
            delta_joint = np.zeros(6)
            print("delta joint is too big. --------")
            print("current joint", current_joint)
            print("target joint", target_joint)
        
        planner_steps = 10
        vel = 0.001
        acc = 0.001
        dt = 1.0 / 500
        lookahead_time = 0.05
        gain = 300

        # target_joint = current_joint + delta_joint
        # self.rtde_c.servoJ(target_joint, vel, acc, dt, lookahead_time, gain)

        # self.rtde_c.moveJ(target_joint, self.vel, self.acc, True)

        # if self._init is False:
        #     time.sleep(0.1)
        # print("---testing1---",target_joint)
        # print("---testing2---",current_joint)
        delta_joint = np.array(delta_joint.tolist())
        # print(delta_joint, type(delta_joint))
        for i in range(planner_steps):
            planned_delta_joint = delta_joint * math.sin((math.pi/2)*(i/planner_steps))
            target_joint = planned_delta_joint + current_joint
            # print(target_joint)
           
            self.rtde_c.servoJ(target_joint, vel, acc, dt, lookahead_time, gain)
            time.sleep(0.001)