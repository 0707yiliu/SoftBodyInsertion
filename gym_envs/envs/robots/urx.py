# MODEL: Universal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 04/10/2022
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

from array import array
from email.mime import base
from typing import Optional, Tuple, Any

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
from gym_envs.utils import quaternion_to_euler
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
        ee_dis_ratio: float = 0.00085, # this one has been transferred from run.py
        ee_rotxy_ratio: float = 0.01,
        ee_rotz_ratio: float = 0.09,
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
        ur_gen: int = 5,
        dsl_dampRatio_d: np.ndarray = np.array([0, 0, 0]),
        dsl_dampRatio_r: np.ndarray = np.array([0, 0, 0]),
        ft_LowPass_damp: float = 0.1,
        ft_xyz_threshold_ur3: np.ndarray = np.array([0, 0, 0]),
        ft_rpy_threshold_ur3: np.ndarray = np.array([0, 0, 0]),
        ft_threshold_xyz: float = 0.3,
        ft_threshold_rpy: float = 0.2,
        ro: bool = True,
        ee_admittancecontrol: bool = True,
    ) -> None:
        # admittance control params ---
        self.ee_admittancecontrol = ee_admittancecontrol
        if self.ee_admittancecontrol is True:
            self._mat = np.zeros((3, 3))
            self.T_mat = np.zeros((3, 3))
            if real_robot is True:
                self.admittance_controller = FT_controller(m=0.5, b=500, k=1000,
                                                           dt=1 / 500,
                                                           tau_m=0.3, tau_b=4, tau_k=8)
            else:
                self.admittance_controller = FT_controller(m=0.4, b=200, k=400,
                                                           dt=1 / 500,
                                                           tau_m=0.4, tau_b=4, tau_k=8)
            self.admittance_params = np.zeros((3, 3))
            self.admittance_paramsT = np.zeros((3, 3))
        # -----------
        self.running_time = datetime.now().strftime("%m%d%H%M%S")
        self.ro = ro
        self.dsl_dampRatio_d = dsl_dampRatio_d
        self.dsl_dampRatio_r = dsl_dampRatio_r
        self.low_filter_gain = ft_LowPass_damp
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
        self.ee_rotxy_ratio = ee_rotxy_ratio
        self.ee_rotz_ratio = ee_rotz_ratio
        # if ur_gen == 5:
        #     self.ee_dis_ratio = ee_dis_ratio / 10 # sim time = 0.001s, so the ee maximum movement is ee_dis_ratio(/m) between 1s.
        # elif ur_gen == 3:
        #     self.ee_dis_ratio = ee_dis_ratio / 20
        self.ee_dis_ratio = ee_dis_ratio
        self.gripper_action_ratio = gripper_action_ratio/40 if self.vision_touch =='vision' else gripper_action_ratio/20
        self.random_high = 0.045 if self.vision_touch == 'vision' else 0.045
        self.random_low = -0.045 if self.vision_touch == 'vision' else -0.045 # 0.014
        self.random_lim_high = 0.025 if self.vision_touch == 'vision' else 0.025
        self.random_lim_low = -0.025 if self.vision_touch == 'vision' else -0.025
        self._z_up = 0.01
        self._z_down = 0.002
        self.ft_last = np.array([0, 0, 0, 0, 0, 0])
        self.ft_last_obs = np.array([0, 0, 0, 0, 0, 0])

        if self.vision_touch == "vision":
            self.ft_record = [np.zeros(6)]

        # self._z_up = 0
        # self.random_high = 0.014 if self.vision_touch == 'vision' else 0.014
        # self.random_low = -0.014 if self.vision_touch == 'vision' else -0.014
        # self.random_lim_high = 0.004 if self.vision_touch == 'vision' else 0.004
        # self.random_lim_low = -0.004 if self.vision_touch == 'vision' else -0.004
        self.joint_dis_ratio = joint_dis_ratio
        # normalization part
        if ur_gen == 5:
            self.ee_position_low = ee_positon_low if ee_positon_low is not None else np.array([-0.1, 0.4, 0.845])
            self.ee_position_high = ee_positon_high if ee_positon_high is not None else np.array([0.3, 0.7, 1.23])
            self.ee_rot_low = np.array([-10, -10, -180])
            self.ee_rot_high = np.array([10, 10, 180])
            ft_xyz = 2
            ft_rxyz = 0.5
            self.ft_sensor_high = np.array([ft_xyz, ft_xyz, ft_xyz, ft_rxyz, ft_rxyz, ft_rxyz])
            self.ft_sensor_low = np.array([-ft_xyz, -ft_xyz, -ft_xyz, -ft_rxyz, -ft_rxyz, -ft_rxyz])
        elif ur_gen == 3:
            self.ft_sensor_high = np.array([ft_xyz_threshold_ur3[0], ft_xyz_threshold_ur3[1], ft_xyz_threshold_ur3[2], ft_rpy_threshold_ur3[0], ft_rpy_threshold_ur3[1], ft_rpy_threshold_ur3[2]])
            self.ft_sensor_low = np.array([-ft_xyz_threshold_ur3[0], -ft_xyz_threshold_ur3[1], -ft_xyz_threshold_ur3[2], -ft_rpy_threshold_ur3[0], -ft_rpy_threshold_ur3[1], -ft_rpy_threshold_ur3[2]])
            rotation_limitation = 35
            self.ee_rot_low = np.array([-rotation_limitation, -rotation_limitation, -rotation_limitation])
            self.ee_rot_high = np.array([rotation_limitation, rotation_limitation, rotation_limitation])
            self.ee_position_low = ee_positon_low if ee_positon_low is not None else np.array([0.25, 0.01, 0.845])
            self.ee_position_high = ee_positon_high if ee_positon_high is not None else np.array([0.45, 0.21, 1.045])
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
        print("action space:", action_space)
        norm_max = 1
        norm_min = -1
        # ee pos norm params --------
        self.ee_pos_scale = self.ee_position_high - self.ee_position_low
        self.ee_pos_norm_scale = (norm_max - norm_min) / self.ee_pos_scale
        self.ee_pos_mean = (self.ee_position_high + self.ee_position_low) / 2
        self.ee_pos_norm_mean = (norm_max + norm_min) / 2 * np.array([1, 1, 1])
        # ee rot norm params --------
        self.ee_rot_scale = self.ee_rot_high - self.ee_rot_low
        self.ee_rot_norm_scale = (norm_max - norm_min) / self.ee_rot_scale
        self.ee_rot_mean = (self.ee_rot_high + self.ee_rot_low) / 2
        self.ee_rot_norm_mean = (norm_max + norm_min) / 2 * np.array([1, 1, 1])
        # F/T sensor norm params ---------
        self.ft_scale = self.ft_sensor_high - self.ft_sensor_low
        self.ft_norm_scale = (norm_max - norm_min) / self.ft_scale
        self.ft_mean = (self.ft_sensor_high + self.ft_sensor_low) / 2
        self.ft_norm_mean = (norm_max + norm_min) / 2 * np.array([1, 1, 1, 1, 1, 1])
        # ------------ for real robot ------------------
        j_acc = 0.03
        j_vel = 0.015
        HOST = "10.42.0.162"
        if self.real_robot is True:
            self._reset = False
            self.z_press_to_hole = False
            
            self.rtde_r = rtde_receive.RTDEReceiveInterface(HOST)
            self.rtde_c = rtde_control.RTDEControlInterface(HOST)
            actual_q = np.array(self.rtde_r.getActualQ())
            print("current qpos:", actual_q)
            self.j_init = np.array([2.96690151, -1.60088217,  1.63154722, -1.60144448, -1.57070149, -1.74708013])
            print("init qpos:", self.j_init)
            self.vel = 0.8
            self.acc = 0.8
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
            self.fixedZoffsetwithSim = 0.87 # use the same range with sim, compensate the fixed high of table in simulation.
            _items = 500
            ft_record = np.zeros((_items, 6))
            for i in range(_items):
                ft_record[i] = self.rtde_r.getActualTCPForce()
                time.sleep(0.01)
            self.real_init_ft_sensor = np.mean(ft_record, axis=0)

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
        if ur_gen == 5:
            self.ft_threashold_xyz = 0.001
            self.ft_threashold_rxyz = 0.001
            self.neutral_joint_values = np.array([1.02992762, -1.44296501,  1.84817859, -1.97600992, -1.57079634, -0.5408687])
            # self.neutral_joint_values = np.array([1.19274333, -1.44175816, 1.74942402, -1.82642832, -1.54568981, 0])
            # self.neutral_joint_values = np.array([1.19274333, -1.84175816, 2.34942402, -2.02642832, -1.54568981, 0])
        elif ur_gen == 3:
            self.ft_threashold_xyz = ft_threshold_xyz
            self.ft_threashold_rxyz = ft_threshold_rpy
            # self.neutral_joint_values = np.array([0.83094203, -1.30108324, 1.05714794, -1.38607852, -1.52985284,  0])
            # self.neutral_joint_values = np.array([0.83094203, -1.80108324, 1.55714794, -1.38607852, -1.52985284,  0])
            # self.neutral_joint_values = np.array([1.19094203, -1.30108324, 1.05714794, -1.38607852, -1.52985284,  0])0.86640135, -1.28055101 , 1.07492697, -1.36517228 ,-1.57079633 ,-0.70439498
            # self.neutral_joint_values = np.array([2.6471949,  -1.57136391,  1.31850185, -1.31793427, -1.57079632, -2.06678673])
            self.neutral_joint_values = np.array([3.03594359, -1.57144191,  1.60263088, -1.60198554, -1.57079633, -1.67803776])
        self.ee_body = "eef"
        self.finger1 = "right_driver_1"
        self.finger2 = "left_driver_1"
        if self.matching_shape is True:
            self.init_z_rot = 0.0
            self.z_rot_high = 160.0
            self.z_rot_low = -160.0
            # print(n_action)
        #! -----------------------------------
        self.y_rot = 0 # testing for F/T sensor in displacement function
    def log_info(self):
        print(f"Pos: {str(self.gripper.get_current_position()): >3}  "
            f"Open: {self.gripper.is_open(): <2}  "
            f"Closed: {self.gripper.is_closed(): <2}  ")

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()
        # print(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.real_robot is False:
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
            self.joint_planner(grasping=False, _time=self.planner_time/50, 
                                current_joint=current_joint_arm,
                                target_joint=target_arm_angles)
        else: # for real robot
            ee_displacement = np.copy(action)
            target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)

            current_joint_arm = np.array(self.rtde_r.getActualQ())
            self.real_joint_planner(grasping=True,
                                    _time=self.planner_time/50, 
                                    current_joint=current_joint_arm, 
                                    target_joint=target_arm_angles)
            time.sleep(0.005)
            # if self._init is True:
            #     time.sleep(0.1)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        xyz = 3
        if self.real_robot is False:
            if self.dsl is True: # Dyn safety lock for EEF-XYZ-RPY
                current_ft = np.copy(self.sim.get_ft_sensor(force_site="ee_force_sensor", torque_site="ee_torque_sensor"))
                # print("current ft:", current_ft)
                for ft_xyz in range(3):
                    current_ft[ft_xyz] = 0 if self.ft_threashold_xyz > current_ft[ft_xyz] > -self.ft_threashold_xyz else current_ft[ft_xyz]
                    current_ft[ft_xyz+3] = 0 if self.ft_threashold_rxyz > current_ft[ft_xyz + 3] > -self.ft_threashold_rxyz else current_ft[ft_xyz + 3]
                # print("current ft threshold:", current_ft)
                current_ft = (current_ft - self.ft_mean) * self.ft_norm_scale + self.ft_norm_mean
                current_ft = self.ft_last * self.low_filter_gain + current_ft * (1 - self.low_filter_gain) # low-pass filter for F/T sensor
                # print("current ft norm and filted:", current_ft)
                self.ft_last = np.copy(current_ft)
                ee_dis_x = -math.tanh(3 * current_ft[0])
                ee_dis_y = -math.tanh(3 * current_ft[1])
                ee_dis_z = math.tanh(3 * current_ft[2])
                ee_dis_rx = -math.tanh(10 * current_ft[3])
                ee_dis_ry = -math.tanh(10 * current_ft[4])
                ee_dis_rz = math.tanh(10 * current_ft[5])
                dsl_ee_dis = np.array([ee_dis_x, ee_dis_y, ee_dis_z, ee_dis_rx, ee_dis_ry, ee_dis_rz])
                # print("dsl dis z:", ee_dis_z)
                # print("dsl dis:", dsl_ee_dis)
                # print("---------")
                ee_displacement[:xyz] = ee_displacement[:xyz] * self.ee_dis_ratio + dsl_ee_dis[:xyz] * self.dsl_dampRatio_d
                ee_displacement[xyz:-1] = ee_displacement[xyz:-1] * self.ee_rotxy_ratio + dsl_ee_dis[xyz:-1] * self.dsl_dampRatio_r[:2]
                ee_displacement[-1] = ee_displacement[-1] * self.ee_rotz_ratio + dsl_ee_dis[-1] * self.dsl_dampRatio_r[2]
                # print("ee displacement:", ee_displacement[:xyz])
            else:
                ee_displacement[:xyz] = ee_displacement[:xyz] * self.ee_dis_ratio
                ee_displacement[xyz:-1] = ee_displacement[xyz:-1] * self.ee_rotxy_ratio
                ee_displacement[-1] = ee_displacement[-1] * self.ee_rotz_ratio
            # print(ee_displacement)
            # target_ee_position = np.clip(ee_displacement[:xyz], self.ee_position_low, self.ee_position_high)
            current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
            current_ee_pos = self.sim.get_site_position('attachment_site')
            current_ee_rot = R.from_matrix(self.sim.get_site_mat('attachment_site').reshape(3, 3)).as_euler('xyz', degrees=True)
            target_ee_pos = current_ee_pos + ee_displacement[:xyz]
            target_ee_rot = current_ee_rot + ee_displacement[xyz:]
            # print("current ee pos:", current_ee_pos)
            # print("current ee rot:", current_ee_rot)
            # print("target_ee_rot:", target_ee_rot)
            target_ee_rot = np.clip(target_ee_rot, self.ee_rot_low, self.ee_rot_high)
            # print(current_ee_rot)
            target_ee_pos = np.clip(target_ee_pos, self.ee_position_low, self.ee_position_high)
            target_ee_quat = R.from_euler('xyz', target_ee_rot, degrees=True).as_quat()
            # print("target ee quat:", target_ee_quat)
            # target_ee_pos = np.array([0.309, 0.08, 0.12192+0.87])
            # # self.y_rot += 0.00005
            # target_ee_quat = np.array([0.0, 0.0, 0.0, 1])
            if self.ee_admittancecontrol is True:
                # print("admittance control is True")
                target_ee_rot = target_ee_rot * self.d2r
                target_ee_pos, target_ee_rot = self.AdmittanceController(desired_pos=target_ee_pos, desired_rot=target_ee_rot)
                target_ee_rot = target_ee_rot / self.d2r
                target_ee_quat = R.from_euler('xyz', target_ee_rot, degrees=True).as_quat()
            target_arm_angles = self.inverse_kinematics(
                    current_joint=current_joint, 
                    target_position=target_ee_pos, 
                    target_orientation=target_ee_quat
                )
            # time.sleep(0.1)
            # print(target_arm_angles)
            return target_arm_angles
        else: # for real robot
            if self.dsl is True:
                current_ft = np.array(self.rtde_r.getActualTCPForce()) - self.real_init_ft_sensor
                for ft_xyz in range(3):
                    current_ft[ft_xyz] = 0 if self.ft_threashold_xyz > current_ft[ft_xyz] > -self.ft_threashold_xyz else current_ft[ft_xyz]
                    current_ft[ft_xyz+3] = 0 if self.ft_threashold_rxyz > current_ft[ft_xyz + 3] > -self.ft_threashold_rxyz else current_ft[ft_xyz + 3]
                current_ft[0] = -current_ft[0]
                current_ft[3] = -current_ft[3]
                current_ft = (current_ft - self.ft_mean) * self.ft_norm_scale + self.ft_norm_mean
                print("real init F/T:", self.real_init_ft_sensor)
                print("current ft (without norm and filtered):", current_ft)
                current_ft = self.ft_last * self.low_filter_gain + current_ft * (
                            1 - self.low_filter_gain)  # low-pass filter for F/T sensor
                # print("current ft norm and filted:", current_ft)
                self.ft_last = np.copy(current_ft)
                # red: - + + + - -
                # black: + - + + - -
                ee_dis_x = math.tanh(3 * current_ft[0]/10)
                ee_dis_y = -math.tanh(3 * current_ft[1]/10)
                ee_dis_z = math.tanh(3 * current_ft[2]/10)
                ee_dis_rx = -math.tanh(10 * current_ft[3])
                ee_dis_ry = math.tanh(10 * current_ft[4])
                ee_dis_rz = -math.tanh(10 * current_ft[5])
                dsl_ee_dis = np.array([ee_dis_x, ee_dis_y, ee_dis_z, ee_dis_rx, ee_dis_ry, ee_dis_rz])
                ee_displacement[:xyz] = ee_displacement[:xyz] * self.ee_dis_ratio + dsl_ee_dis[:xyz] * self.dsl_dampRatio_d
                ee_displacement[xyz:-1] = ee_displacement[xyz:-1] * self.ee_rotxy_ratio \
                                          + dsl_ee_dis[xyz:-1] * self.dsl_dampRatio_r[:2]
                ee_displacement[-1] = ee_displacement[-1] * self.ee_rotz_ratio + dsl_ee_dis[-1] * self.dsl_dampRatio_r[2]
                # print("ee displacement:", ee_displacement[:xyz])
            else:
                ee_displacement[:xyz] = ee_displacement[:xyz] * self.ee_dis_ratio
                ee_displacement[xyz:-1] = ee_displacement[xyz:-1] * self.ee_rotxy_ratio
                ee_displacement[-1] = ee_displacement[-1] * self.ee_rotz_ratio
            current_joint = np.array(self.rtde_r.getActualQ())

            current_ee_pos = np.around(np.array(self.rtde_r.getActualTCPPose()[:3]), 4)
            current_ee_pos[2] += self.fixedZoffsetwithSim # use the same range with sim, compensate the fixed high of table in simulation.
            current_ee_pos_from_urdf, current_ee_quat = self.forward_kinematics(qpos=current_joint)
            # print("current ee pos by URDF:", current_ee_pos_from_urdf)
            current_ee_rot = quaternion_to_euler(current_ee_quat[0],
                                                 current_ee_quat[1],
                                                 current_ee_quat[2],
                                                 current_ee_quat[3]) / self.d2r
            # print("current ee quat:", current_ee_quat)
            # print("current ee rot:", current_ee_rot)
            # print("current ee pos:", current_ee_pos[0], current_ee_pos[1], current_ee_pos[2] - self.fixedZoffsetwithSim)
            # print("current F/T (normed and filtered):", current_ft)
            # current_ee_rot = np.around(np.array(self.rtde_r.getActualTCPPose()[3:]), 4)
            target_ee_pos = current_ee_pos + ee_displacement[:xyz]
            target_ee_rot = current_ee_rot + ee_displacement[xyz:]
            target_ee_rot = np.clip(target_ee_rot, self.ee_rot_low, self.ee_rot_high)
            target_ee_pos = np.clip(target_ee_pos, self.ee_position_low, self.ee_position_high)
            target_ee_pos[2] -= self.fixedZoffsetwithSim # use the same range with sim, compensate the fixed high of table in simulation.
            target_ee_quat = R.from_euler('xyz', target_ee_rot, degrees=True).as_quat()
            # print("target ee pos:", target_ee_pos)
            # print("target ee rot:", target_ee_rot)
            # print("-------------")
            # !fixed the ee position for testing kinematic
            # target_ee_pos = np.array([0.309, 0.0801, 0.1817])
            # # self.y_rot += 0.00005
            # target_ee_quat = np.array([-3.2737539930006814e-05, 1.9495431819390594e-05, -0.15118763521715797, 0.9885050821850261])
            if self.ee_admittancecontrol is True:
                target_ee_rot = target_ee_rot * self.d2r
                target_ee_pos, target_ee_rot = self.AdmittanceController(desired_pos=target_ee_pos,
                                                                         desired_rot=target_ee_rot)
                target_ee_rot = target_ee_rot / self.d2r
                target_ee_quat = R.from_euler('xyz', target_ee_rot, degrees=True).as_quat()

            target_arm_angles = self.inverse_kinematics(
                    current_joint=self.rtde_r.getActualQ(),
                    target_position=target_ee_pos,
                    target_orientation=target_ee_quat,
                )
            # print("current joint:", self.rtde_r.getActualQ())
            # print("target pos:", target_ee_pos)
            # print("target orientation:", target_ee_quat)
            # print("target qpos:", target_arm_angles)
            target_arm_angles_original = np.copy(target_arm_angles)
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
        if self.real_robot is False:
            # ee_position = np.copy(self.get_ee_position())
            ee_position = self.sim.get_site_position('attachment_site')
            ee_rot = R.from_matrix(self.sim.get_site_mat('attachment_site').reshape(3, 3)).as_euler('xyz', degrees=True)
            # ee_position = (ee_position - self.ee_mean) * self.norm_scale + self.norm_mean
            # print("sim-ee-position:", ee_position)
            # ee_velocity = np.array(self.get_ee_velocity())
            # joint_angles = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
            # print(joint_angles)
            # joint_vels = np.array([self.sim.get_joint_velocity(joint=self.joint_list[i]) for i in range(6)])
            ft_sensor = self.sim.get_ft_sensor(force_site="ee_force_sensor", torque_site="ee_torque_sensor")
            ft_sensor = self.ft_last_obs * self.low_filter_gain + ft_sensor * (1 - self.low_filter_gain)
            self.ft_last_obs = np.copy(ft_sensor)
            # print(ft_sensor)
            # print("ee_pos:", ee_position)
            # print("ft sensor raw:", ft_sensor)
            # print("ee rot:", ee_rot)
            # print("---------")
            if self._normalize_obs is True:
                ee_position = (ee_position - self.ee_pos_mean) * self.ee_pos_norm_scale + self.ee_pos_norm_mean
                ee_rot = (ee_rot - self.ee_rot_mean) * self.ee_rot_norm_scale + self.ee_rot_norm_mean
                ft_sensor = (ft_sensor - self.ft_mean) * self.ft_norm_scale + self.ft_norm_mean
                # print("self.ft_norm_mean:", self.ft_mean)
                # print("ft sensor normalized:", ft_sensor)
                # print("---------")
            if self.vision_touch == "vision":
                obs = np.concatenate((ee_position, ee_rot))
            elif self.vision_touch == "vision-touch":
                obs = np.concatenate((ee_position, ee_rot, ft_sensor))
            return obs
        else: # for real robot
            # current_ee_pos = np.around(np.array(self.rtde_r.getActualTCPPose()[:3]), 4)
            # !calculate the ee rot from KDL-URDF
            current_qpos = np.array(self.rtde_r.getActualQ())
            _, current_ee_quat = self.forward_kinematics(qpos=current_qpos)
            current_ee_rot = quaternion_to_euler(current_ee_quat[0],
                                                 current_ee_quat[1],
                                                 current_ee_quat[2],
                                                 current_ee_quat[3]) / self.d2r
            current_ft = np.array(self.rtde_r.getActualTCPForce()) - self.real_init_ft_sensor
            current_ft[0] = -current_ft[0]
            current_ft[3] = -current_ft[3]
            # current_ft = np.array(self.rtde_r.getActualTCPForce())
            current_ft = self.ft_last_obs * self.low_filter_gain + current_ft * (1 - self.low_filter_gain)
            self.ft_last_obs = np.copy(current_ft)
            current_ee_pos = np.array([np.around(np.array(self.rtde_r.getActualTCPPose()[0]), 4),
                                       np.around(np.array(self.rtde_r.getActualTCPPose()[1]), 4),
                                       np.around(np.array(self.rtde_r.getActualTCPPose()[2]), 4) + self.fixedZoffsetwithSim])
            if self._normalize_obs is True:
                current_ee_pos = (current_ee_pos - self.ee_pos_mean) * self.ee_pos_norm_scale + self.ee_pos_norm_mean
                current_ee_rot = (current_ee_rot - self.ee_rot_mean) * self.ee_rot_norm_scale + self.ee_rot_norm_mean
                current_ft = (current_ft - self.ft_mean) * self.ft_norm_scale + self.ft_norm_mean
                print("current ee rotation (normalized):", current_ee_rot)
                print("current ee rot normalise params:", self.ee_rot_mean, self.ee_rot_norm_scale, self.ee_rot_norm_mean)
            if self.vision_touch == "vision":
                obs = np.concatenate((current_ee_pos, current_ee_rot))
                self.ft_record = np.r_[self.ft_record, [current_ft]]
                if self.ro is True:
                    np.save(
                        '/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/src/recording/' +self.running_time+ "force" + "_nodsl_" + '.npy', self.ft_record)

            elif self.vision_touch == "vision-touch":
                obs = np.concatenate((current_ee_pos, current_ee_rot, current_ft))
            return obs
        
    def reset(self) -> None:
        self.sim.reset()
        self.init_z_rot = 0.0
        self.set_joint_neutral()
        self.ft_last = np.array([0, 0, 0, 0, 0, 0])
        self.ft_last_obs = np.array([0, 0, 0, 0, 0, 0])
        self.init_ft_sensor = self.sim.get_ft_sensor(force_site="ee_force_sensor", torque_site="ee_torque_sensor")
        if self.ee_admittancecontrol is True:
            self._mat = np.zeros((3, 3))
            self.T_mat = np.zeros((3, 3))
        if self.real_robot is True:
            _items = 1000
            ft_record = np.zeros((_items, 6))
            acc_record = np.zeros((_items, 3))
            print("reset real robot pos and init the grasping for insertion.")
            # self._init_grasping_for_realRobot() # grasping program, comment out this code when debugging
            print("moveing ee to init pos ...")
            time.sleep(1)
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

            # while True:
            #     print(self.rtde_r.getActualTCPForce() - self.real_init_ft_sensor)
            #     time.sleep(0.01)
            #     pass
            init_acc = np.mean(acc_record, axis=0)
            self.AdmittanceControlLoop(init_acc=init_acc)

    def AdmittanceController(self, desired_pos, desired_rot) -> tuple[np.ndarray, np.ndarray]:
        desired_position = desired_pos
        desired_rotation = desired_rot
        # controller loop
        if self.real_robot is False:
            print("init ft:", self.init_ft_sensor)
            current_FT = self.sim.get_ft_sensor(force_site="ee_force_sensor", torque_site="ee_torque_sensor") - self.init_ft_sensor
        else:
            current_FT = self.rtde_r.getActualTCPForce() - self.real_init_ft_sensor
        updated_pos, update_rotation, self._mat, self.T_mat = self.admittance_controller.admittance_control(desired_position,
                                                                                                  desired_rotation,
                                                                                                  current_FT,
                                                                                                  self._mat,
                                                                                                  self.T_mat)
        return updated_pos, update_rotation


    def AdmittanceControlLoop(self, init_acc):
        desired_position = self.rtde_r.getActualTCPPose()[:3]
        ee_jpos_current = np.array(self.rtde_r.getActualQ())[-1]
        ee_jpos_target = ee_jpos_current / self.d2r
        target_z_rot = np.clip(ee_jpos_target, self.z_rot_low, self.z_rot_high)
        desired_rotation = np.array([-180 * self.d2r, 0, target_z_rot * self.d2r])
        init_vel = self.rtde_r.getActualTCPSpeed()[:3]
        ik_pos = np.zeros(3)
        _mat = np.zeros((3, 3))
        T_mat = np.zeros((3, 3))
        items = 5
        i = 0
        ft_record = [np.zeros(6)]
        current_joint = np.array(self.rtde_r.getActualQ())
        current_ee_pos_from_urdf, current_ee_quat = self.forward_kinematics(qpos=current_joint)
        desired_rotation = quaternion_to_euler(current_ee_quat[0],
                                             current_ee_quat[1],
                                             current_ee_quat[2],
                                             current_ee_quat[3])
        while True:
            i += 1
            current_joint = np.array(self.rtde_r.getActualQ())
            current_ee_pos_from_urdf, current_ee_quat = self.forward_kinematics(qpos=current_joint)


            current_FT = self.rtde_r.getActualTCPForce()
            FT_data = current_FT - self.real_init_ft_sensor
            # print("FT data:", FT_data)
            # print("current pos:", current_ee_pos_from_urdf)
            # print("current rot:", desired_rotation)
            # print("current quat:", R.from_euler('xyz', desired_rotation, degrees=True).as_quat())
            ft_record = np.r_[ft_record, [-FT_data]]
            # updated_pos, update_rotation, _mat, T_mat = self.admittance_controller.admittance_control(desired_position,
            #                                                                                           desired_rotation,
            #                                                                                           FT_data,
            #                                                                                           _mat,
            #                                                                                           T_mat)
            updated_pos, update_rotation = self.AdmittanceController(desired_pos=desired_position,desired_rot=desired_rotation)
            updated_pos = np.around(updated_pos, 5)
            update_rotation = update_rotation / self.d2r
            # print("update pos:", updated_pos)
            print("update rot:", update_rotation)
            # print("update quat:",  R.from_euler('xyz', update_rotation, degrees=True).as_quat())

            ee_jpos_current = np.array(self.rtde_r.getActualQ())[-1]
            ee_jpos_target = ee_jpos_current / self.d2r
            target_z_rot = np.clip(ee_jpos_target, self.z_rot_low, self.z_rot_high)
            target_quat = R.from_euler('xyz', update_rotation, degrees=True).as_quat()
            current_joint = np.array(self.rtde_r.getActualQ())
            target_arm_angles = self.inverse_kinematics(
                    current_joint=current_joint,
                    target_position=updated_pos,
                    target_orientation=target_quat
                )
            # print("target_arm_angles", target_arm_angles)
            # print("moved pos and quat:", self.forward_kinematics(qpos=target_arm_angles))
            # print("----------")

            self.real_joint_planner(grasping=True,
                                    _time=self.planner_time/50,
                                    current_joint=current_joint,
                                    target_joint=target_arm_angles)
            time.sleep(0.01)

    def log_info(self, gripper):
        print(f"Pos: {str(gripper.get_current_position()): >3}  "
            f"Open: {gripper.is_open(): <2}  "
            f"Closed: {gripper.is_closed(): <2}  ")
        
    def _init_grasping_for_realRobot(self):
        print("grapsing for real robot ...")
        # target1joint = np.array([2.28441358, -1.6466042,  1.94246418, -2.05400958, -1.55255968, -0.07506973])
        # target2joint = np.array([2.28441358, -1.41584693,  1.94246418, -2.05400958, -1.55255968, -0.07506973])
        # target1joint = np.array([3.7692291736602783, -1.5450309079936524, 1.704399887715475, -1.8301321468748988, -1.5807388464557093, -2.4963484446154993])
        target1joint = np.array([3.830038547515869, -1.699723859826559, 2.153346363698141, -2.357183118859762, -1.8035014311419886, -2.433138434086935])
        # target2joint = np.array([3.8441805839538574, -1.508592204456665, 2.0356009642230433, -2.378955980340475, -1.7434089819537562, -2.4464884440051478])
        # target2joint = np.array([3.8236942291259766, -1.469771222477295, 2.03717548051943, -2.422732015649313, -1.7380803267108362, -2.466470781956808])
        # target2joint = np.array([3.8618922233581543, -1.3786173623851319, 2.1968749205218714, -2.7827097378172816, -1.8232024351703089, -2.380178991948263])
        target2joint = np.array([3.767681121826172, -1.524015025501587, 2.229475800191061, -2.6224776707091273, -1.7844813505755823, -2.493746821080343])
        self.rtde_c.moveJ(target1joint, self.vel, self.acc)
        self.rtde_c.moveJ(target2joint, self.vel, self.acc)
        self.gripper.move_and_wait_for_pos(245, 255, 255)
        self.rtde_c.moveJ(target1joint, self.vel, self.acc)
        # self.gripper.move_and_wait_for_pos(250, 255, 255)

    def set_joint_neutral(self) -> None:
        self.set_joint_angles(self.neutral_joint_values)
        self.sim.step()
        self.control_joints(self.neutral_joint_values)

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
    
    def real_joint_planner(self, grasping: bool, _time: float, current_joint: np.ndarray, target_joint: np.ndarray) -> None:
        delta_joint = target_joint - current_joint
        abs_delta_joint = np.absolute(delta_joint)
        # print("delta qpos for real joint planner:", abs_delta_joint)
        # print("-------------")
        if np.any(abs_delta_joint > 0.3):
            delta_joint = np.zeros(6)
            print("delta joint is too big!!!!!!")
            print("current joint", current_joint)
            print("target joint", target_joint)
        
        planner_steps = 2
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
            # time.sleep(0.001)