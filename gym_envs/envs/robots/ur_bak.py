# MODEL: Univewrsal Robots UR3 + Robotiq 2F-85 env
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
from gym_envs.mujoco_func import Mujoco_Func
from gym_envs.utils import distance, normalizeVector, euler_to_quaternion
import time

class UR3(MujocoRobot):
    def __init__(
        self,
        sim: Mujoco_Func,
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
    ) -> None:
        self.matching_shape = match_shape # add z-axis rotation
        self.ee_init_pos = ee_init
        self.ee_init_pos[2] += z_offset
        self.init_ft_sensor = np.array([0.34328045,  2.12207036, -9.31005002,  0.27880092,  0.04346038,  0.01519237])
        # self.ee_pos = ee_init
        # self.ee_pos[2] += z_offset
        self._normalize_obs = _normalize
        self.gripper_max_joint = gripper_max_joint
        self.vision_touch = vision_touch
        self.ee_dis_ratio = ee_dis_ratio/40 if self.vision_touch == 'vision' or 'vision-touch' else ee_dis_ratio/65
        self.gripper_action_ratio = gripper_action_ratio/40 if self.vision_touch =='vision' else gripper_action_ratio/20
        self.random_high = 0.045 if self.vision_touch == 'vision' else 0.045
        self.random_low = -0.045 if self.vision_touch == 'vision' else -0.045#0.014
        self.random_lim_high = 0.025 if self.vision_touch == 'vision' else 0.025
        self.random_lim_low = -0.025 if self.vision_touch == 'vision' else -0.025
        # self.random_high = 0.014 if self.vision_touch == 'vision' else 0.014
        # self.random_low = -0.014 if self.vision_touch == 'vision' else -0.014
        # self.random_lim_high = 0.004 if self.vision_touch == 'vision' else 0.004
        # self.random_lim_low = -0.004 if self.vision_touch == 'vision' else -0.004
        self.joint_dis_ratio = joint_dis_ratio
        self.ee_position_low = ee_positon_low if ee_positon_low is not None else np.array([-0.35, 0.3, 0.88])
        self.ee_position_high = ee_positon_high if ee_positon_high is not None else np.array([0.35, 0.38, 1.6])
        self.gripper_joint_low = gripper_joint_low if gripper_joint_low is not None else 0.3
        self.gripper_joint_high = gripper_joint_high if gripper_joint_high is not None else 0.47
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.planner_time = planner_time
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 6 # ur-ee control / ur-joint control
        # n_action += 0 if self.block_gripper else 1 # replace by self.vision_touch
        n_action += 0 if self.vision_touch == 'vision' else 1
        n_action += 1 if self.matching_shape is True else 0
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
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
            self.z_rot_high = 60.0
            self.z_rot_low = -60.0
            # print(n_action)
        #! -----------------------------------
    
    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()
        # print(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            if self.matching_shape is False:
                ee_displacement = action[:3]
            else:
                ee_displacement = action[:4]
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
        if self.vision_touch == 'vision':
            target_fingers_width = 0.1
        else:
            fingers_ctrl = action[-1] * self.gripper_action_ratio
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width / 2 + fingers_ctrl
            target_fingers_width = np.clip(target_fingers_width, self.gripper_joint_low, self.gripper_joint_high)
        target_angles = np.concatenate((target_arm_angles, [target_fingers_width]))
        if self.vision_touch == 'vision':
            # self.control_joints(target_angles=target_angles)
            current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
            current_joint_gripper = np.array(self.get_fingers_width()/2)
            # current_joint_gripper = np.clip(current_joint_gripper, self.gripper_joint_low, self.gripper_joint_high)
            current_joint = np.concatenate((current_joint_arm, [current_joint_gripper]))
            self.joint_planner(grasping=False, _time=self.planner_time/50, current_joint=current_joint, target_joint=target_angles)
        else:
            current_joint_arm = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
            current_joint_gripper = np.array(self.get_fingers_width()/2)
            # current_joint_gripper = np.clip(current_joint_gripper, self.gripper_joint_low, self.gripper_joint_high)
            current_joint = np.concatenate((current_joint_arm, [current_joint_gripper]))
            self.joint_planner(grasping=False, _time=self.planner_time/50, current_joint=current_joint, target_joint=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        ee_displacement[:3] = ee_displacement[:3] * self.ee_dis_ratio
        # print(ee_displacement)
        self.ee_init_pos = self.ee_init_pos + ee_displacement[:3] # use for orignal program
        # ---------- testing kinematic function --------------
        # self.ee_init_pos = self.ee_init_pos
        # target_ee_position = np.clip(self.ee_init_pos, self.ee_position_low, self.ee_position_high)
        # current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        # -----------------------------------------------------
        # target_ee_position = np.array([-0.35, 0.3, 1]) # for testing pid controller
        # print(self.get_body_position("eef")) # for testing pid controller
        #! the constrain of ee position
        target_ee_position = np.clip(self.ee_init_pos, self.ee_position_low, self.ee_position_high) # use for orignal program
        current_joint = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)]) # use for orignal program
        # print("-----------------------:", current_joint)
        # print(current_joint)
        if self.matching_shape is True:
            ee_displacement[3] = ee_displacement[3] * 1
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
            self.init_z_rot = self.init_z_rot + ee_displacement[3]
            # target_z_rot = np.copy(self.init_z_rot)
            target_z_rot = np.clip(self.init_z_rot, self.z_rot_low, self.z_rot_high)
            target_quat = euler_to_quaternion(-180 * self.d2r, 0, target_z_rot * self.d2r)
            target_quat = np.roll(target_quat, 3)
            # print(current_joint, target_z_rot)
            # target_quat = np.array([-1.0, 0.0, 0.0, 0.0])
            target_arm_angles = self.inverse_kinematics(
                current_joint=current_joint, target_position=target_ee_position, target_orientation=target_quat
            )
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
                current_joint=current_joint, target_position=target_ee_position, target_orientation=target_quat
            )
            
        # print('target_angles:', target_arm_angles)
        return target_arm_angles

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
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        joint_angles = np.array([self.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])
        joint_vels = np.array([self.sim.get_joint_velocity(joint=self.joint_list[i]) for i in range(6)])
        # ee_quat = self.sim.get_body_quaternion(self.ee_body)
        # print(ee_quat)
        ft_sensor = self.sim.get_ft_sensor().copy()
        ft_sensor_current = self.init_ft_sensor - ft_sensor
        for i in range(6):
            if ft_sensor_current[i] >= 100:
                ft_sensor_current[i] = 100
            elif ft_sensor_current[i] <= -100:
                ft_sensor_current[i] = -100
            ft_sensor_current[i] = ft_sensor_current[i] / 50
        if self._normalize_obs is True:
            # ee_position = normalizeVector(data=ee_position)
            # ee_velocity = normalizeVector(data=ee_velocity)
            joint_vels = normalizeVector(data=joint_vels)
        if self.vision_touch == 'vision':
            obs = np.concatenate((ee_position, ee_velocity))
            # obs = np.copy(joint_vels)
            # print("ee_force_sensor:", self.sim.get_ft_sensor())
            # # print("ee_torque_sensor:", self.sim.get_touch_sensor('ee_torque_sensor'))
            # print("-----------------------------")
        elif self.vision_touch == 'vision-touch':
            fingers_width = self.get_fingers_width() / 2 / self.gripper_max_joint
            # touch_sensors = np.array([self.sim.get_touch_sensor(sensor=self.sensor_list[i]) for i in range(self.sensor_num)])
            # if self._normalize_obs is True:
                # touch_sensors = normalizeVector(data=touch_sensors, min=0, max=1)
            # obs = np.concatenate((ee_position, ee_velocity, joint_angles, touch_sensors, [fingers_width]))
            # obs = np.concatenate((joint_vels, touch_sensors, [fingers_width]))
            # obs = np.concatenate((ee_position, ee_velocity, [fingers_width], ft_sensor_current))
            obs = np.concatenate((ee_position, ee_velocity, ft_sensor_current, [joint_angles[-1]]))
        # recording_ur_obs = np.r_[[recording_ur_obs], [obs]]
        elif self.vision_touch == 'touch':
            fingers_width = self.get_fingers_width() / 2 / self.gripper_max_joint
            # touch_sensors = np.array([self.sim.get_touch_sensor(sensor=self.sensor_list[i]) for i in range(self.sensor_num)])
            # if self._normalize_obs is True:
                # touch_sensors = normalizeVector(data=touch_sensors, min=0, max=1)
            # obs = np.concatenate((ee_position, ee_velocity, joint_angles, touch_sensors, [fingers_width]))
            # obs = np.concatenate((joint_vels, touch_sensors, [fingers_width]))
            # obs = np.concatenate((ee_position, ee_velocity, [fingers_width], ft_sensor_current))
            obs = np.concatenate((ee_position, ee_velocity, ft_sensor_current))

        return obs

    def reset(self) -> None:
        self.sim.reset()
        self.set_joint_neutral()

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
        _goal = np.array([-0.05, 0.31, 0.93]) # 2-method: fixed the init eef pos and randomize the hole in pih.py
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
        self.set_joint_angles(angles=target_arm_angles)
        self.set_forwad()
        # for i in range(10):
        #     self.init_ft_sensor = self.sim.get_ft_sensor().copy()
        #     self.sim.step()
        # print(self.ee_init_pos, self.sim.get_site_position('ee_tool'))


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
        

