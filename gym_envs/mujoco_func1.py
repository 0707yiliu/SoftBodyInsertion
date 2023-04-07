# MODEL: Universal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 04/10/2022
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

from difflib import restore
import os
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np
from mujoco_py import load_model_from_path, MjSim
import mujoco_py as mj
import gym_envs.envs.ur3_kdl as urkdl

import gym_envs.assets


class Mujoco_Func:
    # mujoco basic function
    def __init__(
        self, 
        render: bool = False, 
        n_substeps: int = 4, 
        background_color: Optional[np.ndarray] = None,
        vision_touch: str = 'vision',
        file_root: str = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/",
        hole_size: str = '2cm',
        real_robot: bool = False,
    ) -> None:
        if real_robot is True:
            urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_robot_real.urdf"
        else:
            urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_robot.urdf"
        self.urxkdl = urkdl.URx_kdl(urDH_file)

        if vision_touch == 'vision':
            xml_file = 'ur3gripper_triangle_in_eef_' + hole_size + '.xml'
            # # xml_file = 'ur3gripper_finger_fixedbox.xml'
            # if hole_size == '2cm':
            #     xml_file = 'ur3gripper_box_in_eef_2cm.xml'
            # elif hole_size == '1cm':
            #     xml_file = 'ur3gripper_box_in_eef_1cm.xml'
            # elif hole_size == '4mm':
            #     xml_file = 'ur3gripper_box_in_eef_4mm.xml'
            # elif hole_size == '4mm':
            #     xml_file = 'ur3gripper_box_in_eef_1mm.xml'
        elif vision_touch == 'vision-touch' or vision_touch == 'touch':
            xml_file = 'ur3gripper_triangle_in_eef_' + hole_size + '.xml'
            # xml_file = 'ur3gripper_box_in_eef_' + hole_size + '.xml'
            # # xml_file = 'ur3gripper_finger_box_hole.xml'
            # # xml_file = 'ur3gripper_box_in_eef.xml'
            # # xml_file = 'ur3gripper_finger_fixedbox.xml'
            # if hole_size == '2cm':
            #     xml_file = 'ur3gripper_box_in_eef_2cm.xml'
            # elif hole_size == '1cm':
            #     xml_file = 'ur3gripper_box_in_eef_1cm.xml'
            # elif hole_size == '4mm':
            #     xml_file = 'ur3gripper_box_in_eef_4mm.xml'
        file = file_root + xml_file
        background_color = background_color if background_color is not None else np.array([223.0, 54.0, 45.0])
        self.model = load_model_from_path(file)
        self.sim = MjSim(self.model)
        if render:
            self.viewer = mj.MjViewer(self.sim)
        self.n_substeps = n_substeps
        # if vision_touch == 'vision':
        #     self.n_substeps = int(n_substeps / 2)
        # elif vision_touch == 'vision-touch' or vision_touch == 'touch':
        #     self.n_substeps = n_substeps
        self.timestep = 1.0 / 500
        self.render = render 
    
    @property
    def dt(self):
        return self.timestep * self.n_substeps
    
    def reset(self) -> None:
        self.sim.reset()

    def step(self) -> None:
        for _ in range(self.n_substeps):
            self.sim.step()
            if self.render is True:
                self.viewer.render()
    
    def get_body_position(self, body: str) -> np.ndarray:
        position = self.sim.data.get_body_xpos(body)
        return np.array(position)

    def get_body_quaternion(self, body: str) -> np.ndarray:
        quat = self.sim.data.get_body_xquat(body) # w x y z
        return np.array(quat)
    
    def get_body_velocity(self, body: str) -> np.ndarray:
        vel = self.sim.data.get_body_xvelp(body)
        return np.array(vel)
    
    def get_body_angular_velocity(self, body: str) -> np.ndarray:
        angular_vel = self.sim.data.get_body_xvelp(body)
        return np.array(angular_vel)
    
    def get_joint_angle(self, joint: str) -> float:
        return self.sim.data.get_joint_qpos(joint)

    def get_touch_sensor(self, sensor: str) -> float:
        return self.sim.data.get_sensor(sensor)
    
    def get_ft_sensor(self) -> np.ndarray:
        return self.sim.data.sensordata
    
    def get_joint_velocity(self, joint: str) -> float:
        return self.sim.data.get_joint_qvel(joint)
    
    def get_site_position(self, site: str) -> np.ndarray:
        return self.sim.data.get_site_xpos(site)

    def set_joint_angles(self, joints: np.ndarray, angles: np.ndarray) -> None:
        sim_state = self.sim.get_state()

        for i in range(len(angles)):
            sim_state.qpos[joints[i]] = angles[i]
        self.sim.set_state(sim_state)

    def set_mocap_pos(self, mocap: str, pos: np.ndarray) -> None:
        self.sim.data.set_mocap_pos(mocap, pos)

    def set_mocap_quat(self, mocap: str, quat: np.ndarray) -> None:
        self.sim.data.set_mocap_quat(mocap, quat)
    
    def control_joints(self, joint_index: np.ndarray, target_angles: np.ndarray) -> None:
        for i in range(len(target_angles)):
            self.sim.data.ctrl[joint_index[i]] = target_angles[i]
    
    def inverse_kinematics(self, current_joint: np.ndarray, target_position: np.ndarray, target_orientation: np.ndarray) -> np.ndarray:
        qpos = self.urxkdl.inverse(current_joint, target_position, target_orientation)
        return qpos
    
    def forward_kinematics(self,qpos) -> np.ndarray:
        ee_pos = self.urxkdl.forward(qpos=qpos)
        return ee_pos

    def set_forward(self) -> None:
        self.sim.forward()

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        pass
