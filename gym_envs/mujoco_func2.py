# MODEL: Universal Robots UR3 + Robotiq 2F-85 env on Mujoco > 2.1 (the new Mujoco Source)
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
#
import numpy as np
import gym_envs.envs.ur3_kdl as urkdl

import gym_envs.assets
import mediapy as media
import mujoco
import mujoco_viewer
# import mujoco.viewer as render

# import subprocess
# if subprocess.run('nvidia-smi').returncode:
#   raise RuntimeError(
#       'Cannot communicate with GPU. '
#       'Make sure you are using a GPU Colab runtime. '
#       'Go to the Runtime menu and select Choose runtime type.')
# # Configure MuJoCo to use the EGL rendering backend (requires GPU)
# print('Setting environment variable to use GPU rendering:')
# # %env MUJOCO_GL=egl

# ur5path = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/universal_robots_ur5e/scene.xml"
# path = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_pih_curved.xml"
# model = mujoco.MjModel.from_xml_path(path)
# data = mujoco.MjData(model)
# mujoco.mj_resetData(model, data)
# # mujoco.mj_forward(model, data)
# # Make renderer, render and show the pixels
# # renderer = mujoco.Renderer(model)
# # media.show_image(renderer.render())
# viewer = mujoco_viewer.MujocoViewer(model, data)
# i = 0
# while i < 5000:
#     i += 2
#     if viewer.is_alive:
#         mujoco.mj_step(model, data)
#         # print(data.qpos[mujoco.mj_name2id(model, type=mujoco.mjtObj.mjOBJ_JOINT, name="right_driver_joint")])
#         # print(data.xpos[mujoco.mj_name2id(model, type=mujoco.mjtObj.mjOBJ_XBODY, name="wrist_3_link")])
#         # print(data.actuator("fingers_actuator").ctrl)
#         # print(model.body_mocapid)
#         # print(data.sensor('ee_force_sensor').data)
#         # for j in range(6):
#         #     data.ctrl[j] = i / 2000
#         data.ctrl[0] = 0.8
#         data.ctrl[6] = i / 10
#         viewer.render()
#         if i > 2000:
#             mujoco.mj_resetData(model, data)
#             i = 0
#             data.qpos[0] = 0.8
#             mujoco.mj_forward(model, data)
#     else:
#         break
#
# # close
# viewer.close()

#
class Mujoco_Func:
    # mujoco basic function
    def __init__(
        self,
        render: bool = True,
        vision_touch: str = 'vision',
        file_root: str = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/",
        hole_size: str = "4mm",
        real_robot: bool = False,
    ) -> None:
        if real_robot is True:
            urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_robot_real.urdf"
        else:
            urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_robot.urdf"
        self.urxkdl = urkdl.URx_kdl(urDH_file)
        if vision_touch == 'vision':
            xml_file = 'ur3_pih_curved.xml'
        elif vision_touch == 'vision-touch' or vision_touch == 'touch':
            xml_file = 'ur3_pih_curved.xml'
        file = file_root + xml_file
        self.model = mujoco.MjModel.from_xml_path(file)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        if render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.render = render

        self.n_substeps = 1
        self.timestep = 0.001

    @property
    def dt(self):
        return self.timestep * self.n_substeps

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        if self.render is True:
            self.viewer.render()

    def get_body_position(self, body: str) -> np.ndarray:
        position = self.data.xpos[mujoco.mj_name2id(self.model, type=1, name=body)]
        return np.array(position)

    def get_body_quaternion(self, body: str) -> np.ndarray:
        quat = self.data.xquat[mujoco.mj_name2id(self.model, type=1, name=body)]
        return np.array(quat)

    def get_body_velocity(self, body: str) -> np.ndarray:
        vel = self.data.cvel[mujoco.mj_name2id(self.model, type=1, name=body)]
        return np.array(vel)

    def get_joint_angle(self, joint: str) -> float:
        return self.data.qpos[mujoco.mj_name2id(self.model, type=3, name=joint)]

    def get_joint_velocity(self, joint: str) -> float:
        return self.data.qvel[mujoco.mj_name2id(self.model, type=3, name=joint)]

    def get_site_position(self, site: str) -> np.ndarray:
        return self.data.site_xpos[mujoco.mj_name2id(self.model, type=6, name=site)]

    def set_joint_angles(self, joints: np.ndarray, angles: np.ndarray) -> None:
        for i in range(len(angles)):
            self.data.qpos[joints[i]] = angles[i]
        mujoco.mj_forward(self.model, self.data)

    def set_mocap_pos(self, mocap: str, pos: np.ndarray) -> None:
        self.data.mocap_pos[0] = pos # TODO:the id is not defined in mujoco, you should design a search method
        # self.data.mocap_pos[mujoco.mj_name2id()]

    def set_mocap_quat(self, mocap: str, quat: np.ndarray) -> None:
        self.data.mocap_quat[0] = quat # TODO: the same problem like set_mocap_pos func

    def control_joints(self, target_angles: np.ndarray) -> None:
        for i in range(len(target_angles)):
            self.data.ctrl[i] = target_angles[i]

    def set_forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    # def get_touch_sensor(self, sensor: str) -> float:
    #     return self.data.sensor(sensor)

    def get_ft_sensor(self, force_site: str, torque_site: str) -> np.ndarray:
        force = self.data.sensor(force_site).data
        torque = self.data.sensor(torque_site).data
        return np.hstack((force, torque))

    def inverse_kinematics(self, current_joint: np.ndarray, target_position: np.ndarray, target_orientation: np.ndarray) -> np.ndarray:
        qpos = self.urxkdl.inverse(current_joint, target_position, target_orientation)
        return qpos

    def forward_kinematics(self, qpos) -> np.ndarray:
        ee_pos = self.urxkdl.forward(qpos=qpos)
        return ee_pos

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        pass

# test_env = Mujoco_Func()
#
# i = 0
# while i < 5000:
#     i += 1
#     test_env.step()
#     test_env.control_joints([0.1, 0.2, 0.3, 0.4])
#     if i > 1000:
#         test_env.reset()
#         i = 0

