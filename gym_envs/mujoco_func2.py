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
from scipy.spatial.transform import Rotation as R

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
        domain_randomization: bool = False,
        ur_gen: int = 3,
    ) -> None:
        if real_robot is True:
            if ur_gen == 5:
                urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur5e_gripper/ur5e_gripper_real.urdf"
            elif ur_gen == 3:
                urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_robot_real.urdf"
        else:
            if ur_gen == 5:
                urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur5e_gripper/ur5e_gripper.urdf"
            elif ur_gen == 3:
                urDH_file = "/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3_robot.urdf"
        self.urxkdl = urkdl.URx_kdl(urDH_file)
        # for mujoco
        if ur_gen == 5:
            xml_file = 'ur5e_gripper/scene.xml'
        elif ur_gen == 3:
            xml_file = 'ur3_pih_curved.xml'
        # if vision_touch == 'vision':
        #     xml_file = 'ur5e_gripper/scene.xml'
        # elif vision_touch == 'vision-touch' or vision_touch == 'touch':
        #     xml_file = 'ur5e_gripper/scene.xml'
        
        self.xml_file = file_root + xml_file
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        if render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.render = render

        self.domain_randomization = domain_randomization
        self.file_root = file_root
        self.tool_stiffness_range = np.array([0.01, 0.1])
        self.tool_damper_range = np.array([0.005, 0.05])

        self.n_substeps = 10
        self.timestep = 0.001

        self.ur_gen = ur_gen

    def reload_xml(self):
        try:
            import xml.etree.cElementTree as ET
        except ImportError:
            import xml.etree.ElementTree as ET
        if self.ur_gen == 5:
            tool_xml = self.file_root + 'ur5e_gripper/ur5e_with_gripper_softTool.xml'
        elif self.ur_gen == 3:
            tool_xml = self.file_root + 'ur3_with_grippersoftTool.xml'
        tree = ET.parse(tool_xml)
        root = tree.getroot()
        # domain randomization for tool's stiffness and damper
        tool_stiffness = str(np.random.uniform(low=self.tool_stiffness_range[0], high=self.tool_stiffness_range[1]))
        tool_damper = str(np.random.uniform(low=self.tool_damper_range[0], high=self.tool_damper_range[1]))
        # file writing
        i = 0
        for student in root.iter('joint'):
            i += 1
            if i == 2:
                student.set("stiffness", tool_stiffness)
                student.set("damping", tool_damper)
                # student.attrib["stiffness"] = 0.08
                # print(student.attrib)
        if self.ur_gen == 3:
            tree.write(self.file_root+'ur3_with_grippersoftTool.xml')
        elif self.ur_gen == 5:
            tree.write(self.file_root + 'ur5e_gripper/ur5e_with_gripper_softTool.xml')
        time.sleep(0.01) # waiting file writing
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        if self.render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    @property
    def dt(self):
        return self.timestep * self.n_substeps

    def reset(self) -> None:
        if self.domain_randomization is True:
            self.reload_xml()
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

    def get_site_mat(self, site: str) -> np.ndarray:
        return self.data.site_xmat[mujoco.mj_name2id(self.model, type=6, name=site)]

    def set_joint_angles(self, angles: np.ndarray) -> None:
        for i in range(len(angles)):
            self.data.qpos[i] = angles[i]
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
        pass#

#
# test_env = Mujoco_Func()
#
# i = 0
# # fw_qpos = np.array([1.53, -1.53, 1.53, -1.53, -1.53, -1.57079633])
# # fw_qpos = np.array([1.19274333, -1.24175816,  1.74942402, -1.02642832, -1.54568981, 0])
# fw_qpos = np.array([0.78781694, -1.42926988,  1.16459104, -1.30611748, -1.57079633, -0.78297938])
# joint_name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'right_driver_joint']
# current_arm_joint = np.zeros(6)
# test_env.reset()
# test_env.set_joint_angles(fw_qpos)
# while i < 5000:
#     i += 1
#     test_env.step()
#     test_env.control_joints(fw_qpos)
#     for j in range(6):
#         current_arm_joint[j] = np.copy(test_env.get_joint_angle(joint_name[j]))
#     r = R.from_matrix(test_env.get_site_mat('attachment_site').reshape(3, 3))
#     # r = test_env.get_body_quaternion("wrist_3_link")
#     site_pos = test_env.get_site_position("attachment_site")
#     print("---------------")
#     print("site pos:", site_pos)
#     print("sim qua:", r.as_quat())
#     # print("ft data:", test_env.get_ft_sensor(force_site="ee_force_sensor", torque_site="ee_torque_sensor"))
#     # print("sim tool:", test_env.get_site_position('obj_bottom'))
#     print("fw:", test_env.forward_kinematics(fw_qpos))
#     current_ee_rot = R.from_matrix(test_env.get_site_mat('attachment_site').reshape(3, 3)).as_euler('xyz', degrees=True)
#     print(current_ee_rot)
#     print("ik qpos:", test_env.inverse_kinematics(current_arm_joint, np.array([0.15, 0.31, 1.22]), np.array([0, 0, 0, 1])))
#     # print(test_env.inverse_kinematics(current_arm_joint, [0.075, 0.575, 0.9], r.as_quat()))
#     current_ft = np.copy(test_env.get_ft_sensor(force_site="ee_force_sensor", torque_site="ee_torque_sensor"))
#     print("ft sensor data:", current_ft)
#     if i > 2000:
#         test_env.reset()
#         test_env.set_joint_angles(fw_qpos)
#         i = 0
#

