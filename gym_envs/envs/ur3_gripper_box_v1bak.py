
# MODEL: Univewrsal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

from curses import flash
from types import DynamicClassAttribute
from xml.sax.handler import property_interning_dict
from mujoco_py import load_model_from_path, MjSim
import mujoco_py as mj

import math
import random
import time
import numpy as np
import cv2
import logging
import gym
import matplotlib as mpl
import os
import copy

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.mujoco import mujoco_env
from scipy.spatial.transform import Rotation as R

import ur3_kdl_func


initial_pos = [90 / 180 * math.pi, -90 / 180 * math.pi, 90 / 180 * math.pi,
               -90 / 180 * math.pi, -90 / 180 * math.pi, 0,
               0, 0, 0, 0, 0, 0, 0, 0] # ur3 + robotiq 2F-85
joint_limit_lower = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
joint_limit_upper = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
action_limit_lower = np.array([-1, -1, -1])
action_limit_upper = np.array([1, 1, 1])
action_dv = 0.001

# print(action_limit_lower.shape)

fktestpose = [-1.5, -0.3, 0.3, 0.3, 0.3, -0.3] # testing
obs_dim = 29 # 12 + 6 + 3 + 1 + 7 
action_dim = 3
weight_obj_hole_z_axis = 0.7
weight_touch_tip = 0.65
z_quat = [0, 1,  0,  0] #  wxyz
# ------------- hard code --------------------------


class ur3_gripper_box_env_v1(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(
        self,
        file="/home/yi/peg_in_hole/ur3_rl_sim2real/gym_envs/models/ur3gripper_finger_box_hole.xml",
        render=False
    ):
        # loading model from xml file
        self.model = load_model_from_path(file)
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        if render is True:
            self.viewer = mj.MjViewer(self.sim)
            self.is_render = True
        else:
            self.is_render = False

        self.done = False
        self.obs_joint = np.zeros(6)
        self.obs = np.zeros(obs_dim)
        self.goal_pos = np.zeros(3)
        self.goal_rot = np.zeros(3)
        self.goal_quat = np.zeros(4)
        self.right_finger_side = np.zeros(4)
        self.left_finger_side = np.zeros(4)
        self.eef_pos = np.zeros(3)
        self.eef_quat = np.zeros(4)
        self.goal_quat[0] = 1
        self.reward = -10
        self.iter_num = 0
        self.max_iter = 3000

        self.current_joint = np.zeros(6)
        self.current_eef_pos = np.zeros(3)
        
        self.joint_list = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_num = len(self.joint_list)
        self.tac_sensor_list = ["touchsensor_r1", "touchsensor_r2", "touchsensor_r3", "touchsensor_r4", "touchsensor_r5", "touchsensor_r6", 
                        "touchsensor_l1", "touchsensor_l2", "touchsensor_l3", "touchsensor_l4", "touchsensor_l5", "touchsensor_l6"]
        self.tac_sensor_num = len(self.tac_sensor_list)
        self.gripper_joint_list = ["right_driver_1"]
        self.gripper_joint_num = len(self.gripper_joint_list)
        self.hole_list = ["box"]
        self.hole_num = len(self.hole_list)
        self.obj_list = ["cylinder_obj"]
        self.action_space = spaces.Box(low=action_limit_lower, high=action_limit_upper, shape=(3,), dtype=np.float64)
        self.observation_space = spaces.Box(-10, 10, shape=(obs_dim,), dtype=np.float64)
        self.seed()
        self.testing = 1
        # print("---------------------", self.action_space)
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        dx = action[0] * action_dv
        dy = action[1] * action_dv
        dz = action[2] * action_dv
        self.iter_num += 1
        self.done = False
        dx = -0.001
        dy = -0.001
        dz = -0.001
        # print("delta_xyz:", dx, dy, dz)
        # ur3_kdl_func.forward(qpos)
        # goal_rot = [-0.09021003, -0.21542238,  1.55788809]
        # goal_pos = [0.221516,   -0.429323,     1.00631]
        # goal_rot = [0.7011962630600858, -0.04493421368790836, -0.10734956869169622, 0.7034065589774718]
        # self.get_observation()
        # qpos = joint_limitation(ur3_kdl_func.inverse(goal_pos, goal_rot), joint_limit_lower, joint_limit_upper)
        # print(qpos)
        # # self.sim.data.ctrl[0] = remap(action[0], -1, 1, -30 / 180 * math.pi, 45 / 180 * math.pi)
        # # self.sim.data.ctrl[1] = remap(action[1], -1, 1, -105 / 180 * math.pi, -50 / 180 * math.pi)
        # # self.sim.data.ctrl[2] = remap(action[2], -1, 1, 0 / 180 * math.pi, 180 / 180 * math.pi)
        #! -------------------- initial pos ------------------
        # print("touch_sensor ----------------------")
        # for i in range(self.tac_sensor_num):
        #     print(self.sim.data.get_sensor(self.tac_sensor_list[i]))
            # self.qpos = joint_limitation(ur3_kdl_func.inverse(obj_init_pos, self.goal_quat), joint_limit_lower, joint_limit_upper)
        #! use KDL ik and fk control robot in mujoco.
        if self.step_one is True:
            for i in range(6):
                self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
            self.current_joint = self.obs_joint
            self.current_eef_pos =  ur3_kdl_func.forward(self.obs_joint)
            self.step_one = False
        self.goal_pos = [self.current_eef_pos[0] + dx, self.current_eef_pos[1] + dy, self.current_eef_pos[2] + dz]
        self.qpos = joint_limitation(ur3_kdl_func.inverse(self.current_joint, self.goal_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
        self.current_joint = copy.deepcopy(self.qpos)
        self.current_eef_pos = ur3_kdl_func.forward(self.qpos)
        # print("next_pos:", self.current_eef_pos)
        # print("next_joint:", self.current_joint)
        # #! ------------------ generating new action for env ---------------------------
        # #! getting the UR3-joint statement and the obj pos, statement + action[]
        # # self.eef_quat = self.sim.data.get_body_xquat("eef")
        # # print(action)
        # for i in range(3):
        #     self.goal_pos[i] = np.around(self.sim.data.get_body_xpos("ee_link")[i], 3)
        # print("last_goal:", self.goal_pos)
        # # self.goal_pos[2] = self.goal_pos[2] 
        # # print(self.sim.data.get_body_xpos("eef"))
        # # print(self.sim.data.get_body_xpos("ee_link"))
        # # print(self.goal_pos)
        # # for i in range(4):
        # #     self.goal_quat[i] = self.obs[i+3+self.joint_num+self.tac_sensor_num+self.gripper_joint_num+self.hole_num * 3]
        # # self.goal_rot = np.array(R.from_quat([self.goal_quat[1], self.goal_quat[2], self.goal_quat[3], self.goal_quat[0]]).as_euler('zyx', degrees=False))
        # # self.gripper_joint = self.obs[self.joint_num+self.tac_sensor_num]
        # # self.get_statement()

        # # # TODO: the actions are incremental, so they need to be added into observation above
        # # #! actions: delta-eef-pos(3-dim) delta-eef-rot(3-dim) delta-gripper-joint(1-dim)
        # self.goal_pos = [self.goal_pos[0] + dx, self.goal_pos[1] + dy, self.goal_pos[2] + dz]
        
        # # print(self.sim.data.get_body_xpos("ee_link"))
        # print("present_goal", self.goal_pos)
        # for i in range(6):
        #     self.obs_joint[i] = np.around(self.sim.data.get_joint_qpos(self.joint_list[i]), 4)
        # print("last_joint:", self.obs_joint)
        # # ur3_kdl_func.forward(self.obs_joint)
        # #     self.goal_rot[i] += action[i+3]
        # # self.gripper_joint += action[-1]
        # # if self.gripper_joint > 0.7:
        # #     self.gripper_joint = 0.7
        # # if self.gripper_joint <= 0:
        # #     self.gripper_joint = 0
        # # self.goal_quat = np.array(R.from_euler('zyx', [self.goal_rot[0], self.goal_rot[1], self.goal_rot[2]], degrees=False).as_quat())
       
        # # print(np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.goal_pos))
        
        # self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.goal_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
        # print("present joint:", self.qpos)

        #! controlling -----------------------------
        for i in range(6):
            self.sim.data.ctrl[i] = self.qpos[i]
        # print("ik----")
        # ur3_kdl_func.forward(self.qpos)
        # print(self.goal_pos)
        self.sim.data.ctrl[6] = 0.4
        self.sim.step()
        # print(self.sim.data.get_body_xpos("ee_link"))
        # time.sleep(0.02)
        # print(np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.goal_pos) )

        # ik_num = 0
        # self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.goal_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
        # for i in range(6):
        #     self.sim.data.ctrl[i] = self.qpos[i]
        # self.sim.data.ctrl[6] = 0.48
        # self.sim.step()
        # while np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.goal_pos) > 0.003:
        #     print(np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.goal_pos) )
        #     self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.goal_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
        #     for i in range(6):
        #         self.sim.data.ctrl[i] = self.qpos[i]
        #     self.sim.data.ctrl[6] = 0.48
        #     ik_num += 1
        #     self.sim.step()
        #     self.viewer.render()
        #     if ik_num > 3000:
        #         self.done = True
        #         self.reward = -100
        #         self.get_observation()
        #         self.info = {}
        #         return self.obs, self.reward, self.done, self.info
        # ik_num = 0
        # print(self.qpos)
        # print("base_link pos:", self.sim.data.get_body_xpos("base_link"))
        # print("mujoco ee_link pos:", self.sim.data.get_body_xpos("ee_link"))
        # print("gripper end-effector pos:", self.sim.data.get_body_xquat("eef"))
        # print("tactile sensor:", self.sim.data.get_sensor("touchsensor_r1"))
        # for i in range(self.joint_num):
        #     self.obs[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
        # self.sim.data.ctrl[6] = 0.5 # finger-joint controller
        # print("action:", action)
        
        self.get_observation()
        self.get_reward()
        # if self.reward != 0:
        #     self.done = True
        # elif self.iter_num > self.max_iter:
        #     self.reward = -0.1
        #     self.done = True
        # else:
        #     self.done = False

        if self.iter_num > self.max_iter:
            self.reward = -1
            self.done = True
        elif self.reward > 0:
            self.done = True
        else:
            self.done = False
        # print(self.reward)
        # if self.reward > 60:
        #     self.done = True
        # if np.linalg.norm(self.sim.data.get_body_xpos("eef") - self.sim.data.get_body_xpos("cylinder_obj")) > 0.05:
        #     self.done = True
        #! TODO: the finger will press the desk and produce large number, which can not be added into the reward function.
        # print(self.reward)
        # self.done = False
        # print(self.done)
        # print(self.iter_num)
        
        # print(self.sim.data.get_body_xquat("eef"))
        # print(self.obs)
        # print(self.reward)
        self.info = {}
        # self.viewer.render()
        if self.is_render is True:
            self.viewer.render()
        return self.obs, self.reward, self.done, self.info

    def reset(self):
        self.sim.reset()
        self.step_one = True
        self.reset_flag = True
        self.get_obj = False
        self.move_to_hole = False
        self.lift_flag = False
        self.done = False
        self.iter_num = 0
        self.obj_num = random.randint(0, len(self.obj_list))
        self.obj_num = 0 # single object now  (0:box) 
        self.hole_num = random.randint(0, len(self.hole_list))
        self.hole_num = 0 # single object now  (0:box_hole) 
        self.hole_pos_xy = self.np_random.uniform(low=-0.04, high=0.04, size=(2,))
        self.hole_pos_z = self.np_random.uniform(low=-0.02, high=0.01, size=(1,))
        
        sim_state = self.sim.get_state()
        # print(sim_state)
        for i in range(self.joint_num):
            sim_state.qpos[i] = initial_pos[i]
        self.sim.set_state(sim_state)
        # 

        self.sim.forward()
        self.hole_init = self.sim.data.get_body_xpos("box")
        self.hole_pos_x = self.hole_init[0] + self.hole_pos_xy[0]
        self.hole_pos_y = self.hole_init[1] + self.hole_pos_xy[1]
        self.hole_pos_z = self.hole_init[2] + self.hole_pos_z
        self.sim.data.set_mocap_pos("box", [self.hole_pos_x, self.hole_pos_y, self.hole_pos_z])
        
        self.sim.forward()

        # print("grasping.")
        self.grasping_moving_init()
        # print("reset success")
        self.get_observation()
        return self.obs


    def render(self, open=False):
        if open is True:
            # print(close)
            # self.viewer.cam.azimuth += 0.1
            self.viewer.render()


    def get_observation(self):
        for i in range(self.joint_num):
            self.obs[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
        for i in range(self.tac_sensor_num):
            self.obs[i+self.joint_num] = self.sim.data.get_sensor(self.tac_sensor_list[i])
        for i in range(self.gripper_joint_num):
            self.obs[i+self.joint_num+self.tac_sensor_num] = self.sim.data.get_joint_qpos(self.gripper_joint_list[i])
        # next for hole body
        for i in range(self.hole_num * 3):
            if (i % 2) == 0 and i == 0:
                self.obs[i+self.joint_num+self.tac_sensor_num+self.gripper_joint_num] = self.sim.data.get_body_xpos(self.hole_list[int(i/3)])[0]
            elif (i % 2) == 0 and i == 2:
                self.obs[i+self.joint_num+self.tac_sensor_num+self.gripper_joint_num] = self.sim.data.get_body_xpos(self.hole_list[int(i/3)])[2]
            else:
                self.obs[i+self.joint_num+self.tac_sensor_num+self.gripper_joint_num] = self.sim.data.get_body_xpos(self.hole_list[int(i/3)])[1]
        # next for target object body
        for i in range(7):
            if i < 3:
                self.obs[i+self.joint_num+self.tac_sensor_num+self.gripper_joint_num+self.hole_num * 3] = self.sim.data.get_body_xpos(self.obj_list[self.obj_num])[i]
            else:
                self.obs[i+self.joint_num+self.tac_sensor_num+self.gripper_joint_num+self.hole_num * 3] = self.sim.data.get_body_xquat(self.obj_list[self.obj_num])[i-3]
                # print("obj quaternion:", self.sim.data.get_body_xquat(self.obj_list[self.obj_num])[i-3])

    
    def get_reward(self):
        d_o_h = 12.33
        t_s = 0.25
        d_o_g = 35.42
        finger_tip_touch_desired = 1
        finger_touch_desired = 10
        if np.linalg.norm(self.sim.data.get_body_xpos("eef") - self.sim.data.get_body_xpos("cylinder_obj")) > 0.05:
            t_s = 0
        self.touch_strength = 0
        self.dis_obj_hole = -math.sqrt((1 - weight_obj_hole_z_axis) * (self.sim.data.get_body_xpos(self.hole_list[0])[0]+0.08 - self.sim.data.get_body_xpos(self.obj_list[0])[0]) ** 2
                            + (1 - weight_obj_hole_z_axis) * (self.sim.data.get_body_xpos(self.hole_list[0])[1]+0.08 - self.sim.data.get_body_xpos(self.obj_list[0])[1]) ** 2
                            + weight_obj_hole_z_axis * (self.sim.data.get_body_xpos(self.obj_list[0])[2] - self.sim.data.get_body_xpos(self.obj_list[0])[2]) ** 2) 
        for i in range(self.tac_sensor_num):
            if i == 4 or i ==5 or i == 10 or i == 11: # the finger tip sensor
                self.touch_strength += -weight_touch_tip * np.absolute(self.obs[i+self.joint_num] - finger_tip_touch_desired) 
            else:
                self.touch_strength += -(1 - weight_touch_tip) * np.absolute(self.obs[i+self.joint_num] - finger_touch_desired)
        self.dis_obj_gripper = -np.linalg.norm(self.sim.data.get_body_xpos("eef") - self.sim.data.get_body_xpos("cylinder_obj"))
        self.reward = d_o_h * self.dis_obj_hole + t_s * self.touch_strength + d_o_g * self.dis_obj_gripper
        hole_pos = self.sim.data.get_body_xpos("box") 
        hole_pos[2] += 0.02 # transform to hole surface
        # dis_obj_hole = np.linalg.norm(self.sim.data.get_body_xpos("cylinder_obj") - hole_pos)
        dis_obj_hole = np.linalg.norm(self.sim.data.get_site_xpos("obj_button") - hole_pos)
        # print(dis_obj_hole)
        # print("button:", self.sim.data.get_site_xpos("obj_button"))
        # print("obj:", self.sim.data.get_body_xpos("cylinder_obj"))
        # print(np.linalg.norm(self.sim.data.get_body_xpos("eef") - self.sim.data.get_body_xpos("cylinder_obj")))
        if np.linalg.norm(self.sim.data.get_body_xpos("eef") - self.sim.data.get_body_xpos("cylinder_obj")) > 0.085:
            # print(np.linalg.norm(self.sim.data.get_body_xpos("eef") - self.sim.data.get_body_xpos("cylinder_obj")))
            self.reward = -10
        elif dis_obj_hole < 0.002:
            self.reward = 4000
            
        # elif (np.linalg.norm(self.sim.data.get_body_xpos("cylinder_obj") - hole_pos)) < 0.01:
        #     self.reward = 1000
        else:
            self.reward = dis_obj_hole
        if np.all([self.sim.data.get_sensor(self.tac_sensor_list[0]), self.sim.data.get_sensor(self.tac_sensor_list[1]), self.sim.data.get_sensor(self.tac_sensor_list[2]), self.sim.data.get_sensor(self.tac_sensor_list[3])] == 0):
            self.reward = -1.5
        elif np.all([self.sim.data.get_sensor(self.tac_sensor_list[6]), self.sim.data.get_sensor(self.tac_sensor_list[7]), self.sim.data.get_sensor(self.tac_sensor_list[8]), self.sim.data.get_sensor(self.tac_sensor_list[9])] == 0):
            self. reward = -1.5


    def get_statement(self):
        for i in range(3):
            self.eef_pos[i] = self.sim.data.get_body_xpos("eef")[i]
        for i in range(4):
            self.eef_quat[i] = self.sim.data.get_body_xquat("eef")[i]

    def grasping_moving_init(self):
        random_init_pos = self.np_random.uniform(low=-0.05, high=0.05, size=(3,))
        down_moving = False
        while self.done is False:
            for i in range(self.joint_num):
                self.sim.data.ctrl[i] = initial_pos[i]
            self.sim.data.ctrl[6] = 0
            self.sim.step()
            for i in range(6):
                self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
            # print(np.linalg.norm(self.obs_joint - np.array(initial_pos[:6])))
            if np.linalg.norm(self.obs_joint - np.array(initial_pos[:6])) < 0.003:
                self.done = True
                # print("init success")
        self.done = False
        # while self.move_to_hole is False:
            
        #     if self.reset_flag is True:
        #         self.obj_init_pos = copy.deepcopy(self.sim.data.get_body_xpos("cylinder_obj"))
        #         self.obj_init_pos[-1] += 0.261
                
        #         self.obj_init_quat = np.roll(np.array(z_quat), -1)
        #         # inv_test_pos = [0.15,      0.38,     1.206475]
        #         # inv_test_pos = self.sim.data.get_body_xpos("eef")
        #         for i in range(6):
        #             self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
        #         self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.obj_init_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
        #         # print(self.qpos)
        #         for i in range(6):
        #             self.sim.data.ctrl[i] = self.qpos[i]
        #         # print("qpos:", self.qpos)
        #         # ur3_kdl_func.forward(initial_pos)
        #         # print("obj_pos:", self.sim.data.get_body_xpos("cylinder_obj"))
        #         # print("ee_pos:", self.sim.data.get_body_xpos("eef"))
        #         # print("ee_quat:", self.sim.data.get_body_xquat("eef"))
        #         # print(np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos))
        #         if np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos) < 0.015:
        #             self.obj_init_pos[-1] -= 0.0815
        #             # print("grasping object ...")
        #             self.reset_flag = False
        #     if self.get_obj is False and self.reset_flag is False:
        #         for i in range(6):
        #             self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
        #         self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.obj_init_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
                
        #         for i in range(6):
        #             self.sim.data.ctrl[i] = self.qpos[i]
        #         if np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos) < 0.015:
        #             self.sim.data.ctrl[6] = 0.4
        #         for i in range(4):
        #             self.right_finger_side[i] = self.sim.data.get_sensor(self.tac_sensor_list[i])
        #             self.left_finger_side[i] = self.sim.data.get_sensor(self.tac_sensor_list[i+6])
        #         # print("r_:", self.right_finger_side)
        #         # print("l_:", self.left_finger_side)
        #         if (self.left_finger_side>2.8).any() and (self.right_finger_side>2.8).any():
        #             self.obj_init_pos[-1] += 0.125
        #             # print("lift ...")
        #             self.get_obj = True
        #     if self.move_to_hole is False and self.get_obj is True and self.reset_flag is False:
        #         if self.lift_flag is False:
        #             for i in range(6):
        #                 self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
        #             self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.obj_init_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
                    
        #             for i in range(6):
        #                 self.sim.data.ctrl[i] = self.qpos[i]
        #             # print(np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos))
        #             if np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos) < 0.02:
        #                 # print("moving to the hole ...")
        #                 # for i in range(2):
        #                 #     self.obj_init_pos[i] = self.sim.data.get_body_xpos("box")[i]
        #                 self.obj_init_pos[0] = self.sim.data.get_body_xpos("box")[0]
        #                 self.obj_init_pos[1] = self.sim.data.get_body_xpos("box")[1]
        #                 self.obj_init_pos[-1] += 0.03
        #                 self.lift_flag = True
        #         if self.lift_flag is True:
        #             for i in range(6):
        #                 self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
        #             self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.obj_init_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
                    
        #             for i in range(6):
        #                 self.sim.data.ctrl[i] = self.qpos[i]
        #             # print(np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos))
        #             if np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos) < 0.02:
        #                 # print("over grasping and moving")
        #                 self.move_to_hole = True

        #     self.sim.step()
        #     # time.sleep(0.02)
        #     if self.is_render is True:
        #         self.viewer.render()

        
        # print(self.obj_init_pos)
        # self.obj_init_pos[-1] -= 0.04
        # while down_moving is False:
        #     for i in range(6):
        #         self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
        #     self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.obj_init_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
        #     # print("qpos:", self.qpos)
        #     # print("arm_joint:", self.obs_joint)
        #     # print("distance:", np.linalg.norm(self.sim.data.get_body_xpos("cylinder_obj") - self.sim.data.get_body_xpos("box")))
        #     for i in range(6):
        #         self.sim.data.ctrl[i] = self.qpos[i]
        #     # print(np.linalg.norm(self.sim.data.get_body_xpos("cylinder_obj") - self.sim.data.get_body_xpos("box")))
        #     if np.linalg.norm(self.sim.data.get_body_xpos("cylinder_obj") - self.sim.data.get_body_xpos("box")) < 0.004:
        #         down_moving = True
        #     self.sim.step()
        #     if self.is_render is True:
        #         self.viewer.render()




def remap(x, lb, ub, LB, UB):
    return (x - lb) / (ub - lb) * (UB - LB) + LB


def joint_limitation(qpos, j_lower, j_upper):
    for i in range(len(joint_limit_lower)):
        while j_lower[i] > qpos[i]:
            qpos[i] = qpos[i] + math.pi
        while qpos[i] > j_upper[i]:
            qpos[i] = qpos[i] - math.pi
    return qpos
