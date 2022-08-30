
# MODEL: Univewrsal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

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

from gym import error, spaces, utils
from gym.utils import seeding

import ur3_kdl_func


initial_pos = [-90 / 180 * math.pi, -100 / 180 * math.pi, -120 / 180 * math.pi,
               -90 / 180 * math.pi, -90 / 180 * math.pi, 0,
               0, 0, 0, 0, 0, 0, 0, 0] # ur3 + robotiq 2F-85
dof = len(initial_pos)
joint_limit_lower = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
joint_limit_upper = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
action_limit_lower = [-0.01, -0.01, -0.01, -0.314, -0.314, -0.314]
action_limit_upper = [0.01, 0.01, 0.01, 0.314, 0.314, 0.314]

fktestpose = [-1.5, -0.3, 0.3, 0.3, 0.3, -0.3] # testing
obs_dim = 29 # 12 + 6 + 3 + 1 + 7 
action_dim = 6
weight_obj_hole_z_axis = 0.7
weight_touch_tip = 0.65
# ------------- hard code --------------------------


class ur3_gripper_box_env(gym.Env):

    def __init__(self):
        # loading model from xml file
        self.model = load_model_from_path('gym_envs/models/ur3gripper_finger_box.xml')
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = mj.MjViewer(self.sim)
        self.done = False

        self.obs = np.zeros(obs_dim)
        self.reward = -10
        
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
        self.action_space = spaces.Box(low=np.float32(action_limit_lower),high=np.float32(action_limit_upper))
        self.observation_space = spaces.Box(low=-10 * np.float32(np.zeros(obs_dim)),high=10 * np.float32(np.ones(obs_dim)))
        
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        # 0.1476259  0.04700822 0.30628886 # target euler
        # 0.170409,    0.167794,    0.538925 # target position

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
        self.get_observation()
        self.goal_pos = self.obs[i+self.joint_num+self.tac_sensor_num+self.gripper_joint_num+self.hole_num * 3]
        qpos = joint_limitation(ur3_kdl_func.inverse(goal_pos, goal_rot), joint_limit_lower, joint_limit_upper)
        for i in range(6):
            self.sim.data.ctrl[i] = qpos[i]
        # print("base_link pos:", self.sim.data.get_body_xpos("base_link"))
        # print("mujoco ee_link pos:", self.sim.data.get_body_xpos("ee_link"))
        # print("gripper end-effector pos:", self.sim.data.get_body_xquat("eef"))
        # print("tactile sensor:", self.sim.data.get_sensor("touchsensor_r1"))
        self.sim.data.ctrl[6] = 0.5 # finger-joint controller
        self.sim.step()
        self.get_observation()
        self.get_reward()
        if self.reward > 60:
            self.done = True
        # print(self.obs)
        # print(self.reward)
        self.info = {"the processing is going on."}
        return self.obs, self.reward, self.done, self.info

    def reset(self):
        self.done = False
        self.obj_num = random.randint(0, len(self.obj_list))
        self.obj_num = 0 # single object now
        sim_state = self.sim.get_state()
        for i in range(dof):
            sim_state.qpos[i] = initial_pos[i]
        self.sim.set_state(sim_state)   
        self.sim.forward()
        print("reset success")


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

    
    def get_reward(self):
        self.touch_strength = 0
        self.dis_obj_hole = -math.sqrt((1 - weight_obj_hole_z_axis) * (self.obs[self.joint_num+self.tac_sensor_num+self.gripper_joint_num] - self.obs[self.joint_num+self.tac_sensor_num+self.gripper_joint_num+self.hole_num * 3]) ** 2
                            + (1 - weight_obj_hole_z_axis) * (self.obs[self.joint_num+self.tac_sensor_num+self.gripper_joint_num+1] - self.obs[self.joint_num+self.tac_sensor_num+self.gripper_joint_num+self.hole_num * 3 + 1]) ** 2
                            + weight_obj_hole_z_axis * (self.obs[self.joint_num+self.tac_sensor_num+self.gripper_joint_num+2] - self.obs[self.joint_num+self.tac_sensor_num+self.gripper_joint_num+self.hole_num * 3 + 2]) ** 2) 
        for i in range(self.tac_sensor_num):
            if i == 4 or i ==5 or i == 10 or i == 11: # the finger tip sensor
                self.touch_strength += weight_touch_tip * self.obs[i+self.joint_num]
            else:
                self.touch_strength += (1 - weight_touch_tip) * self.obs[i+self.joint_num]
        self.reward = self.dis_obj_hole + self.touch_strength


def remap(x, lb, ub, LB, UB):
    return (x - lb) / (ub - lb) * (UB - LB) + LB


def joint_limitation(qpos, j_lower, j_upper):
    for i in range(len(joint_limit_lower)):
        while j_lower[i] > qpos[i]:
            qpos[i] = qpos[i] + math.pi
        while qpos[i] > j_upper[i]:
            qpos[i] = qpos[i] - math.pi
    return qpos



