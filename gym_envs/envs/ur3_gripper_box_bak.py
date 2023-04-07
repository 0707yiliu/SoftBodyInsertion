
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
import copy

from gym import error, spaces, utils
from gym.utils import seeding
from scipy.spatial.transform import Rotation as R

import ur3_kdl_func


initial_pos = [90 / 180 * math.pi, -90 / 180 * math.pi, 90 / 180 * math.pi,
               -90 / 180 * math.pi, -90 / 180 * math.pi, 0,
               0, 0, 0, 0, 0, 0, 0, 0] # ur3 + robotiq 2F-85
joint_limit_lower = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
joint_limit_upper = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
action_limit_lower = np.array([-0.01, -0.01, -0.01])
action_limit_lower = action_limit_lower/10.0
action_limit_upper = np.array([0.01, 0.01, 0.01])
action_limit_upper = action_limit_upper/10.0
# print(action_limit_lower.shape)

fktestpose = [-1.5, -0.3, 0.3, 0.3, 0.3, -0.3] # testing
obs_dim = 29 # 12 + 6 + 3 + 1 + 7 
action_dim = 3
weight_obj_hole_z_axis = 0.7
weight_touch_tip = 0.65
z_quat = [0, 1,  0,  0] #  wxyz
# ------------- hard code --------------------------


class ur3_gripper_box_env(gym.Env):

    def __init__(self):
        # loading model from xml file
        self.model = load_model_from_path('gym_envs/models/ur3gripper_finger_box.xml')
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = mj.MjViewer(self.sim)
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
        self.action_space = spaces.Box(-0.010, 0.010, shape=(3,), dtype=np.float64)
        self.observation_space = spaces.Box(-10, 10, shape=(obs_dim,), dtype=np.float64)
        # print("---------------------", self.action_space)
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        # 0.1476259  0.04700822 0.30628886 # target euler
        # 0.170409,    0.167794,    0.538925 # target position
        self.iter_num += 1
        self.done = False
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
        #! ------------------ generating new action for env ---------------------------
        #! getting the UR3-joint statement and the obj pos, statement + action[]
        # self.eef_quat = self.sim.data.get_body_xquat("eef")
        # print(action)
        for i in range(3):
            self.goal_pos[i] = self.sim.data.get_body_xpos("eef")[i]
        # for i in range(4):
        #     self.goal_quat[i] = self.obs[i+3+self.joint_num+self.tac_sensor_num+self.gripper_joint_num+self.hole_num * 3]
        # self.goal_rot = np.array(R.from_quat([self.goal_quat[1], self.goal_quat[2], self.goal_quat[3], self.goal_quat[0]]).as_euler('zyx', degrees=False))
        # self.gripper_joint = self.obs[self.joint_num+self.tac_sensor_num]
        self.get_statement()

        # # TODO: the actions are incremental, so they need to be added into observation above
        # #! actions: delta-eef-pos(3-dim) delta-eef-rot(3-dim) delta-gripper-joint(1-dim)
        for i in range(3):
            self.goal_pos[i] += action[i]
        for i in range(6):
            self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
        #     self.goal_rot[i] += action[i+3]
        # self.gripper_joint += action[-1]
        # if self.gripper_joint > 0.7:
        #     self.gripper_joint = 0.7
        # if self.gripper_joint <= 0:
        #     self.gripper_joint = 0
        # self.goal_quat = np.array(R.from_euler('zyx', [self.goal_rot[0], self.goal_rot[1], self.goal_rot[2]], degrees=False).as_quat())
        self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.goal_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
        for i in range(6):
            self.sim.data.ctrl[i] = self.qpos[i]
        self.sim.data.ctrl[6] = 0.4
        # print(self.qpos)
        # print("base_link pos:", self.sim.data.get_body_xpos("base_link"))
        # print("mujoco ee_link pos:", self.sim.data.get_body_xpos("ee_link"))
        # print("gripper end-effector pos:", self.sim.data.get_body_xquat("eef"))
        # print("tactile sensor:", self.sim.data.get_sensor("touchsensor_r1"))
        # for i in range(self.joint_num):
        #     self.obs[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
        # self.sim.data.ctrl[6] = 0.5 # finger-joint controller
        # print("action:", action)
        self.sim.step()
        self.get_observation()
        self.get_reward()
        if self.reward > 60:
            self.done = True
        if np.linalg.norm(self.sim.data.get_body_xpos("eef") - self.sim.data.get_body_xpos("cylinder_obj")) > 0.03:
            self.done = True

        # print(self.done)
        # print(self.iter_num)
        
        # print(self.sim.data.get_body_xquat("eef"))
        # print(self.obs)
        # print(self.reward)
        self.info = {}
        # self.viewer.render()
        return self.obs, self.reward, self.done, self.info

    def reset(self):
        self.sim.reset()
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
        sim_state = self.sim.get_state()
        for i in range(self.joint_num):
            sim_state.qpos[i] = initial_pos[i]
        self.sim.set_state(sim_state)   
        self.sim.forward()
        print("grasping.")
        self.grasping_moving_init()
        print("reset success")
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
    
    def get_statement(self):
        for i in range(3):
            self.eef_pos[i] = self.sim.data.get_body_xpos("eef")[i]
        for i in range(4):
            self.eef_quat[i] = self.sim.data.get_body_xquat("eef")[i]

    def grasping_moving_init(self):
        while self.done is False:
            for i in range(self.joint_num):
                self.sim.data.ctrl[i] = initial_pos[i]
            self.sim.data.ctrl[6] = 0
            self.sim.step()
            for i in range(6):
                self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
            if np.linalg.norm(self.obs_joint - np.array(initial_pos[:6])) < 0.003:
                self.done = True
                print("init success")
        self.done = False
        while self.move_to_hole is False:
            
            if self.reset_flag is True:
                self.obj_init_pos = copy.deepcopy(self.sim.data.get_body_xpos("cylinder_obj"))
                self.obj_init_pos[-1] += 0.241
                
                self.obj_init_quat = np.roll(np.array(z_quat), -1)
                # inv_test_pos = [0.15,      0.38,     1.206475]
                # inv_test_pos = self.sim.data.get_body_xpos("eef")
                for i in range(6):
                    self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
                self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.obj_init_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
                # print(self.qpos)
                for i in range(6):
                    self.sim.data.ctrl[i] = self.qpos[i]
                # print("qpos:", self.qpos)
                # ur3_kdl_func.forward(initial_pos)
                # print("obj_pos:", self.sim.data.get_body_xpos("cylinder_obj"))
                # print("ee_pos:", self.sim.data.get_body_xpos("eef"))
                # print("ee_quat:", self.sim.data.get_body_xquat("eef"))
                if np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos) < 0.003:
                    self.obj_init_pos[-1] -= 0.0815
                    print("grasping object ...")
                    self.reset_flag = False
            if self.get_obj is False and self.reset_flag is False:
                for i in range(6):
                    self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
                self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.obj_init_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
                
                for i in range(6):
                    self.sim.data.ctrl[i] = self.qpos[i]
                if np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos) < 0.003:
                    self.sim.data.ctrl[6] = 0.4
                for i in range(4):
                    self.right_finger_side[i] = self.sim.data.get_sensor(self.tac_sensor_list[i])
                    self.left_finger_side[i] = self.sim.data.get_sensor(self.tac_sensor_list[i+6])
                if (self.left_finger_side[i]>3).any() and (self.right_finger_side[i]>3).any():
                    self.obj_init_pos[-1] += 0.125
                    print("lift ...")
                    self.get_obj = True
            if self.move_to_hole is False and self.get_obj is True and self.reset_flag is False:
                if self.lift_flag is False:
                    for i in range(6):
                        self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
                    self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.obj_init_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
                    
                    for i in range(6):
                        self.sim.data.ctrl[i] = self.qpos[i]
                    # print(np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos))
                    if np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos) < 0.004:
                        print("moving to the hole ...")
                        # for i in range(2):
                        #     self.obj_init_pos[i] = self.sim.data.get_body_xpos("box")[i]
                        self.obj_init_pos[0] = self.sim.data.get_body_xpos("box")[0] + 0.08
                        self.obj_init_pos[1] = self.sim.data.get_body_xpos("box")[1] + 0.08
                        self.obj_init_pos[-1] += 0.005
                        self.lift_flag = True
                if self.lift_flag is True:
                    for i in range(6):
                        self.obs_joint[i] = self.sim.data.get_joint_qpos(self.joint_list[i])
                    self.qpos = joint_limitation(ur3_kdl_func.inverse(self.obs_joint, self.obj_init_pos, self.obj_init_quat), joint_limit_lower, joint_limit_upper)
                    
                    for i in range(6):
                        self.sim.data.ctrl[i] = self.qpos[i]
                    if np.linalg.norm(self.sim.data.get_body_xpos("ee_link") - self.obj_init_pos) < 0.004:
                        print("over grasping and moving")
                        self.move_to_hole = True
            self.sim.step()
            # self.viewer.render()




def remap(x, lb, ub, LB, UB):
    return (x - lb) / (ub - lb) * (UB - LB) + LB


def joint_limitation(qpos, j_lower, j_upper):
    for i in range(len(joint_limit_lower)):
        while j_lower[i] > qpos[i]:
            qpos[i] = qpos[i] + math.pi
        while qpos[i] > j_upper[i]:
            qpos[i] = qpos[i] - math.pi
    return qpos
