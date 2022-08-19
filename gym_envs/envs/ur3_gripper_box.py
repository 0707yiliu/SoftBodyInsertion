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

initial_pos = [-90 / 180 * math.pi, -100 / 180 * math.pi, -120 / 180 * math.pi,
               -90 / 180 * math.pi, -90 / 180 * math.pi, 0,
               0, 0, 0, 0, 0, 0, 0, 0] # ur3 + robotiq 2F-85
dof = len(initial_pos)

class ur3_gripper_box_env(gym.Env):

    def __init__(self):

        # loading model from xml file
        self.model = load_model_from_path('gym_envs/models/ur3gripper.xml')
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = mj.MjViewer(self.sim)

    def step(self, action):
        self.sim.data.qpos[0] = remap(action[0], -1, 1, -30 / 180 * math.pi, 45 / 180 * math.pi)
        self.sim.data.qpos[1] = remap(action[1], -1, 1, -105 / 180 * math.pi, -50 / 180 * math.pi)
        self.sim.data.qpos[2] = remap(action[2], -1, 1, 0 / 180 * math.pi, 180 / 180 * math.pi)
        self.sim.data.qpos[6] = 0.7
        self.sim.data.qpos[10] = 0.7 # the qpos_6 and qpos_10 are used for robotiq 2F-85, they have the same value
        self.sim.step()
        # print(self.sim.get_state())
        return self.sim.get_state()[:4]

    def reset(self):
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


def remap(x, lb, ub, LB, UB):
    return (x - lb) / (ub - lb) * (UB - LB) + LB

