# MODEL: Universal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 04/10/2022
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering
from ctypes.wintypes import PINT
from typing import Any, Dict, Union

import math

import numpy as np

from gym_envs.envs.core import Task
from gym_envs.utils import distance, normalizeVector, euler_to_quaternion
from gym_envs.envs.apriltagDetection import AprilTag
from threading import Thread
import time


class PeginHole(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.008,
        z_distance_threshold=0,
        vision_touch="vision",
        # goal_range_low=np.array([-0.1, 0.3, 0.885]),
        # goal_range_high=np.array([0.1, 0.38, 0.955]),
        # goal_range_low=np.array([-0.1, 0.31, 0.93]), # for touch
        # goal_range_high=np.array([0.0, 0.35, 0.93]),
        # goal_range_low=np.array([-0.1, 0.29, 0.85]), # for vision or nodsl
        # goal_range_high=np.array([0.0, 0.36, 0.91]),
        goal_range_low=np.array([0.05, 0.51, 0.865]),  # for vision or nodsl
        goal_range_high=np.array([0.1, 0.56, 0.935]),
        # goal_range_low=np.array([0, 0, 0]),  # for vision or nodsl
        # goal_range_high=np.array([0, 0, 0]),
        # goal_range_low=np.array([-0.04145, 0.31122, 0.88]), # for testing
        # goal_range_high=np.array([-0.04145, 0.31122, 0.88]),
        vision_touch_list=['vision', 'touch', 'vision-touch'],
        _normalize: bool = False,
        real_robot: bool = False,
        ur_gen: int = 5,
    ) -> None:
        super().__init__(sim)
        self.z_distance_threshold = z_distance_threshold
        self._normalize_obs = _normalize
        self.vision_touch=vision_touch
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold/10 if self.vision_touch == 'touch' else distance_threshold
        self.get_ee_position = get_ee_position
        if ur_gen == 5:
            self.goal_range_high = goal_range_high
            self.goal_range_low = goal_range_low
        elif ur_gen == 3:
            self.goal_range_low = np.array([0.05, 0.31, 0.865]) # for vision or nodsl
            self.goal_range_high = np.array([0.1, 0.36, 0.935])
        self.deg2rad = np.pi/180
        self.real_robot = real_robot
        self.suc_times = 0
        self.suc_ratio = 1
        # AprilTag func ---------------------
        self.sim_test_with_camera = False
        if self.sim_test_with_camera is True:
            self.TagDetection = AprilTag()
            self.TransferX = 0
            self.TransferY = 0
            self.TransferZ = 0
            self.AprilTagThread = Thread(target=self.MarkerDetectionThread)
            self.AprilTagThread.setDaemon(True)
            self.AprilTagThread.start()

        if self.real_robot is True and self.sim_test_with_camera is False:
            self.TagDetection = AprilTag()
            self.TransferX = 0
            self.TransferY = 0
            self.TransferZ = 0
            self.AprilTagThread = Thread(target=self.MarkerDetectionThread)
            self.AprilTagThread.setDaemon(True)
            self.AprilTagThread.start()
            
        # -----------------------------------
        # if self.vision_touch == 'touch':
        #     self.goal_range_low = goal_range_low
        #     self.goal_range_high = goal_range_high
        # else:
        #     goal_range_high[2] += 0.02
        #     goal_range_low[2] += 0.02
        #     self.goal_range_high = goal_range_high
        #     self.goal_range_low = goal_range_low
        self.goal = self._sample_goal()
        self.vision_touch_list = vision_touch_list

        norm_max = 1
        norm_min = -1
        self.goal_scale = self.goal_range_high - self.goal_range_low
        self.norm_scale = (norm_max - norm_min) / self.goal_scale
        self.goal_mean = (self.goal_range_high + self.goal_range_low) / 2
        self.norm_mean = (norm_max + norm_min) / 2 * np.array([1, 1, 1])

        # self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        # self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])

        self.get_top_hole = False
        self.get_mid_hole = False
        self.r_step = 0

    def MarkerDetectionThread(self):
        while True:
            self.TransferX, self.TransferY, self.TransferZ = self.TagDetection.calculate_UR3Root2Target()
            # print(self.TransferX)
    def get_obs(self) -> np.ndarray:
        assert self.vision_touch in self.vision_touch_list # model in pre-defined model list
        #! object position, hole position
        # if self.vision_touch=="vision":
        #     object_bottom_position = self.sim.get_site_position("obj_bottom")
        #     hole_position = self.sim.get_site_position('box_surface')
        #     return np.concatenate((object_bottom_position, hole_position))
        # elif self.vision_touch=='touch':
        #     return np.array([])
        # elif self.vision_touch=='vision-touch':
        #     object_bottom_position = self.sim.get_site_position("obj_bottom")
        #     hole_position = self.sim.get_site_position('box_surface')
        #     return np.concatenate((object_bottom_position, hole_position))
        object_bottom_position = np.copy(self.sim.get_site_position("ee_site"))
        # object_bottom_position[0] = -1 + self.goal_range_high[0] - object_bottom_position[0]
        hole_top_position = np.copy(self.sim.get_site_position("hole_top"))
        hole_top_position[0] += (2.0 * np.random.random() + (-1.0)) * 0.003
        hole_top_position[1] += (2.0 * np.random.random() + (-1.0)) * 0.003
        hole_top_position[2] += (2.0 * np.random.random() + (-1.0)) * 0.003

        hole_bot_position = np.copy(self.sim.get_site_position("hole_bottom"))
        hole_bot_position[0] += (2.0 * np.random.random() + (-1.0)) * 0.008
        hole_bot_position[1] += (2.0 * np.random.random() + (-1.0)) * 0.008
        hole_bot_position[2] += (2.0 * np.random.random() + (-1.0)) * 0.008
        # print("sim tool bot:", self.sim.get_site_position('hole_bottom'))
        # print("sim tool top:", self.sim.get_site_position('hole_top'))
        # obs = np.concatenate((hole_top_position, hole_bot_position))
        # print("hole top pos:", hole_top_position)
        # print("hole bot pos:", hole_bot_position)
        # print("---------")
        if self._normalize_obs is True:
            hole_top_position = (hole_top_position - self.goal_mean) * self.norm_scale + self.norm_mean
        obs = np.copy(hole_top_position)
        # print("hole top pos:", obs[:3])
        return obs

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.copy(self.sim.get_site_position("obj_bottom"))
        # print(object_position)
        # object_position[2] += 0.01
        # print("obj:", object_position[2])
        # print("goal:", self.goal[2])
        # if self.vision_touch == 'vision' or self.vision_touch == 'vision-touch':
        #     object_position = self.sim.get_site_position("obj_bottom")
        #     object_position[2] += 0.01
        # else:
        #     object_position = self.sim.get_body_position('cylinder_obj')
        #     object_position[2] -= 0.02
        # object_position = self.sim.get_body_position('cylinder_obj')
        # print("obj pos:",object_position)
        return object_position
    
    def reset(self) -> None:
        self.get_top_hole = False
        self.get_mid_hole = False
        self.r_step = 0
        self.suc_ratio = 1
        self.total_step = 0

        if self.sim_test_with_camera is True:
            time.sleep(1)
            hole_x_offset = 0
            hole_y_offset = 0
            hole_z_offset = 0
            hole_position = np.array([-self.TransferX + hole_x_offset, 
                                            -self.TransferY + hole_y_offset, 
                                            self.TransferZ + hole_z_offset])
            hole_position[2] += 0.87
            desired_goal = np.copy(hole_position)
            self.goal = np.copy(desired_goal)


        else:
            self.goal = self._sample_goal()
            # self.goal = np.array([0.0, 0.36, 0.91])
            desired_goal = np.copy(self.goal)
            if self.vision_touch == 'vision' or 'vision-touch': 
                self.goal[0] += (2.0 * np.random.random() + (-1.0)) * 0.0025
                self.goal[1] += (2.0 * np.random.random() + (-1.0)) * 0.0025
                self.goal[2] += (2.0 * np.random.random() + (-1.0)) * 0.0025
            # if self.vision_touch == 'vision' or self.vision_touch == 'vision-touch':
                # desired_goal[2] -= 0.02
            # desired_goal[2] -= 0.02
            # print("self.goal:", self.goal)
            # print('desired_goal:', desired_goal)
            # # print(hole_position)

        self.sim.set_mocap_pos(mocap="box", pos=desired_goal)
        ## randomize the rotation of the hole in z-axis direction
        z_deg = (2.0 * np.random.random() + (-1.0)) * 90
        xy_deg = (2.0 * np.random.random() + (-1.0)) * 10
        desired_quat = euler_to_quaternion(z_deg * self.deg2rad, (-90+xy_deg) * self.deg2rad, 0)
        self.sim.set_mocap_quat(mocap="box", quat=desired_quat)

        # z_deg = (2.0 * np.random.random() + (-1.0)) * 60
        # desired_quat = euler_to_quaternion(0, 0, z_deg * self.deg2rad)
        # self.sim.set_mocap_quat(mocap="cylinder_obj", quat=desired_quat)

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.random.uniform(self.goal_range_low, self.goal_range_high)
        # goal[2] += 0.05
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        #TODO: transform the achieved and desired goal position to object surface.
        # if self.vision_touch == 'vision' or self.vision_touch == 'vision-touch':
        #     desired_goal[2] += 0.02 # surface

        target_bottom = np.copy(self.sim.get_site_position("hole_bottom"))
        target_top = np.copy(self.sim.get_site_position("hole_top"))
        d_bottom = distance(achieved_goal, target_bottom)
        d_center = distance(achieved_goal, desired_goal)
        d_top = distance(achieved_goal, target_top)

        r_top = 1 - math.tanh(10 * d_top)
        r_center = 1 - math.tanh(10 * d_center)
        r_bot = 1 - math.tanh(10 * d_bottom)

        d = r_bot
        # print("bot r_bot>>",d)
        # print("bot real bot>>", d_bottom)
        # print("-------")

        return np.array(d > 0.9, dtype=np.float64)

        # d = distance(achieved_goal, desired_goal)
        # d_x = abs(achieved_goal[0] - desired_goal[0])
        # d_y = abs(achieved_goal[1] - desired_goal[1])
        # d_xy = d_x + d_y
        # # print(self.goal)
        # # print(d_xy)
        # if d_xy < 0.01:
        #     z_d = achieved_goal[2] - desired_goal[2]
        #     # print("z:", z_d, achieved_goal[2], desired_goal[2])
        #     return np.array(z_d < self.z_distance_threshold, dtype=np.float64)
        # # print(self.goal, self.sim.get_site_position('box_surface'))
        #
        # else:
        #
        # # print(d)
        #     return np.array(d < self.distance_threshold, dtype=np.float64)
    
    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        target_bottom = np.copy(self.sim.get_site_position("hole_bottom"))
        target_top = np.copy(self.sim.get_site_position("hole_top"))
        obj_top = np.copy(self.sim.get_site_position("obj_top"))
        # ---------------------
        # scaled_ratio = np.array([5, 5, 1.5])
        # achieved_goal_scaled = achieved_goal * scaled_ratio
        # target_bottom_scaled = target_bottom * scaled_ratio
        # target_top_scaled = target_top * scaled_ratio
        # desired_goal_scaled = desired_goal * scaled_ratio
        #
        # d_bottom = distance(achieved_goal_scaled, target_bottom_scaled)
        # d_center = distance(achieved_goal_scaled, desired_goal_scaled)
        # d_top = distance(achieved_goal_scaled, target_top_scaled)
        # # self.r_step -= 0.001
        # r_top = 1 - math.tanh(10 * d_top) * 0.95
        # g_top = 0
        # g_mid = 0
        # # print("r_top:", r_top)
        # # print("achieved goal:", achieved_goal)
        # # print("target top:", target_top)
        # # print("distance top:", d_top)
        # # print("distance center:", d_center)
        # # print("distance bottom:", d_bottom)
        # # print("------------")
        # if r_top > 0.9:
        #     self.get_top_hole = True
        #     g_top = 5
        #     print("step 1")
        # # elif r_top < 0.8:
        # #     r_top = -0.01
        # if self.get_top_hole is True:
        #     r_center = 1 - math.tanh(10 * d_center) * 1.5
        # else:
        #     r_center = 0
        #
        # if r_center > 1.4:
        #     g_mid = 10
        #     print("step 2")
        #     self.get_mid_hole = True
        # if self.get_mid_hole is True:
        #     r_bot = 1 - math.tanh(10 * d_bottom) * 2.45
        # else:
        #     r_bot = 0
        # ------------------------
        scaled_ratio_top = np.array([1.25, 1.25, 1])
        scaled_ratio_mid = np.array([1, 1, 1])
        scaled_ratio_bot = np.array([1, 1, 1.25])

        # self.r_step -= 0.0005
        self.r_step += 1
        # print(self.r_step)
        total_steps = 400
        step_ratio = math.tanh((self.r_step / total_steps) * 10) + 0.3
        _step_val = - self.r_step / total_steps # 300 is the total step in one episode
        # step_ratio = math.tanh(self.r_step / 100)
        step_val = _step_val * step_ratio
        # print(step_val)
        d_bottom = distance(achieved_goal * scaled_ratio_bot, target_bottom * scaled_ratio_bot)
        d_center = distance(achieved_goal * scaled_ratio_mid, desired_goal * scaled_ratio_mid)
        d_top = distance(achieved_goal * scaled_ratio_top, target_top * scaled_ratio_top)
        r_top = (1 - math.tanh(50 * d_top)) * 0.2
        r_center = (1 - math.tanh(30 * d_center)) * 0.3
        r_bot = (1 - math.tanh(10 * d_bottom)) * 0.98

        _r_top = -math.tanh(50 * d_top) * 0.2
        _r_center = -math.tanh(50 * d_center) * 0.5
        _r_bot = -math.tanh(50 * d_bottom) * 0.98

        if achieved_goal[2] < target_top[2]:
            _r_top = 0
        if achieved_goal[2] < desired_goal[2]:
            _r_center = 0

        if r_bot > 0.882:
            self.suc_ratio += 0.1
            get_suc = 1.5 * self.suc_ratio
        elif r_bot > 0.85:
            self.suc_ratio = 1.1
            get_suc = 1
        else:
            self.suc_ratio = 1
            get_suc = 0
        scaled_ratio_toptop = np.array([8, 8, 0.1])
        d_objtop_holetop = distance(obj_top * scaled_ratio_toptop, target_top * scaled_ratio_toptop)
        r_objtop_holetop = (1 - math.tanh(20 * d_objtop_holetop)) * 0.1

        left_step = total_steps - self.r_step
        threashold_step = 70
        if left_step < threashold_step:
            step_v = (left_step - threashold_step) / (threashold_step * 70)
        else:
            step_v = 0
        # print("get suc:", get_suc)
        # print("step_v:", step_v)


        # self.total_step += 1
        # print("total step:", self.total_step)
        # if r_top > 0.4:
        #     print("step1")
        # print("dis:", d_top)
        # return r_top + r_center + r_bot + g_top + g_mid + self.r_step
        return (r_top + r_center + r_bot) + get_suc + r_objtop_holetop
        # return ((_r_top + _r_center + _r_bot) * step_ratio/step_ratio)/1 + get_suc
        # # print(achieved_goal[2] - desired_goal[2])
        # # print(d)
        # # x_ratio = 2.3
        # # y_ratio = 2.3
        # x_ratio = 1.5
        # y_ratio = 1.5
        # d_x = abs(achieved_goal[0] - desired_goal[0]) * x_ratio
        # d_y = abs(achieved_goal[1] - desired_goal[1]) * y_ratio
        # d_z = abs(achieved_goal[2] - desired_goal[2]) * 1.5
        # # print(achieved_goal - desired_goal)
        # d_p = 0
        # if d_center < 0.01:
        #     d_p = 1
        # if d_bottom < 0.005:
        #     d_p += 1
        # elif d_center < 0.01 and (achieved_goal[2] - target_bottom[2]) < self.z_distance_threshold:
        #     self.suc_times += 1
        #     d_p += self.suc_times * 5
        # else:
        #     self.suc_times = 0
        # if self.reward_type == "sparse":
        #     return -np.array(d_bottom > self.distance_threshold, dtype=np.float64)
        # else:
        #     # if np.array(d < self.distance_threshold, dtype=np.float64).any():
        #     #     d = -d
        #     # print(d)
        #     # d_ee_obj = distance(self.sim.get_body_position('eef'), self.sim.get_body_position('cylinder_obj'))
        #     # # print(d_ee_obj)
        #     # if d_ee_obj > 0.02:
        #     #     d_ratio = 50.0
        #     # else:
        #     #     d_ratio = 1
        #
        #     return -(d_x + d_y + d_z) + d_p