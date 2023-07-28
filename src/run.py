# MODEL: Universal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 04/10/2022
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering
import os.path
import time
from ast import arg
from pydoc import render_doc
from socket import VM_SOCKETS_INVALID_VERSION
from turtle import Turtle
import gym
import gym_envs
from datetime import datetime
import argparse

import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
import torch as th
import numpy as np

from loguru import logger

parser = argparse.ArgumentParser(description="runner for UR_Gym")
parser.add_argument('-a', '--alg', type=str, default="TD3", help="pick the used algorithm")
parser.add_argument('-t', '--total_timesteps', type=int, default=5e5, help='the total trainning time steps')
parser.add_argument('-m', '--model_dir', default="pih", help='chose the displayed model')
parser.add_argument('-l', '--learn', action='store_true', help="learning (True)/displaying (False)")
parser.add_argument('-e', '--env', type=str, default="URReach-v1", help='select the trainning environment')
parser.add_argument('-r', '--render', action='store_true', help='open render')
parser.add_argument('-v', '--vision_touch', type=str, default='vision', help='vison model or touch model')
parser.add_argument('-eval', '--evaluate_best', action='store_false',
                    help="default to save the best model when training")
parser.add_argument('-s', '--save_model', action='store_false',
                    help="default to save the model when training in many steps")
parser.add_argument('-nor', '--normalize', action='store_true', help='normalize the observation space')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0003, help='the learning_rate')
parser.add_argument('-dl', '--dyn_learning_rate', action='store_false',
                    help='make the learning rate has a linear schedule')
parser.add_argument('-hs', '--hole_size', type=str, default="2cm", help="pick the hole size (2cm 1cm 4mm ...)")
parser.add_argument('-ms', '--match_shape', action='store_false',
                    help="default to match the shape (add z-axis action into action space)")
parser.add_argument('-real', '--realrobot', action='store_true', help='execte the model on the real robot')
parser.add_argument('-dsl', '--d_s_l', action='store_true', help='execte the model with dynamic safety lock method')
parser.add_argument('-dr', '--domain_randomization', action='store_true',
                    help='execte the model with domain randomization')
parser.add_argument('-g', '--ur_gen', type=int, default=5, help='the generation of URx (3/5/...)')
parser.add_argument('-ro', '--record_obs', action='store_true', help='recording the observation in real/sim.')

args = parser.parse_args()

policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))

# there are some hyperparameters in the whole environment. We can define them here or not.
ft_LowPass_damp = 0.8  # the damp of low pass filter for F/T sensor
if args.realrobot is False:
    dsl_dampRatio_d = np.array([0.0006, 0.0006, 0.0013])  # Dynamic Safety Lock method's hyperparameters in xyz-axis
    dsl_dampRatio_r = np.array([0.015, 0.015, 0.015])  # Dynamic Safety Lock method's hyperparameters in rpy-axis
    ft_xyz_threshold_ur3 = np.array([0.02, 0.02, 0.035])  # threshold of F/T sensor in xyz normalization
    ft_rpy_threshold_ur3 = np.array([0.001, 0.001, 0.001])  # threshold of F/T sensor in rpy normalization
    ft_threshold_xyz = 0.0001  # threshold of F/T sensor for clipping
    ft_threshold_rpy = 0.0001  # same as ft_threshold_xyz
    ee_dis_ratio = 0.00085  # the maximum velocity is (ee_dis_ratio*100)m/s in sim and real.
else:
    dsl_dampRatio_d = np.array([0.001, 0.001, 0.006]) # rigid
    dsl_dampRatio_r = np.array([0.35, 0.35, 0.2])
    # dsl_dampRatio_d = np.array([0.005, 0.005, 0.0085]) # red
    # dsl_dampRatio_r = np.array([0.5, 0.5, 0.3])
    # dsl_dampRatio_d = np.array([0.006, 0.006, 0.00999])  # red
    # dsl_dampRatio_r = np.array([0.7, 0.7, 0.2])
    ft_xyz_threshold_ur3 = np.array([4.5, 4.5, 4])
    ft_rpy_threshold_ur3 = np.array([0.8, 0.8, 0.5])
    ft_threshold_xyz = 0.3
    ft_threshold_rpy = 0.2
    ee_dis_ratio = 0.0009
reward_surface_top_weight = 0.1  # the reward function's weight for the distance between the top of surface and EEF
reward_surface_mid_weight = 0.15  # same function as reward_surface_top_weight
reward_surface_bot_weight = 1.5
c_range = 0.25
ent_coef = 0.0016
# --------------------

root_dir_tensorboard = '/home/yi/project_ghent/tensorboard/'
root_dir_model = '/home/yi/project_ghent/model/'

if args.domain_randomization is True:
    _dr = "_dr"
else:
    _dr = ""
if args.normalize is True:
    _nor = "_nor"
else:
    _nor = ""
if args.d_s_l is True:
    _dsl = "_dsl"
else:
    _dsl = ""
if args.ur_gen == 5:
    _ur_gen = "_g5"
elif args.ur_gen == 3:
    _ur_gen = "_g3"
else:
    _ur_gen = str(args.ur_gen)
    print("chose the generation of UR, the default generation is 5.")
if args.vision_touch == "vision":
    _vision_touch = "vision"
elif args.vision_touch == "vision-touch":
    _vision_touch = "visiontouch"
elif args.vision_touch == "touch":
    _vision_touch = "touch"
else:
    _vision_touch = args.vision_touch
    print("chose the env mode, the default mdoe is VISION.")
saved_model_dir = args.model_dir + "_" + _vision_touch + _dsl + _dr + _nor + _ur_gen + "_"

from typing import Callable

running_time = datetime.now().strftime("%m%d%H%M%S")


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass

if args.d_s_l is False:
    if args.realrobot is False:
        recording_path = '/home/yi/project_ghent/recording/' + _vision_touch + running_time + "_nodsl/"
    else:
        recording_path = '/home/yi/project_ghent/recording/' + _vision_touch + running_time + "_real_nodsl/"
else:
    if args.realrobot is False:
        recording_path = '/home/yi/project_ghent/recording/' + _vision_touch + running_time + "_dsl/"
    else:
        recording_path = '/home/yi/project_ghent/recording/' + _vision_touch + running_time + "_real_dsl/"



def linear_schedule(initial_value: float, lowest_value: float = 0.000) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value + lowest_value

    return func


for i in range(1):
    if args.learn is True:
        # running_time = datetime.now().strftime("%m%d%H%M%S")
        if args.evaluate_best is True:
            eval_env = gym.make(
                args.env,
                vision_touch=args.vision_touch,
                render=args.render,
                normalizeObs=args.normalize,
                hole_size=args.hole_size,
                match_shape=args.match_shape,
                dsl=args.d_s_l,
                domain_randomization=args.domain_randomization,
                ur_gen=args.ur_gen,
                dsl_dampRatio_d=dsl_dampRatio_d,
                dsl_dampRatio_r=dsl_dampRatio_r,
                ft_LowPass_damp=ft_LowPass_damp,
                ft_xyz_threshold_ur3=ft_xyz_threshold_ur3,
                ft_rpy_threshold_ur3=ft_rpy_threshold_ur3,
                ft_threshold_xyz=ft_threshold_xyz,
                ft_threshold_rpy=ft_threshold_rpy,
                ee_dis_ratio=ee_dis_ratio,
                enable_record=args.record_obs,
                recording_path=recording_path,
                reward_surface_weight=np.array(
                    [reward_surface_top_weight, reward_surface_bot_weight, reward_surface_mid_weight]),
            )
            # print("--------")
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=root_dir_model + saved_model_dir + running_time + '/',
                log_path=root_dir_tensorboard + saved_model_dir + running_time + '/',
                eval_freq=20000,
                deterministic=True,
                render=False,
            )
            print('saveing best model in training.')
        if args.save_model is True:
            checkpoint_callback = CheckpointCallback(
                save_freq=200000,
                save_path=root_dir_model + saved_model_dir + running_time + '/',
                name_prefix=saved_model_dir + running_time,
                save_replay_buffer=True,
                save_vecnormalize=True,
            )
            print('saveing updated model in training.')

        if args.evaluate_best is True and args.save_model is True:
            callback = CallbackList([checkpoint_callback, eval_callback])
        elif args.evaluate_best is True and args.save_model is False:
            callback = CallbackList([eval_callback])
        elif args.evaluate_best is False and args.save_model is True:
            callback = CallbackList([checkpoint_callback])
        else:
            callback = CallbackList([])
            print('the model would not be saved by any callback methods in training process.')
        if args.dyn_learning_rate is True:
            _learning_rate = linear_schedule(args.learning_rate, lowest_value=0.0001)
        else:
            _learning_rate = args.learning_rate

        _clip_range = linear_schedule(c_range)

        env = gym.make(
            args.env,
            render=args.render,
            vision_touch=args.vision_touch,
            normalizeObs=args.normalize,
            hole_size=args.hole_size,
            match_shape=args.match_shape,
            dsl=args.d_s_l,
            domain_randomization=args.domain_randomization,
            ur_gen=args.ur_gen,
            dsl_dampRatio_d=dsl_dampRatio_d,
            dsl_dampRatio_r=dsl_dampRatio_r,
            ft_LowPass_damp=ft_LowPass_damp,
            ft_xyz_threshold_ur3=ft_xyz_threshold_ur3,
            ft_rpy_threshold_ur3=ft_rpy_threshold_ur3,
            ft_threshold_xyz=ft_threshold_xyz,
            ft_threshold_rpy=ft_threshold_rpy,
            ee_dis_ratio=ee_dis_ratio,
            enable_record=args.record_obs,
            recording_path=recording_path,
            reward_surface_weight=np.array(
                [reward_surface_top_weight, reward_surface_bot_weight, reward_surface_mid_weight]),
        )
        log_dir = root_dir_tensorboard + saved_model_dir + running_time

        if args.alg == 'TD3':
            print("using TD3 algorithm.")
            model = TD3(
                policy="MultiInputPolicy",
                env=env,
                buffer_size=500000,
                replay_buffer_class=HerReplayBuffer,
                verbose=1,
                learning_rate=_learning_rate,
                tensorboard_log=log_dir)
            model.learn(total_timesteps=args.total_timesteps, callback=callback)
            # model.save(root_dir_model + saved_model_dir + running_time + ".pkl")
            print("the model has saved:", root_dir_model + saved_model_dir + running_time + ".pkl")
        elif args.alg == 'PPO':
            print("using PPO algorithm.")
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                ent_coef=ent_coef,
                # clip_range_vf=0.5,
                clip_range=_clip_range,
                learning_rate=_learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                policy_kwargs=policy_kwargs,
                # target_kl=0.0027,
                tensorboard_log=log_dir)
            # model = RecurrentPPO(
            #     policy="MlpLstmPolicy", 
            #     env=env, 
            #     n_steps=1024, 
            #     batch_size=1024,
            #     verbose=1,
            #     ent_coef=0.01,
            #     clip_range=_clip_range,
            #     learning_rate=_learning_rate, 
            #     tensorboard_log=log_dir)
            print("set up PPO model env")
            trained_model_log = logger.add(root_dir_model + saved_model_dir + running_time + "/params.log")
            logger.info("end-effector is displacement (m/s):" + str(ee_dis_ratio * 100))
            logger.info("dsl damping ratio displacement xyz:" + str(dsl_dampRatio_d))
            logger.info("dsl damping ratio rotation rpy:" + str(dsl_dampRatio_r))
            logger.info("low-pass filter damper for F/T sensor:" + str(ft_LowPass_damp))
            logger.info("F/T snesor's threashold  xyz:" + str(ft_xyz_threshold_ur3))
            logger.info("F/T snesor's threashold  rpy:" + str(ft_rpy_threshold_ur3))
            logger.info("F/T snesor's threashold  xyz(for clip):" + str(ft_threshold_xyz))
            logger.info("F/T snesor's threashold  rpy(for clip):" + str(ft_threshold_rpy))
            logger.info("reward function, the weight of top/surface:" + str(reward_surface_top_weight))
            logger.info("reward function, the weight of mid:" + str(reward_surface_mid_weight))
            logger.info("reward function, the weight of bottom:" + str(reward_surface_bot_weight))
            logger.info("entropy coefficient:" + str(model.ent_coef))
            logger.info("clip range:" + str(c_range))
            logger.info("learning rate:" + str(args.learning_rate))
            logger.info("n_steps:" + str(model.n_steps))
            logger.info("batch_size:" + str(model.batch_size))
            logger.info("n_epochs:" + str(model.n_epochs))
            logger.info("target_kl:" + str(model.target_kl))

            model.learn(total_timesteps=args.total_timesteps, callback=callback)
            # model.save(root_dir_model + saved_model_dir + running_time + ".pkl")
            print("the model has saved:", root_dir_model + saved_model_dir + running_time + ".pkl")

        elif args.alg == 'SAC':
            print("using SAC algorithm.")
            model = TD3(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                buffer_size=100000,
                learning_starts=10000,
                learning_rate=_learning_rate,
                tensorboard_log=log_dir)
            model.learn(total_timesteps=args.total_timesteps, callback=callback)
            # model.save(root_dir_model + saved_model_dir + running_time + ".pkl")
            print("the model has saved:", root_dir_model + saved_model_dir + running_time + ".pkl")

            # print("mdoel learning error, choose the right model algorithm.")
    elif args.learn is False and args.realrobot is False:  # render in sim
        env = gym.make(
            args.env,
            render=args.render,
            vision_touch=args.vision_touch,
            normalizeObs=args.normalize,
            hole_size=args.hole_size,
            dsl=args.d_s_l,
            domain_randomization=args.domain_randomization,
            ur_gen=args.ur_gen,
            dsl_dampRatio_d=dsl_dampRatio_d,
            dsl_dampRatio_r=dsl_dampRatio_r,
            ft_LowPass_damp=ft_LowPass_damp,
            ft_xyz_threshold_ur3=ft_xyz_threshold_ur3,
            ft_rpy_threshold_ur3=ft_rpy_threshold_ur3,
            ft_threshold_xyz=ft_threshold_xyz,
            ft_threshold_rpy=ft_threshold_rpy,
            ee_dis_ratio=ee_dis_ratio,
            enable_record=args.record_obs,
            recording_path=recording_path,
            reward_surface_weight=np.array(
                [reward_surface_top_weight, reward_surface_bot_weight, reward_surface_mid_weight]),
        )

        if args.alg == 'TD3':
            # print(root_dir_model + args.model_dir + '.pkl')
            model = TD3.load(root_dir_model + args.model_dir, env=env)
        elif args.alg == 'PPO':
            model = PPO.load(root_dir_model + args.model_dir, env=env)
        elif args.alg == 'SAC':
            model = TD3.load(root_dir_model + args.model_dir, env=env)
        # plt.figure(1)
        obs, _ = env.reset()

        observation_shape = obs.shape
        obs_record = [np.zeros(observation_shape)]
        # print(obs_record)
        i = 0
        while i < 10:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            obs_record = np.r_[obs_record, [obs]]
            if done:
                # np.save("obs.npy", obs_record)
                i += 1
                print('Done')
                obs, _ = env.reset()
            # plt.subplot(1, 2, 1)
            # plt.show()
            if args.record_obs is True:
                mkdir(recording_path)
                np.save(recording_path + str(i) + '.npy', obs_record)

    elif args.realrobot is True:
        env = gym.make(
            args.env,
            render=args.render,
            vision_touch=args.vision_touch,
            normalizeObs=args.normalize,
            hole_size=args.hole_size,
            real_robot=args.realrobot,
            dsl=args.d_s_l,
            domain_randomization=args.domain_randomization,
            ur_gen=args.ur_gen,
            dsl_dampRatio_d=dsl_dampRatio_d,
            dsl_dampRatio_r=dsl_dampRatio_r,
            ft_LowPass_damp=ft_LowPass_damp,
            ft_xyz_threshold_ur3=ft_xyz_threshold_ur3,
            ft_rpy_threshold_ur3=ft_rpy_threshold_ur3,
            ft_threshold_xyz=ft_threshold_xyz,
            ft_threshold_rpy=ft_threshold_rpy,
            ee_dis_ratio=ee_dis_ratio,
            enable_record=args.record_obs,
            recording_path=recording_path,
            reward_surface_weight=np.array(
                [reward_surface_top_weight, reward_surface_bot_weight, reward_surface_mid_weight]),
        )
        if args.alg == 'TD3':
            # print(root_dir_model + args.model_dir + '.pkl')
            model = TD3.load(root_dir_model + args.model_dir, env=env)
        elif args.alg == 'PPO':
            model = PPO.load(root_dir_model + args.model_dir, env=env)
        elif args.alg == 'SAC':
            model = TD3.load(root_dir_model + args.model_dir, env=env)

        obs, _ = env.reset()

        observation_shape = obs.shape
        obs_record = [np.zeros(observation_shape)]
        i = 0
        while i < 10000:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            obs_record = np.r_[obs_record, [obs]]
            i += 1
            print(i)
            if args.record_obs is True:
                mkdir(recording_path)
                np.save(recording_path + 'realworld.npy', obs_record)

            # if done:
            #     i += 1
            #     print('Done')
            #     obs = env.reset()
