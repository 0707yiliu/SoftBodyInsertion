# MODEL: Universal Robots UR3 + Robotiq 2F-85 env
# AUTHOR: Yi Liu @AiRO 04/10/2022
# UNIVERSITY: UGent-imec
# DEPARTMENT: Faculty of Engineering and Architecture
# Control Engineering / Automation Engineering

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

import numpy as np

parser = argparse.ArgumentParser(description="runner for UR_Gym")
parser.add_argument('-a', '--alg', type=str, default="TD3", help="pick the used algorithm")
parser.add_argument('-t', '--total_timesteps', type=int, default=5e5, help='the total trainning time steps')
parser.add_argument('-m', '--model_dir',  help='chose the displayed model')
parser.add_argument('-l', '--learn', action='store_true', help="learning (True)/displaying (False)")
parser.add_argument('-e', '--env', type=str, default="URReach-v1", help='select the trainning environment')
parser.add_argument('-r', '--render', action='store_true', help='open render')
parser.add_argument('-v', '--vision_touch', type=str, default='vision', help='vison model or touch model')
parser.add_argument('-eval', '--evaluate_best', action='store_false', help="default to save the best model when training")
parser.add_argument('-s', '--save_model', action='store_false', help="default to save the model when training in many steps")
parser.add_argument('-n', '--normalize', action='store_true', help='normalize the observation space')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0003, help='the learning_rate')
parser.add_argument('-dl', '--dyn_learning_rate', action='store_false', help='make the learning rate has a linear schedule')
parser.add_argument('-hs', '--hole_size', type=str, default="2cm", help="pick the hole size (2cm 1cm 4mm ...)")
parser.add_argument('-ms', '--match_shape', action='store_false', help="default to match the shape (add z-axis action into action space)")
parser.add_argument('-real', '--realrobot', action='store_true', help='execte the model on the real robot')
parser.add_argument('-dsl', '--d_s_l', action='store_true', help='execte the model with dynamic safety lock method')
parser.add_argument('-dr', '--domain_randomization', action='store_true', help='execte the model with domain randomization')

args = parser.parse_args()

root_dir_tensorboard = '/home/yi/project_ghent/tensorboard/'
root_dir_model = '/home/yi/project_ghent/model/'


from typing import Callable


running_time = datetime.now().strftime("%m%d%H%M%S")
def linear_schedule(initial_value: float) -> Callable[[float], float]:
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
        return progress_remaining * initial_value

    return func

for i in range(1):
    if args.learn is True:
        running_time = datetime.now().strftime("%m%d%H%M%S")
        if args.evaluate_best is True:
            eval_env = gym.make(
                args.env, 
                vision_touch=args.vision_touch,
                render=args.render,
                normalizeObs=args.normalize,
                hole_size=args.hole_size,
                match_shape=args.match_shape,
                dsl = args.d_s_l,
                domain_randomization = args.domain_randomization,
                )
            # print("--------")
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=root_dir_model+args.model_dir+running_time+'/',
                log_path=root_dir_tensorboard+args.model_dir+running_time+'/',
                eval_freq=100000,
                deterministic=True,
                render=False,
            )
            print('saveing best model in training.')
        if args.save_model is True:
            checkpoint_callback = CheckpointCallback(
                save_freq=200000,
                save_path=root_dir_model+args.model_dir+running_time+'/',
                name_prefix=args.model_dir+running_time,
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
            _learning_rate = linear_schedule(args.learning_rate)
        else:
            _learning_rate = args.learning_rate
        
        _clip_range = linear_schedule(0.2)
        
        env = gym.make(
            args.env,
            render=args.render,
            vision_touch=args.vision_touch,
            normalizeObs=args.normalize,
            hole_size=args.hole_size,
            match_shape=args.match_shape,
            dsl = args.d_s_l,
            domain_randomization = args.domain_randomization,
            )
        log_dir = root_dir_tensorboard + args.model_dir + running_time
       
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
            # model.save(root_dir_model + args.model_dir + running_time + ".pkl")
            print("the model has saved:", root_dir_model + args.model_dir + running_time + ".pkl")
        elif args.alg == 'PPO':
            print("using PPO algorithm.")
            model = PPO(
                policy="MlpPolicy", 
                env=env, 
                verbose=1,
                ent_coef=0.01,
                clip_range=_clip_range,
                learning_rate=_learning_rate,
                n_steps=4096,
                batch_size=256,
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
            model.learn(total_timesteps=args.total_timesteps, callback=callback)
            # model.save(root_dir_model + args.model_dir + running_time + ".pkl")
            print("the model has saved:", root_dir_model + args.model_dir + running_time + ".pkl")
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
            # model.save(root_dir_model + args.model_dir + running_time + ".pkl")
            print("the model has saved:", root_dir_model + args.model_dir + running_time + ".pkl")

            # print("mdoel learning error, choose the right model algorithm.")
    elif args.learn is False and args.realrobot is False: # render in sim
        env = gym.make(
            args.env,
            render=args.render,
            vision_touch=args.vision_touch,
            normalizeObs=args.normalize,
            hole_size = args.hole_size,
            dsl = args.d_s_l,
            domain_randomization = args.domain_randomization,
            render_mode="human",
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
            obs, reward, done, info = env.step(action)
            obs_record = np.r_[obs_record, [obs]]
            if done:
                # np.save("obs.npy", obs_record)
                i += 1
                print('Done')
                obs, _ = env.reset()
        # plt.subplot(1, 2, 1)
        # plt.show()
                if args.d_s_l is False:
                    np.save('/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/src/recording/' + args.vision_touch + args.hole_size + running_time + "_nodsl_"+ str(i) +'.npy', obs_record)
                else:
                    np.save('/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/src/recording/' + args.vision_touch + args.hole_size + running_time + "_"+ str(i) +'.npy', obs_record)
    
    elif args.realrobot is True:
        env = gym.make(
            args.env,
            render=args.render,
            vision_touch=args.vision_touch,
            normalizeObs=args.normalize,
            hole_size = args.hole_size,
            real_robot = args.realrobot,
            dsl = args.d_s_l,
            domain_randomization = args.domain_randomization,
            render_mode="human",
            )
        if args.alg == 'TD3':
            # print(root_dir_model + args.model_dir + '.pkl')
            model = TD3.load(root_dir_model + args.model_dir, env=env)
        elif args.alg == 'PPO':
            model = PPO.load(root_dir_model + args.model_dir, env=env)
        elif args.alg == 'SAC':
            model = TD3.load(root_dir_model + args.model_dir, env=env)
        obs = env.reset()

        observation_shape = obs.shape
        obs_record = [np.zeros(observation_shape)]
        i = 0
        while i < 90:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            obs_record = np.r_[obs_record, [obs]]
            i += 1
            # print(i)
            # if args.d_s_l is False:
            #     np.save('/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/src/recording/' + args.vision_touch + args.hole_size + running_time + '_real_nodsl.npy', obs_record)
            # else:
            #     np.save('/home/yi/robotic_manipulation/peg_in_hole/ur3_rl_sim2real/src/recording/' + args.vision_touch + args.hole_size + running_time + '_real.npy', obs_record)

            # if done:
            #     i += 1
            #     print('Done')
            #     obs = env.reset()
