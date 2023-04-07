# import gym
# import gym_envs
# from stable_baselines3 import PPO 

# env = gym.make('ur3-pih-box-v1', render=True)
# obs = env.reset()
# step = 0

# while True:
#   observation, reward, done, _ = env.step(env.action_space.sample())
#   print(env.action_space.sample())
#   step += 1
#   if step == 500:
#     env.reset()
#     step = 0
#   env.render(True)
#-----------------------------------------------------------------------
# from xml.etree.ElementTree import TreeBuilder
# import gym
# import gym_envs
# from stable_baselines3 import PPO, TD3, SAC, HerReplayBuffer
# from stable_baselines3.common.evaluation import evaluate_policy
# from datetime import datetime

# running_time = datetime.now().strftime("%m%d%H%M%S")
# is_trainning = False
# is_render = bool(1-is_trainning)
# # is_render = True

# if is_trainning is True:
#     env = gym.make('ur3-pih-box-v1', render=is_render)
#     # model = PPO(
#     #     'MlpPolicy', 
#     #     env=env, 
#     #     verbose=1,
#     #     learning_rate=1e-3,
#     #     batch_size=64,
#     #     n_steps=256,
#     #     gamma=0.99,
#     #     # gae_lambda=0.99,
#     #     n_epochs=10,
#     #     ent_coef=0.01,
#     #     tensorboard_log="/home/yi/project_docker/project_ghent/tensorboard/" "PPO_" + running_time
#     #     )
#     model = TD3(
#         policy="MultiInputPolicy", 
#         env=env, 
#         buffer_size=100000, 
#         replay_buffer_class=HerReplayBuffer, 
#         verbose=1,
#         tensorboard_log="/home/yi/project_docker/project_ghent/tensorboard/" "TD3_" + running_time
#         )
#     model.learn(total_timesteps=5e5)
#     model.save("/home/yi/project_docker/project_ghent/model/ur3_pih_box_v1" + running_time + ".pkl")
# else:
#     env = gym.make('ur3-pih-box-v1', render=is_render)
#     model = TD3.load("/home/yi/project_docker/project_ghent/model/ur3_pih_box_v11004123723.pkl", env=env)
# obs = env.reset()
# print("over learning. ------------------------------------------")
# # for i in range(1000):
# while is_render is True:
    
#     # print(action)
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render(True)
#     if done:
#         print("done")
#         obs = env.reset()


# --------------------------------------------------------------------------------
# import gym

# from stable_baselines3 import PPO, SAC

# env = gym.make('CartPole-v1')

# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=2e4)

# obs = env.reset()
# print("over learning. ------------------------------------------")
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()


# --------------------------------------------------------------------------------
# import gym
# import gym_envs
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common import make_vec_env
# from stable_baselines.common.vec_env import VecCheckNan
# from stable_baselines import PPO2

# # multiprocess environment
# env = make_vec_env('ur3-pih-box-v0', n_envs=1)
# env = VecCheckNan(env, raise_exception=True)
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo2_UR3")

# del model # remove to demonstrate saving and loading

# model = PPO2.load("ppo2_UR3")

# # Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

# --------------------------------------------------------------------------------
import gym
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make("Pendulum-v1")

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save("td3_pendulum")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = TD3.load("td3_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
