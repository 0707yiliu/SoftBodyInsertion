# import gym
# import gym_envs
# from stable_baselines3 import PPO 

# env = gym.make('ur3-pih-box-v0')
# env.reset()

# while True:
#   observation, reward, done, _ = env.step(env.action_space.sample())
#   env.render(True)
#-----------------------------------------------------------------------
# import gym
# import gym_envs
# from stable_baselines3 import PPO

# env = gym.make('ur3-pih-box-v0')
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=2e4)
# env.reset()
# while True:
#   observation, reward, done, _ = env.step(env.action_space.sample())
#   env.render(True)

# --------------------------------------------------------------------------------
# import gym

# from stable_baselines3 import PPO, SAC

# env = gym.make('CartPole-v1')

# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=2e3)

# obs = env.reset()
# print("over learning. ------------------------------------------")
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()


# --------------------------------------------------------------------------------
import gym
import gym_envs
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecCheckNan
from stable_baselines import PPO2

# multiprocess environment
env = make_vec_env('ur3-pih-box-v0', n_envs=1)
env = VecCheckNan(env, raise_exception=True)
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_UR3")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_UR3")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

