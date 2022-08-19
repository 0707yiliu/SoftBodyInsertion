import gym
import gym_envs
from stable_baselines3 import SAC

env = gym.make('ur3-pih-box-v0')
env.reset()
# observation, reward, done, info = env.step([0.2, -0.2, 0.4, -1])
# print(env.model.joint_names)
while True:
    observation, reward, done, info= env.step([0.2, -0.2, 0.4, -1])
    env.render(True)




# import gym

# from stable_baselines3 import A2C

# env = gym.make('CartPole-v1')

# model = A2C('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)

# obs = env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
