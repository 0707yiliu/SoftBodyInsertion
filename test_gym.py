import gym
import gym_envs
from stable_baselines3 import PPO 

env = gym.make('ur3-pih-box-v0')
env.reset()

while True:
  observation, reward, done, _ = env.step(env.action_space.sample())
  env.render(True)




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
