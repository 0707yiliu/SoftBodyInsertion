# import gym
# import gym_envs
# from stable_baselines3 import PPO 

# env = gym.make('ur3-pih-box-v1', reder=False)
# obs = env.reset()
# step = 0

# while True:
#   observation, reward, done, _ = env.step(env.action_space.sample())
#   step += 1
#   if step == 500:
#     env.reset()
#     step = 0
#   env.render(True)
#-----------------------------------------------------------------------
import gym
import gym_envs
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime

running_time = datetime.now().strftime("%m%d%H%M%S")
is_trainning = False
is_render = bool(1-is_trainning)

if is_trainning is True:
    env = gym.make('ur3-pih-box-v1', render=is_render)
    model = PPO(
        'MlpPolicy', 
        env=env, 
        verbose=1,
        learning_rate=5e-5,
        batch_size=256,
        n_steps=2048,
        gamma=0.97,
        gae_lambda=0.91,
        n_epochs=50,
        tensorboard_log="./tensorboard/ur3_pih_box_v1/" "PPO_" + running_time
        )
    model.learn(total_timesteps=5e5)
    model.save("./model/ur3_pih_box_v1" + running_time + ".pkl")
else:
    env = gym.make('ur3-pih-box-v1', render=is_render)
    model = PPO.load("./model/ur3_pih_box_v10907163127.pkl", env=env)
obs = env.reset()
print("over learning. ------------------------------------------")
# for i in range(1000):
while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(True)
    if done:
      obs = env.reset()


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

