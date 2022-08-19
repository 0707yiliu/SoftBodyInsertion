from gym.envs.registration import register

register(
    id='foo-v0',
    entry_point='gym_envs.envs:FooEnv',
)

register(
    id='ur3-pih-box-v0',
    entry_point='gym_envs.envs:ur3_gripper_box_env',
)
