from gym.envs.registration import register

register(
    id='foo-v0',
    entry_point='gym_envs.envs:FooEnv',
)

register(
    id='ur3-pih-box-v0',
    entry_point='gym_envs.envs:ur3_gripper_box_env',
)

register(
    id='ur3-pih-box-v1',
    entry_point='gym_envs.envs:ur3_gripper_box_env_v1',
    
    max_episode_steps=500,
)

for reward_type in ["sparse", "dense"]:
    for control_type in ["ee", "joints"]:
        reward_suffix = "Dense" if reward_type == "dense" else ""
        control_suffix = "Joints" if control_type == "joints" else ""
        kwargs = {"reward_type": reward_type, "control_type": control_type}

        register(
            id="URReach{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="gym_envs.envs:URReachEnv",
            kwargs=kwargs,
            max_episode_steps=500,
        )
        register(
            id="URPiH{}{}-v1".format(control_suffix, reward_suffix),
            entry_point="gym_envs.envs:URPeginHoleEnv",
            kwargs=kwargs,
            max_episode_steps=150,
        )

