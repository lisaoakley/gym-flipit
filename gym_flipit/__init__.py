from gym.envs.registration import register

register(
    id='flipit-v0',
    entry_point='gym_flipit.envs:FlipitEnv',
)
