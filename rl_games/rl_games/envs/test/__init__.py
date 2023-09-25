import gym

gym.envs.register(
     id='TestRnnEnv-v0',
     entry_point='rl_games.envs.test.rnn_env:TestRNNEnv',
     max_episode_steps=100500,
)

gym.envs.register(
     id='TestAsymmetricEnv-v0',
     entry_point='rl_games.envs.test.test_asymmetric_env:TestAsymmetricCritic'
)