from hrro_env import HRROenv
from ray.rllib.algorithms.ppo import PPOConfig
from auxiliary_classes import (
    Membrane,
    Solution,
    DesignParameters,
    OperationParameters,
)


config_PPO = (
    PPOConfig()
    .environment(
        HRROenv,
        env_config={
            'membrane': Membrane().membrane_xus180808_double(),
            'solution': Solution(),
            'design': DesignParameters.Nijhuis_BIA(),
            'operation': OperationParameters()
        }
    )
    .rollouts(
        num_rollout_workers=25,
        num_envs_per_worker=2,
        create_env_on_local_worker=True
    )
    .framework('tf2', eager_tracing=True)
    .evaluation(evaluation_num_workers=1)
    .resources(num_gpus=1, num_cpus_per_worker=1)
    .experimental(_disable_action_flattening=True)
    .training(
        lr=0.00005,
        kl_coeff=0.0,
        lambda_=0.99,
        clip_param=0.2,
        num_sgd_iter=10,
        sgd_minibatch_size=2048,
        train_batch_size=20000,
        model={
            'fcnet_hiddens': [256, 256],
            'use_lstm': True,
            'lstm_cell_size': 256,
            'max_seq_len': 30,
            'lstm_use_prev_action': True,
            'lstm_use_prev_reward': True,
        }
    )
)

algo = config_PPO.build()

for _ in range(100):
    print(algo.train())

algo.stop()
