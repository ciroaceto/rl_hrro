## Files content

auxiliary_classes.py: defines 4 different classes (Membrane, Solution, DesignParameters and OperationParameters) that are used for the initialization of the simulator class StepHRRO.

env.py: first version of the environment. It is used in ppo.py, ppo_lstm.py and tune_dnn.ipynb.

env_norm.py: the normalized version (normalized observations and rewards) of env.py. After the results of the first hyperparameter tuning this was one of the changes for the second one (tune_lstm.ipynb)

lstm_agent.ipynb: in this jupyter notebook the policy from the best perfoming agent of the second iteration is restored and analyzed. 

ppo.py: single agent PPO algorithm training example.

ppo_lstm.py: single agent PPO algorithm with LSTM layer training example.

ppo_lstm_norm.py: single agent PPO algorithm with LSTM layer using normalized environment training example.

simulator.py: hybrid semibatch/batch reverse osmosis simulator class.

tune_dnn.ipynb: first iteration of hyperparameter tuning using PPO, PBT (Population Based Training) and a simple DNN.

tune_lstm.ipynb: second iteration of hyperparameter tuning using PPO, PBT (Population Based Training) and a LSTM layer added to the previous DNN.

## Training process

### Debugging and first trainings

Once the environment is finished the best way to find bugs is to start training. After some thousands or millions of iterations all the bugs will show up. Sometimes error messages will guide you on how to fix them. Sometimes not.

First training were performed using some simple DNN architectures with only 2 to 9 hidden layers, hyperparameters fixed. Best results were achieved with a 4 hidden layer architecture: [256, 256, 256, 128].

### First iteration

This first hyperparameter tuning objective was to achieve better results than the first trainings. The initial hyperparameters and its variations can be found in tune_dnn.ipynb.

### Second iteration

The best performing agents from the first iteration were still failing to switch from semibatch to batch phase. In consequence, the maximum supply pressure was reached frequently. In order to stabilize learning, the environment observations and rewards were normalized. Observation new values between -1 and 1. Reward new values divided by 1000 (to ease interpretation of results) and, therefore, between -1 and 0.05, approximately. To provide some memory to the model, a LSTM layer was added.

The results improvement doubled in comparison to the first iterations. The best agents achieved a positive mean reward and a minimum reward of a whole batch of data near -100.

## Conclusions

- The agent was able to learn a complex policy to control a novel membrane process.
- The LSTM layer was key for achieving the best results.
- Even though in some cases the agent seems to finish the semibatch phase too soon, it is an inteligent strategy to achieve a better reward in the end of the episode.

## Future developments (intelWATT Project)

### Multi-agent setup

RLlib does not allow action masking of continuous actions or nested action spaces. With the current implementation the agent decides all the action at every time step. 4 out 5 actions are not used in every time step. A multi-agent setup could solve this issue. Although the agent's collaboration is a challenge, the final result could possibly be improved.

### Input variable

Even though the feed concentration was used as an input, this value won't be available in the real pilot. Therefore, the next agent version will use conductivity to approximate the concentration of the feed solution. This approximation allowed a simplification of the simulator to be used for training.

### Edge deployment

To provide high availability (independent of network conection) and low latency.



