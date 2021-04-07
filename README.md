# 707 project
# Code for course Reinforcement learning
### Daniela Stern- Gabsi

### github- dgabsi/707project
(updats were made from danielaneuralx which is my working github but its all mine)

This project includes 4 different main tasks. The first task I will build an environment based on a famous Israel board game 
where your aim is to pickup a package. The second task I will build and agent using the Q learning algorithm, and I will 
demonstrate that by using this algorithm the agent is abel to play the game successfully without prior knowledge of the game rules.
In the third task I will implement an advanced Reinforcement learning algorithm-The DDPG algorithm which is based on Deterministic policy gradient approximation
.I will use the Lunar Lander Gym environment. In the fourth taks I will use another model-the Proximal Policy Optimization (PPO) 
which is a very effective and stable model for policy approximation.
The code is implemented in pytorch.
Main notebooks are:
- Task1_u_got_package.ipynb (Main for task 1) 
- Task2_q_learning.ipynb (Main for task 2)
- Task3_ddpg.ipynb (Main for task 3)
- Task4_ppo.ipynb (Main for task 4)

Please fill cuda in  notebooks.
Since running the models takes a long time I have saved trained models and pickle dictionaries containing the results statistics.

Project structure:
- luna_lander (package- task 3+4)
    - experiments (Directory for tensorboard)
    - saved (Directory for pickles and saved models)
    - __init__.py
    - agent_ddpg.py
    - ddpg_actor_critic_networks.py
    - replay_memory.py
    - experiment_lander.py
    - ppo_agent.py
    - ppo_memory.py
    - ppo_networks.py 
    - utils.py
- u_got_package (package- task1+2)
    - saved (Directory for pickles and saved models)
    - __init__.py 
    - env_u_got_package.py 
    - experiments_u_got_package.py
    - policies.py
    - q_learning_agent.py
    - utils.py 
- Task1_u_got_package.ipynb (Main for task 1)
- Task2_q_learning.ipynb (Main for task 2)
- Task3_ddpg.ipynb (Main for task 3)
- Task4_ppo.ipynb (Main for task 4)

packages needed :
- torch
- datetime
- time
- itertools
- matplotlib
- numpy
- pandas
- pytorch
- pyyaml
- scikit-learn
- torchsummary
- torchvision
- yaml
- abc
- enum
- math
- random
