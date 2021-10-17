# reinforcement learning DDPG and PPO for Moon Lander
# Code for course Reinforcement learning
### Daniela Stern- Gabsi

### github- dgabsi/RL_project_707

(updats were made from danielaneuralx which is my working github but its all mine)

This project includes 4 different main tasks. The first task I will build an environment based on a famous Israel board game 
where your aim is to pickup a package. The second task I will build and agent using the Q learning algorithm, and I will 
demonstrate that by using this algorithm the agent is abel to play the game successfully without prior knowledge of the game rules.
In the third task I will implement an advanced Reinforcement learning algorithm-The DDPG algorithm which is based on Deterministic policy gradient approximation
.I will use the Lunar Lander Gym environment. In the fourth task I will use another model-the Proximal Policy Optimization (PPO) 
which is a very effective and stable model for policy approximation.
The code is implemented in pytorch.
Main notebooks are:
- Task1_u_got_package.ipynb (Main for task 1) 
- Task2_q_learning.ipynb (Main for task 2)
- Task3_ddpg.ipynb (Main for task 3)
- Task4_ppo.ipynb (Main for task 4)

Please fill cuda in  notebooks.
Since running the models takes a long time I have saved trained models and pickle dictionaries containing the results statistics.

**Important**- Due to the limitation in moodle file sizes. The project was compressed to two zip files
But when extracted the **contents** should be put together. See project structure.


Project structure:
- luna_lander (package- task 3+4)
    - experiments (Directory for tensorboard)
    - saved (Directory for pickles and saved models)
      -best
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
- collections
- matplotlib
- numpy
- pandas
- termcolor
- gym
- scikit-learn
- tensorboard
- math
- random
- pickle

- please install gym box2 : pip install Box2D gym
  (more instructions are at https://github.com/openai/gym/blob/master/docs/environments.md)
