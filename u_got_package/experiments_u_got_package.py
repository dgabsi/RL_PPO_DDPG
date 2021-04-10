from .env_u_got_package import U_GotPackage
from .qlearning_agent import QLearnAgent
import numpy as np
import os
import matplotlib.pyplot as plt
from .utils import save_to_pickle



SAVED_DIR='./saved'

NUM_EXPERIMENTS=2

#Number of episodes in each experiment
NUM_EPISODES=3


#Environemnt configurations
#Environemnt sizes
ENVIRONMENT_PARAMS={'N':[10], 'CUSTOMS_PAYS':[2]}

#Reward on wate on time- this is the fine for coming to a station with no right document or twice.
#We will examine if when the penalry is small will the agent still do the safe  option of going to the gp
#or will he try to use the randomality  of sometime the gp approval is overlooked.


#Agent Hyperparameters
AGENT_PARAMS={'GAMMAS':[0.3], 'STEP_SIZES':[0.2], 'EPSILONS':[0.6]}


#RUNS_DIR='./experiments'
EPSILON_DECAY=0.999
EPSILON_MIN=0.1

def run_experiments(num_experiments, num_episodes,env_search_config, agent_search_config, epsilon_min, models_dir, train=True, render=False, save_agent=True, episode_avg=20):
    '''
    Run Q learning experiments in the U got package environment. calculate statistics according to different configuration of the agent.
    The agent different configurations will be explored in a grid search manner
    :param num_experiments: Number of experiments on each configuration
    :param num_episodes: Max number of episodes run in each experiment
    :param env_search_config: different environment configuration - dictionary of lists
    :param agent_search_config :different agent configurations - dictionary of lists
    :param epsilon_decay: epsilon decay rate of agent. will be multiplied in each episode
    :param epsilon_min: min epsilon value for agent
    :return: results_dict- dicitnary containing statisitcs rsults for each configuration and episode. averaged over the experiments(Avg return, Avg num timestep, sucess rate)
            , q_values_dict- Q values for environment grid for each configuration. Averaged over the experiments
    '''
    results_dict={'N':[], 'customs_pay':[], 'gamma':[], 'step_size':[] ,'epsilon':[], 'episode':[],'min_reward':[], 'max_reward':[], 'avg_reward':[],'avg_timestep':[], 'avg_got_package':[]}
    q_values_dict={'N':[], 'customs_pay':[], 'gamma':[], 'step_size':[] ,'epsilon':[],'q_values_grid':[]}

    for N in env_search_config['N']:
        for gamma in agent_search_config['GAMMAS']:
            for step_size in agent_search_config['STEP_SIZES']:
                for epsilon in agent_search_config['EPSILONS']:
                    for custom_pay in env_search_config['CUSTOMS_PAYS']:
                        env = U_GotPackage(N, custom_pay)
                        qlearning_agent=QLearnAgent(env.num_states, env.num_actions, gamma, step_size, epsilon,epsilon_min)
                        env.reset()

                        #These arrays will collect statistics of each episode/ experiment pair in the current configuration
                        rewards_per_episode=np.zeros((num_episodes, num_experiments))
                        timestep_per_episode=np.zeros((num_episodes, num_experiments))
                        got_package_per_episode=np.zeros((num_episodes, num_experiments))

                        #We will take the learned q values table of the agent after each experimnet. To enable our understanding

                        q_values=np.zeros((num_experiments, N, N))

                        #current configuration display
                        print(f'********************************************************************************************************')
                        print(f'Env size {N}, Customs pay: {custom_pay } Gamma {gamma} Step size {step_size} Epsilon {epsilon}')
                        print(f'*******************************************************************************************************')

                        #Running all experiment for the configuration
                        for exper in range(num_experiments):
                            qlearning_agent.reset_agent(num_episodes)
                            for episode in range(num_episodes):
                                #Run episode games
                                total_reward, time_steps, got_package= run_episode(env,qlearning_agent, train=train,render=render )

                                running_avg_total_reward=(np.sum(rewards_per_episode[episode-episode_avg:episode, exper]) + total_reward)//episode_avg


                                if not episode%20:
                                    print(f'Expr:{exper} Eps:{episode} RunnAReward:{running_avg_total_reward} Reward:{total_reward} Timesteps:{time_steps} GotPack:{got_package} Utiltime:{env.game_stats["has utility bill"]} Customstime:{env.game_stats["paid customs"]} Money:{env.game_stats["money earned"]}')


                                #Decay the epsilon each episode
                                qlearning_agent.decay_epsilon()

                                #Update statisitcs arrays
                                rewards_per_episode[episode, exper] = total_reward
                                timestep_per_episode[episode, exper] = time_steps
                                got_package_per_episode[episode, exper]=float(got_package)

                            grid_q_values=env.translate_q_values_to_position_grid(qlearning_agent.q_values)

                            q_values_img=np.expand_dims(grid_q_values,2)
                            #print(q_values_img.shape)
                            q_values[exper]=grid_q_values

                            #Show q values grid
                            plt.imshow(q_values_img)
                            plt.title(f'Q values Experiment {exper} Env size {N}, Customs pay: {custom_pay } \n Gamma {gamma} Step size {step_size} Epsilon {epsilon}')
                            plt.show()
                            if save_agent:
                                filename='qlearn_agent '+'env: '+str(N)+'customs_pay:_'+str(custom_pay)+'gamma:_'+str(gamma)+'epsilon:_'+str(epsilon)+'step:_'+str(step_size)+'.pickle'
                                qlearning_agent.save_agent_to_pickle(models_dir, filename)
                        #Calculating the statistics for the set of configuration hyperparamaters
                        for episode in range(num_episodes):
                            results_dict['N'].append(N)
                            results_dict['gamma'].append(gamma)
                            results_dict['step_size'].append(step_size)
                            results_dict['customs_pay'].append(custom_pay)
                            results_dict['epsilon'].append(epsilon)
                            results_dict['episode'].append(episode)

                            #Results are averaged over the experiments
                            results_dict['min_reward'].append(np.mean(rewards_per_episode[episode]))
                            results_dict['max_reward'].append(np.max(rewards_per_episode[episode]))
                            results_dict['avg_reward'].append(np.mean(rewards_per_episode[episode]))
                            results_dict['avg_timestep'].append(np.mean(timestep_per_episode[episode]))
                            results_dict['avg_got_package'].append(np.mean(got_package_per_episode[episode]))
                        q_values_dict["N"].append(N)
                        q_values_dict['gamma'].append(gamma)
                        q_values_dict['step_size'].append(step_size)
                        q_values_dict['customs_pay'].append(custom_pay)
                        q_values_dict['epsilon'].append(epsilon)
                        q_values_dict['q_values_grid'].append(np.mean(q_values, axis=0))

    return results_dict, q_values_dict



def run_episode(env, agent, train=True, render=False):
    '''
    Run one episode in the environment
    :param env: environment
    :param agent: Policy object(has to have get_value_by_policy function) . If parameter train==True has to be also an agent that trains according to a transition
    :param train: If true call train method of the agent
    :param render: True/False -If to display the environment
    :return:  total_reward -episode total reward
              time_steps- num timestamps of episode
              got_package-True/False if the agent got the package (success)
    '''
    done=False
    total_reward=0
    reward=0

    state_index=env.reset()
    while not done:
        action_index=agent.get_action_by_policy(state_index)
        if render:
            env.display(action_index)
        next_state_index, reward, done, got_package= env.step(action_index, render)
        total_reward += reward
        if train:
            agent.train(state_index, action_index, next_state_index, reward)
        state_index=next_state_index
    time_steps=env.time

    return total_reward, time_steps, got_package


if __name__ == '__main__':

    results_dict, q_values_dict=run_experiments(NUM_EXPERIMENTS, NUM_EPISODES,  ENVIRONMENT_PARAMS, AGENT_PARAMS, EPSILON_DECAY, EPSILON_MIN, SAVED_DIR, save_agent=False)

    #filename_results = os.path.join(SAVED_DIR, 'task2_experiments_results.pickle')
    #filename_q_values = os.path.join(SAVED_DIR, 'task2_experiments_q_values.pickle')
    #save_to_pickle(results_dict, filename_results)
    #save_to_pickle(results_dict, filename_q_values)

