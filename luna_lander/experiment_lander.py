import gym
from .agent_ddpg import DDPG_Agent
from .ppo_agent import PPO_Agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import datetime
from .utils import plot_all_experiments,plot_episodes,save_to_pickle




def run_episodes(env, agent, writer, num_episodes, num_episodes_avg=20, train=False, save_freq=100, models_dir=None, filename_critic=None, filename_actor=None, save_rewards=150,game_solved_reward=200, render=False, agnet_type='DDPG'):
    '''
    Run one experiments on the Lunar lander environment for number of episodes(parameters).
    Save checkpoints if requested, Return running average rewards
    '''

    #Prepare lists for results
    all_total_rewards = []
    all_num_timesteps = []
    running_avg_rewards=[]
    running_avg_timesteps = []
    max_reward=0
    number_times_solved=0

    #run the game on the environment
    for episode in range(num_episodes):
        observation=env.reset()
        done=False
        total_reward=0
        num_timesteps=0
        while not done:
            if render:
                env.render()
            if agnet_type=='DDPG':
                action=agent.return_action_by_policy(observation, train=train)
            else:
                action, log_prob, entropy = agent.return_action_by_policy(observation, train=train)
                value = agent.get_value(observation)
            observation_next, reward, done, info=env.step(action)
            #If in training mode , store transition and run agent train
            if train:
                if agnet_type == 'DDPG':
                    agent.store_transition(observation, action, observation_next, reward, done)
                else:
                    agent.store_transition(observation, action, log_prob, entropy, value, reward, done)
                agent.train()
            total_reward+=reward
            observation=observation_next
            num_timesteps+=1

        #Update episode results
        all_total_rewards.append(total_reward)
        all_num_timesteps.append(num_timesteps)
        #Update moving avaraege reward and timesteps array
        running_moving_avg_reward=np.mean(all_total_rewards[-num_episodes_avg:])
        running_avg_num_timesteps=np.mean(all_num_timesteps[-num_episodes_avg:])
        running_avg_rewards.append(running_moving_avg_reward)
        running_avg_timesteps.append(running_avg_num_timesteps)
        #Collect also statistics about maximum reward and number of times the game was solved
        if total_reward>max_reward:
            max_reward=total_reward
        if total_reward>game_solved_reward:
            number_times_solved+=1
        #I reqested to save -save the checkpoints
        if save_freq is not None:
            if not episode%save_freq:
                date_str = datetime.datetime.now().strftime("%m%d%Y %H%M")
                file_name_actor = filename_actor+'_' +date_str
                file_name_critic = filename_critic +'_' +date_str
                agent.save_model_and_optimizer(models_dir, filename_critic=file_name_critic, filename_actor=file_name_actor)
            if running_moving_avg_reward>save_rewards:
                file_str = 'Over '+str(save_rewards)+' '+datetime.datetime.now().strftime("%m%d%Y %H")
                file_name_actor = filename_actor +'_' +file_str
                file_name_critic = filename_critic +'_' +file_str
                agent.save_model_and_optimizer(models_dir, filename_critic=file_name_critic , filename_actor=file_name_actor)

        #Write information to tensorboard
        writer.add_scalar('Training running avg reward', running_moving_avg_reward, episode)
        writer.add_scalar('Training running avg timesteps', running_avg_num_timesteps, episode)
        print(f"Episode {episode}/{num_episodes} Total Reward: {total_reward:.2f} Num Timesteps: {num_timesteps:.2f} Avg Reward: {running_moving_avg_reward:.2f} Avg Timesteps: {running_avg_num_timesteps:.2f}")

    #Final model save
    if save_freq is not None:
        final_str = 'Final'+datetime.datetime.now().strftime("%m%d%Y %H%M")
        file_name_actor = filename_actor + '_' + final_str
        file_name_critic = filename_critic + '_' + final_str
        agent.save_model_and_optimizer(models_dir, filename_critic=file_name_critic, filename_actor=file_name_actor)

    #Return all statistics collected
    return running_avg_rewards, running_avg_timesteps,max_reward,number_times_solved


def experiment_luna_lander(device, runs_dir, models_dir, num_episodes, config_agents, num_experiments, num_episodes_avg=20, save_freq=50, train=True, model_evaluate=None, one_model_return=False, agent_type='DDPG', render=False):
    '''
    Run multiple experiments -creating and training an agent on the Lunar lander environment
    Looping on different configurations(hyperparam search) and collect and plot experiment rewards statistics
    '''

    #Build and collect environment information
    if agent_type=='DDPG':
        env = gym.make('LunarLanderContinuous-v2')#LunarLanderContinuous-v2 BipedalWalker-v3
        action_dim = env.action_space.shape[0]
        action_space_high = env.action_space.high
        action_space_low = env.action_space.low
    else:
        env = gym.make('LunarLander-v2')
        action_dim = env.action_space.n

    observation_dim=env.observation_space.shape[0]
    dict_results={}

    #Prepare lists for results
    list_rewards_means_experiments=[]
    list_rewards_stds_experiments = []
    list_rewards_max_experiments = []
    list_timesteps_means_experiments = []
    list_timesteps_stds_experiments = []
    list_timesteps_max_experiments = []
    list_max_rewards_experiments = []
    list_solved_counts_experiments = []
    labels=[]

    #Looping on different configurations (list of dictionaries). Each configuration is a dictinary
    for config in config_agents:
        all_total_mean_reward=np.zeros((num_experiments,num_episodes))
        all_total_mean_timesteps=np.zeros((num_experiments,num_episodes))
        all_max_reward=[]
        all_solved_counts=[]

        #On each configuration running multiple experiments(parameter)
        print(f"*********Start Running config: {config['run_name']}*************")
        for experiment in range(num_experiments):
            #Prepare names of checkpoints files and writers for tensorboard
            if agent_type == 'DDPG':
                # Creating tensorboard writer for each experiment
                writer_name = os.path.join(runs_dir, 'luna_lander_ddpg_cont_' + config['run_name'] + '_expr:' + str(experiment))  # 'ddpg lunalander'
                writer = SummaryWriter(writer_name)
                actor_filename='ddpg_lunal_ACTOR' + config['run_name']+'expr:_'+str(experiment)
                critic_filename = 'ddpg_lunal_CRITIC' + config['run_name'] + 'expr:_' +str(experiment)
            else:
                # Creating tensorboard writer for each experiment
                writer_name = os.path.join(runs_dir, 'luna_lander_ppo_' + config['run_name'] + '_expr:' + str(experiment))  # 'ddpg lunalander'
                writer = SummaryWriter(writer_name)
                actor_filename = 'ppo_lunal_ACTOR' + config['run_name'] + 'expr:_' + str(experiment)
                critic_filename = 'ppo_lunal_CRITIC' + config['run_name'] + 'expr:_' + str(experiment)
            if model_evaluate is None:
                if agent_type=='DDPG':
                    #Create agent according to the configuration
                    agent=DDPG_Agent(device, observation_dim, action_dim, action_space_low, action_space_high, config, writer, models_dir,actor_filename,critic_filename)
                else:
                    agent = PPO_Agent(device, observation_dim, action_dim, config, writer,models_dir,actor_filename,critic_filename)
                #Load a saved agent if requested
                if config["checkpoint"] is not None:
                    agent.load_model_and_optimizer(models_dir, config["checkpoint"]["critic"], config["checkpoint"]["actor"])
            else:
                agent=model_evaluate
            #Running experiment episodes
            rewards, num_timesteps, max_reward,number_times_solved=run_episodes(env, agent, writer, num_episodes, num_episodes_avg=num_episodes_avg, train=train, \
                                                save_freq=save_freq, models_dir=models_dir, filename_critic=critic_filename, filename_actor=actor_filename, agnet_type=agent_type, render=render)
            #After finish running on experiment collects its statistics
            all_total_mean_reward[experiment]=rewards
            all_total_mean_timesteps[experiment]= num_timesteps
            all_max_reward.append(max_reward)
            all_solved_counts.append(number_times_solved)
        #After finishing running all experiment for a specific configuration-avarage the configuration results over all its experiments
        #collect mean max and std of this configuration
        mean_rewards=np.mean(all_total_mean_reward, axis=0)
        std_rewards = np.std(all_total_mean_reward, axis=0)
        max_rewards = np.max(all_total_mean_reward, axis=0)
        mean_timesteps = np.mean(all_total_mean_timesteps, axis=0)
        std_timesteps = np.std(all_total_mean_timesteps, axis=0)
        max_timesteps = np.max(all_total_mean_timesteps, axis=0)
        all_max_reward=np.mean(all_max_reward)
        all_solved_counts = np.mean(all_solved_counts)
        #Plot results.Save also plots to directory

        if agent_type=='DDPG':
            title_mean="Lunal_DDPG_mean_rewards_"+config['run_name']
            title_timestep = "Lunal_DDPG_mean_num_timesteps_" + config['run_name']
        else:
            title_mean = "Lunal_PPO_mean_rewards_" + config['run_name']
            title_timestep = "Lunal_PPO_mean_num_timesteps_" + config['run_name']
        #Plot configuration episodes
        plot_episodes(mean_rewards,std_rewards,max_rewards,'Average and max Rewards',title_mean, models_dir, title_mean)
        plot_episodes(mean_timesteps, std_timesteps,max_timesteps,'Average and max num timesteps',title_timestep, models_dir, title_timestep)
        list_rewards_means_experiments.append(mean_rewards)
        list_rewards_stds_experiments.append(std_rewards)
        list_rewards_max_experiments.append(max_rewards)
        list_timesteps_means_experiments.append(mean_timesteps)
        list_timesteps_stds_experiments.append(std_timesteps)
        list_timesteps_max_experiments.append(max_timesteps)
        list_max_rewards_experiments.append(all_max_reward)
        list_solved_counts_experiments.append(all_solved_counts)
        labels.append(config['run_name'])

    #finish running all configuration and statitics. Prepare results to return
    dict_results['rewards_mean']=list_rewards_means_experiments
    dict_results['rewards_std'] = list_rewards_stds_experiments
    dict_results['rewards_max'] = list_rewards_max_experiments
    dict_results['timesteps_mean'] = list_timesteps_means_experiments
    dict_results['timesteps_std'] = list_timesteps_stds_experiments
    dict_results['timesteps_max'] = list_timesteps_max_experiments
    dict_results['max_reward'] = list_max_rewards_experiments
    dict_results['solved_count'] = list_solved_counts_experiments
    #Labels are the run name-which identifies the configuration
    dict_results['labels'] = labels
    #Plot all configuration and compare (each configuration is shown with its avarges)
    if agent_type == 'DDPG':
        title='Luna lander DDPG experiments'
    else:
        title = 'Luna lander PPO experiments'
    plot_all_experiments(list_rewards_means_experiments, labels, 'Average Rewards', title,models_dir, title+'_Average_rewards')
    plot_all_experiments(list_timesteps_means_experiments, labels, 'Average num timesteps', title, models_dir, title+'_Average_timesteps')

    if one_model_return:
        return dict_results

    return dict_results




if __name__ == '__main__':

    device = torch.device('cpu')
    if (torch.cuda.is_available()):
        device = torch.device('cuda:1')

    print(device)


