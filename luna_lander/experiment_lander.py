import gym
from agent_ddpg import DDPG_Agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import datetime
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import utils


SAVED_DIR='./saved'
RUNS_DIR='./experiments'
NUM_EPISODES=1000

#configuration according to ddpg paper
CONFIGS_LUNA_LANDER=[{'run_name':"critic:400-300 actor:400-300 lr actor 5e-5 lr critic 5e-4", 'batch_size':64, "critic_linear_sizes":[400,300], "actor_linear_sizes":[400,300],  "config_optim_actor":{'lr':0.00005},\
                        "config_optim_critic":{"lr":0.0005}, "memory_capacity":1000000, "tau": 0.001, "gamma": 0.99, "checkpoint":None},
                     {'run_name':"critic:600-300 actor:600-300 lr actor 5e-5 lr critic 5e-4", 'batch_size':64, "critic_linear_sizes":[600,300], "actor_linear_sizes":[600,300],  "config_optim_actor":{'lr':0.00005},\
                        "config_optim_critic":{"lr":0.0005}, "memory_capacity":1000000, "tau": 0.001, "gamma": 0.99, "checkpoint":None},
                     {'run_name':"critic:400-300 actor:400-300 lr actor 3e-5 lr critic 3e-4", 'batch_size':64, "critic_linear_sizes":[400,300], "actor_linear_sizes":[400,300],  "config_optim_actor":{'lr':0.00003},\
                        "config_optim_critic":{"lr":0.0003}, "memory_capacity":1000000, "tau": 0.001, "gamma": 0.99, "checkpoint":None},
                     {'run_name':"critic:600-300 actor:600-300 lr actor 3e-5 lr critic 3e-4", 'batch_size':64, "critic_linear_sizes":[600,300], "actor_linear_sizes":[600,300],  "config_optim_actor":{'lr':0.00003},\
                        "config_optim_critic":{"lr":0.0003}, "memory_capacity":1000000, "tau": 0.001, "gamma": 0.99, "checkpoint":None}]


NUM_EXPERIMENTS=2
BATCH_SIZE=64

#AGENT_config_bipedal={'run_name':"critic:400-300 actor:400-300 bs:64", 'batch_size':64, "critic_linear_sizes":[400,300], "actor_linear_sizes":[400,300],  "config_optim_actor":{'lr':0.0001},
#"config_optim_critic":{"lr":0.001, 'weight_decay': 0.001}, "memory_capacity":1000000, "tau": 0.001, "gamma": 0.99}

def run_episodes(env, agent, writer, num_episodes, num_episodes_avg=20, train=False, save_freq=50, models_dir=None, filename_critic=None, filename_actor=None, save_rewards=150):


    all_total_rewards = []
    all_num_timesteps = []
    running_avg_rewards=[]
    running_avg_timesteps = []

    for episode in range(num_episodes):
        observation=env.reset()
        done=False
        total_reward=0
        num_timesteps=0
        while not done:
            #env.render()
            action=agent.return_action_by_policy(observation, train=True)
            observation_next, reward, done, info=env.step(action)
            if train:
                agent.store_transition(observation, action, observation_next, reward, done)
                agent.train()
            total_reward+=reward
            observation=observation_next
            num_timesteps+=1

        all_total_rewards.append(total_reward)
        all_num_timesteps.append(num_timesteps)
        running_moving_avg_reward=np.mean(all_total_rewards[-num_episodes_avg:])
        running_avg_num_timesteps=np.mean(all_num_timesteps[-num_episodes_avg:])
        running_avg_rewards.append(running_moving_avg_reward)
        running_avg_timesteps.append(running_avg_num_timesteps)
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

        writer.add_scalar('Training running avg reward', running_moving_avg_reward, episode)
        writer.add_scalar('Training running avg timesteps', running_avg_num_timesteps, episode)
        print(f"Episode {episode}/{num_episodes} Total Reward: {total_reward} Num Timesteps: {num_timesteps} Avg Reward: {running_moving_avg_reward} Avg Timesteps: {running_avg_num_timesteps}")

    final_str = 'Final'+datetime.datetime.now().strftime("%m%d%Y %H%M")
    file_name_actor = filename_actor + '_' + final_str
    file_name_critic = filename_critic + '_' + final_str
    agent.save_model_and_optimizer(models_dir, filename_critic=file_name_critic, filename_actor=file_name_actor)

    return running_avg_rewards, running_avg_timesteps


def train_luna_lander(device, runs_dir, models_dir, num_episodes, config_agents, num_experiments, num_episodes_avg=10, save_freq=50):

    env = gym.make('LunarLanderContinuous-v2')#LunarLanderContinuous-v2 BipedalWalker-v3

    observation_dim=env.observation_space.shape[0]
    action_dim=env.action_space.shape[0]
    action_space_high = env.action_space.high
    action_space_low = env.action_space.low
    dict_results={}

    list_rewards_means_experiments=[]
    list_rewards_stds_experiments = []
    list_timesteps_means_experiments = []
    list_timesteps_stds_experiments = []
    labels=[]

    for config in config_agents:
        all_total_mean_reward=np.zeros((num_experiments,num_episodes))
        all_total_mean_timesteps=np.zeros((num_experiments,num_episodes))


        for experiment in range(num_experiments):
            writer_name = os.path.join(runs_dir, 'luna_lander_ddpg_cont_'+ config['run_name']+'_expr:'+str(experiment)) #'ddpg lunalander'
            writer = SummaryWriter(writer_name)
            actor_filename='ddpg_lunal_actor' + config['run_name']+'expr:_'+str(experiment)
            critic_filename = 'ddpg_lunal_critic' + config['run_name'] + 'expr:_' +str(experiment)
            agent=DDPG_Agent(device, observation_dim, action_dim, action_space_low, action_space_high, config, writer, SAVED_DIR,actor_filename,critic_filename)
            if config["checkpoint"] is not None:
                agent.load_model_and_optimizer(models_dir, config["checkpoint"])
            rewards, num_timesteps=run_episodes(env, agent, writer, num_episodes, num_episodes_avg=num_episodes_avg, train=True, \
                                                save_freq=save_freq, models_dir=models_dir, filename_critic=critic_filename, filename_actor=actor_filename)
            all_total_mean_reward[experiment]=rewards
            all_total_mean_timesteps[experiment]= num_timesteps
        mean_rewards=np.mean(all_total_mean_reward, axis=0)
        std_rewards = np.std(all_total_mean_reward, axis=0)
        mean_timesteps = np.mean(all_total_mean_timesteps, axis=0)
        std_timesteps = np.std(all_total_mean_timesteps, axis=0)
        title="Lunal_DDPG_mean_rewards\n"+config['run_name']
        utils.plot_episodes(mean_rewards,std_rewards,'Average Rewards',title, models_dir, title)
        title = "Lunal_DDPG_mean_num_timesteps\n" + config['run_name']
        utils.plot_episodes(mean_timesteps, std_timesteps,'Average num timesteps',title, models_dir, title)
        list_rewards_means_experiments.append(mean_rewards)
        list_rewards_stds_experiments.append(std_rewards)
        list_timesteps_means_experiments.append(mean_timesteps)
        list_timesteps_stds_experiments.append(std_timesteps)
        labels.append(config['run_name'])

    dict_results['rewards_mean']=list_rewards_means_experiments
    dict_results['rewards_std'] = list_rewards_stds_experiments
    dict_results['timesteps_mean'] = list_timesteps_means_experiments
    dict_results['timesteps_std'] = list_timesteps_stds_experiments
    dict_results['labels'] = labels
    title='Luna lander DDPG experiments'
    utils.plot_all_experiments(list_rewards_means_experiments,list_rewards_stds_experiments, labels, 'Average Rewards', title,models_dir, title)
    utils.plot_all_experiments(list_timesteps_means_experiments, list_timesteps_stds_experiments, labels, 'Average num timesteps', title, models_dir, title)


    return dict_results



def evaluate_ddpg(env, num_experiments, num_episodes, agent_checkpint):
    pass

if __name__ == '__main__':

    device = torch.device('cpu')
    if (torch.cuda.is_available()):
        device = torch.device('cuda:1')

    print(device)

    dict_results=train_luna_lander(device, RUNS_DIR, SAVED_DIR, NUM_EPISODES, CONFIGS_LUNA_LANDER, NUM_EXPERIMENTS)
    utils.save_to_pickle(dict_results, os.path.join(SAVED_DIR, 'all_results_dict.pkl'))