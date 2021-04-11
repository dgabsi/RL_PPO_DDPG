import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import os
from .ppo_memory import PPOMemory
from .ppo_networks import Actor, Critic
from .utils import save_model, load_model


class PPO_Agent():
    '''
        This class is an implementation of PPO algorithms.
        The agent has two networks - Actor the calculates a probability distribution of the action and a critic that approximates the q value
        The critic is learned through bellman equations on sampled transitions and calculating reward to go based on the trajectory
        The actor is learned by directing the distribution in the direction leading to greater value advantage
        The agent holds a experience memory buffer.
        The agent is also a policy class- which is implemented by outputting the actor policy -sampling from the distribution.
        The learning is done using by running epochs on mini batches from the memory trajectories.

        '''

    def __init__(self, device, state_dim, action_dim, config_agent, writer, models_dir, checkpoint_file_actor, checkpoint_file_critic):
        '''
        Initialization of inner attributes. Creating the actor and critic networks, optimizers,loss function
        :param device: device
        :param state_dim: dimension size of states
        :param action_dim: dimension size of actions
        :param config_agent: configuration dictionary that holds parameters of the agent- optimizer params:(learning rate), batch_size
        :param writer: writer for tensorboard
        :param models_dir: directory for saving the models
        :param checkpoint_file_actor:
        :param checkpoint_file_critic:
        '''

        super(PPO_Agent, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config_agent["gamma"]
        self.batch_size = config_agent["batch_size"]
        critic_linear_sizes = config_agent["critic_linear_sizes"]
        actor_linear_sizes = config_agent["actor_linear_sizes"]

        self.num_epochs = config_agent["num_epochs"]

        self.capacity = config_agent["capacity"]
        self.lambda_smoothing = config_agent["lambda_smoothing"]

        self.config_optim = config_agent["config_optim"]

        self.ppo_memory = PPOMemory(self.batch_size)

        self.device = device

        self.Actor = Actor(state_dim, action_dim, actor_linear_sizes).to(device)
        self.Critic = Critic(state_dim, critic_linear_sizes).to(device)


        #Joint optimizer for actor and critic
        self.optimizer_actor_critic = optim.Adam(list(self.Actor.parameters()) + list(self.Critic.parameters()), **self.config_optim)

        self.criterion_critic_mse_loss=nn.MSELoss()

        # writer and global step initialization for tensorboard
        self.writer = writer
        self.global_step = 0

        self.models_dir = models_dir
        self.checkpoint_actor = checkpoint_file_actor
        self.checkpoint_critic = checkpoint_file_critic

    def calculate_rewards2g_and_advantages(self):
        '''
        Calculate advantages and rewards2fo for all transition in memory and store back to the PPO memory for fetching
        :return:
        '''


        #epsilon_norm = 1e-5

        values = self.ppo_memory.memory['value']
        rewards = self.ppo_memory.memory['reward']
        not_done = self.ppo_memory.memory['not done']

        rewards_to_go = []
        advantages = []

        advantage_next = torch.zeros(1).to(self.device)
        values_next = values[-1]
        # advantages.insert(0, advantage)
        # rewards_to_go.insert(0, values_next)

        #loop in reverse on the values in memory and calculate the the advantages.
        #We loop in reverse because the values in the future affect the former advanatgea
        #Rewards2go  are the advanatge+value
        for ind in reversed(range(len(values)-1)):
            delta = rewards[ind] + self.gamma * values_next * not_done[ind] - values[ind]

            advantage = delta + self.gamma * self.lambda_smoothing * not_done[ind] * advantage_next
            advantages.insert(0, advantage)

            rewards_to_go.insert(0, advantage + values[ind])

            values_next = values[ind]
            advantage_next = advantage



        #mean=torch.mean(torch.tensor(rewards_to_go))
        #std=torch.std(torch.tensor(rewards_to_go))

        #stack_r2g=torch.stack(rewards_to_go)

        # normalising the rewards

        #rewards_to_go_norm = (stack_r2g - mean) / (std-epsilon_norm)
        #rewards_to_go_norm=list(torch.unbind(rewards_to_go_norm))



        #Store the caculated values in memory
        self.ppo_memory.add_advanatges_and_rewards2go(advantages, rewards_to_go)


        return



    def store_transition(self, observation, action, log_probs, entropy, value, reward, done):
        '''
        Store transition in the ppo memory buffer. We will first convert the value from outside(numpy) to inner values (tensors)
        :param observation:
        :param action:
        :param log_probs:
        :param entropy
        :params value
         :param reward:
        :param done:
        :return:
        '''

        state_inner = self.convert_inner_types_to_outer(observation, flag_outer_to_inner=True)
        action_inner = self.convert_inner_types_to_outer(action, flag_outer_to_inner=True)
        log_probs_inner = self.convert_inner_types_to_outer(log_probs, flag_outer_to_inner=True)
        entropy_inner = self.convert_inner_types_to_outer(entropy, flag_outer_to_inner=True)

        value_inner = self.convert_inner_types_to_outer(np.array([value]), flag_outer_to_inner=True)
        reward_inner = self.convert_inner_types_to_outer(np.array([reward]), flag_outer_to_inner=True)
        not_done_inner = self.convert_inner_types_to_outer(np.array([not done]), flag_outer_to_inner=True)

        self.ppo_memory.push(state_inner, action_inner, log_probs_inner, entropy_inner, value_inner, reward_inner,
                             not_done_inner)

    def convert_inner_types_to_outer(self, object, flag_outer_to_inner=False):
        '''
        Convert from numpy values to tensors and vise versa
        :param object:
        :param flag_outer_to_inner: if True it output from outer to inner (usually we output inner to outer)
        :return:
        '''

        if not flag_outer_to_inner:
            returned_obj = object.detach().cpu().numpy()
        else:
            returned_obj = torch.from_numpy(object).to(torch.float32).to(self.device)

        return returned_obj

    def return_action_by_policy(self, observation, train=True):
        '''
        Return an action according to the current policy(We sample from the current actor distribution )
        :param observation:
        :param train:
        :return: Action , log probability of the action entropy of the distribution
        '''

        state = self.convert_inner_types_to_outer(observation, flag_outer_to_inner=True)

        self.Actor.eval()
        #Calculate action distrbution according to state-using the actor network
        distr = self.Actor(state.unsqueeze(0))

        #Sampling an action from the distribution
        action = distr.sample().squeeze()

        #Take the log probability of the action in the distribution
        log_prob = distr.log_prob(action)

        #Added entropy but it is not used in final implementation
        entropy = distr.entropy()

        action = self.convert_inner_types_to_outer(action)
        log_prob = self.convert_inner_types_to_outer(log_prob)
        entropy = self.convert_inner_types_to_outer(entropy)
        self.Actor.train()

        return action, log_prob, entropy


    def get_value(self, observation):
        '''
        get value of the observation according tot he critic
        :param observation:
        :return: value of the observation according to critic
        '''

        state = self.convert_inner_types_to_outer(observation, flag_outer_to_inner=True)
        self.Critic.eval()
        values = self.Critic(state.unsqueeze(0)).squeeze()
        values = self.convert_inner_types_to_outer(values)
        self.Critic.train()

        return values



    def train(self):
        '''
        train the ppo agent . calculate advantages and rewards to go from the trajectories in memory.
        then run a few epochs  in each epoch we loop over the batches(in a random way) and caculate critic and actor loss and do gradient descent
        The critic values the compared to the rewards2go and the actor ratio between new log probability and former log probability is limited and cliiped according to
        PPO algorithm
        :return:
        '''

        #Clip 0.2 value -best according to PPO paper
        clip_value = 0.2


        if len(self.ppo_memory) < self.capacity:
            return


        #Calculating the advanatges and reward to go of the trajectories in the memory
        self.calculate_rewards2g_and_advantages()

        for epoch in range(self.num_epochs):
            # print(epoch)

            self.ppo_memory.permute_batches()
            for batch_states, batch_actions, batch_log_probs, batch_entropy, batch_advantages, batch_rewards2go in self.ppo_memory.generate_batch():


                batch_rewards2go.detach()
                batch_advantages.detach()
                batch_log_probs.detach()
                batch_actions.detach()
                batch_states.detach()

                # Important - In the final implementation I didnt use the entropy - but it is strored from my previous attempt

                self.optimizer_actor_critic.zero_grad()
               # self.optimizer_actor.zero_grad()

                new_values = self.Critic(batch_states)

                #Critic loss is mse from target reward to go
                critic_loss = self.criterion_critic_mse_loss( new_values, batch_rewards2go) #((batch_rewards2go - new_values) ** 2).mean()

                # Actor loss

                # calculate distribution
                distr = self.Actor(batch_states)

                # Take probabilites for current action according to actor
                new_log_prob = distr.log_prob(batch_actions.squeeze())

                entropy = distr.entropy().squeeze().mean()

                # ratios according to PPO
                ratios = (new_log_prob-batch_log_probs.squeeze()).exp()
                # print(ratios)

                updated_advantages = ratios * batch_advantages.squeeze()

                # clipping the ratios- PPO
                cliped_advatages = torch.clamp(ratios, 1. - clip_value, 1 + clip_value) * batch_advantages.squeeze()

                actor_loss = -torch.min(updated_advantages, cliped_advatages).mean()

                #0.4- value for c1 from the PPO paper
                combined_loss = actor_loss + 0.5*critic_loss #-0.001*entropy


                combined_loss.backward()
                self.optimizer_actor_critic.step()


                #For Tensorboard
                self.writer.add_scalar('Training ppo loss', combined_loss.item(), self.global_step)
                self.writer.add_scalar('Training actor loss', actor_loss.item(), self.global_step)
                self.writer.add_scalar('Training critic loss', critic_loss.item(), self.global_step)


                self.global_step += 1

        #Empty the memory
        self.ppo_memory.reset()

    def save_model_and_optimizer(self, models_dir, filename_critic, filename_actor):
        '''
            Save critic and actor networks and optimizer to a checkpoint file
            :param models_dir:
            :param filename_critic: checkpoint file
            :param filename_actor: checkpoint file
            :return:
        '''
        save_model(self.Critic, self.optimizer_actor_critic, models_dir, filename_critic)

        save_model(self.Actor, self.optimizer_actor_critic, models_dir, filename_actor)





    def load_model_and_optimizer(self, models_dir, filename_critic, filename_actor):
        '''
            Load critic and actor networks and optimizer from a checkpoint file
            :param models_dir:
            :param filename_critic: checkpoint file
            :param filename_actor: checkpoint file
            :return:
        '''

        critic_state_dict, optimizer_actor_critic_state_dict, _= load_model(models_dir, filename_critic)
        actor_state_dict, _, _ = load_model(models_dir, filename_actor)
        self.Critic.load_state_dict(critic_state_dict)
        self.optimizer_actor_critic.load_state_dict(optimizer_actor_critic_state_dict)
        self.Actor.load_state_dict(actor_state_dict)

