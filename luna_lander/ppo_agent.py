import torch
import numpy as np
from repley_memory import ReplayMemory
from actor_critic_networks import Actor, Critic
import torch.optim as optim
import torch.nn as nn
import utils
import os
from ppo_memory import PPOMemory
from torch.distributions.multivariate_normal import MultivariateNormal




#gamma=0.99 , batch_size=64, critic_linear_sizes=[256,64], actor_linear_sizes=[256,64], tau=0.005, memory_capacity=100000,
                 #config_optim_critic={'lr':0.003}, config_optim_actor={'lr':0.003}
class PPO_Agent():

    def __init__(self,device, state_dim, action_dim, low_bound_actions, high_bound_actions, config_agent, writer, num_epochs, models_dir, checkpoint_file_actor, checkpoint_file_critic):
        super(PPO_Agent, self).__init__()

        self.state_dim=state_dim
        self.action_dim=action_dim
        self.gamma = config_agent["gamma"]
        self.batch_size = config_agent["batch_size"]
        critic_linear_sizes = config_agent["critic_linear_sizes"]
        actor_linear_sizes = config_agent["actor_linear_sizes"]
        self.tau = config_agent["tau"]

        self.config_optim_actor = config_agent["config_optim_actor"]
        self.config_optim_critic = config_agent["config_optim_critic"]

        self.replay_memory=PPOMemory(self.batch_size)

        self.device=device
        self.low_bound_actions=low_bound_actions
        self.high_bound_actions=high_bound_actions
        self.num_epochs=num_epochs


        self.Actor=Actor(state_dim, action_dim, actor_linear_sizes).to(device)
        self.Critic = Critic(state_dim, critic_linear_sizes).to(device)

        self.optimizer_critic= optim.Adam(self.Critic.parameters(), **self.config_optim_critic)
        self.optimizer_actor = optim.Adam(self.Actor.parameters(), **self.config_optim_actor)

        self.criterion_critic_mse=nn.MSELoss()

        #writer and global step initialization for tensorboard
        self.writer=writer
        self.global_step=0

        self.models_dir=models_dir
        self.checkpoint_actor=checkpoint_file_actor
        self.checkpoint_critic = checkpoint_file_critic


    def calculate_advantages(self,batch_values, batch_rewards, batch_not_done):

        #epsilon=
        advantage=0
        advantages = []
        for ind in reversed(range(batch_values.size()[0])-1):
            lambda_td_error = batch_rewards[ind] + self.gamma * batch_values[ind+1] * batch_not_done[ind]-batch_values[ind]
            new_advantage=lambda_td_error+self.gamma*self.tau*advantage
            advantages.insert(0, new_advantage)
            advantage=new_advantage

         return torch.stack(advantages)
        #rewards_to_go_norm = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + epsilon)



    def store_transition(self, observation, action, log_probs, reward, done):

        state_inner=self.convert_inner_types_to_outer(observation, flag_outer_to_inner=True)
        action_inner=self.convert_inner_types_to_outer(action, flag_outer_to_inner=True)
        log_probs_inner = self.convert_inner_types_to_outer(log_probs, flag_outer_to_inner=True)
        reward_inner = self.convert_inner_types_to_outer(np.array([reward]), flag_outer_to_inner=True)
        not_done_inner = self.convert_inner_types_to_outer(np.array([not done]), flag_outer_to_inner=True)

        self.replay_memory.push(state_inner, action_inner, log_probs_inner, reward_inner,not_done_inner)


    def convert_inner_types_to_outer(self, object, flag_outer_to_inner=False):

        if not flag_outer_to_inner:
            returned_obj=object.detach().cpu().numpy()
        else:
            returned_obj = torch.from_numpy(object).to(torch.float32).to(self.device)

        return returned_obj



    def return_action_by_policy(self, observation):


        state=self.convert_inner_types_to_outer(observation, flag_outer_to_inner=True)

        means=self.Actor(state.unsqueeze(0)).squeeze().detach().cpu()
        cov_matrix=torch.ones(self.action_dim)*(self.std**2)
        distribution=MultivariateNormal(means, cov_matrix)

        action=distribution.sample()
        log_prob = distribution.log_prob(action)

        action=self.convert_inner_types_to_outer(action)
        log_prob = self.convert_inner_types_to_outer(log_prob)

        return action, log_prob

    def save_model_and_optimizer(self, models_dir, filename_critic=None, filename_actor=None):

        if filename_critic:
            utils.save_model(self.Critic, self.optimizer_critic, models_dir, filename_critic, self.Critic_target)

        if filename_actor:
            utils.save_model(self.Actor, self.optimizer_actor, models_dir, filename_actor, self.Actor_target)


    def load_model_and_optimizer(self, models_dir, filename_critic=None, filename_actor=None):

        if filename_critic:
            self.Critic, self.optimizer_critic, self.Critic_target=utils.load_model(models_dir, filename_critic)

        if filename_actor:
            self.Actor, self.optimizer_actor, self.Actor_target=utils.load_model(models_dir, filename_actor)



    def train(self):
        #Target network are used to calcualte the target values. Therefore in eval mode
        epsilon = 1e-5

        if len(self.replay_memory)<self.batch_size:
            return

        self.replay_memory.permute_batches()
        discounted_reward=0


        for epoch in self.num_epochs:

            for ind, batch_states, batch_actions, batch_log_probs, batch_rewards, batch_values, batch_not_done in enumerate(self.replay_memory.sample_batch(self.batch_size)):


                advantages=self.calculate_advantages(batch_values, batch_rewards, batch_not_done)

                advantages=advantages.detach()
                self.optimizer_critic.zero_grad()

                mu, std=self.Actor(batch_states)

                new_values=self.Critic(batch_states)

                distribution=self.distribution(mu, std)

                new_log_prob=distribution.log_prob(mu, std)

                ratios=new_log_prob.exp()/batch_log_probs.exp()

                cliped_ratios=torch.clamp(ratios, 1.-self.clip_value, 1+self.clip_value)
                updated_advantage=ratios*advantages

                actor_loss=-torch.min(ratios,cliped_ratios ).mean()
                critic_loss=self.critic_loss_mse(new_values, batch_values)

                combined_loss=0.5*actor_loss+0.5*critic_loss

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                combined_loss.loss.backward()
                self.optimizer_actor.step()
                self.optimizer_critic.step()

                self.writer.add_scalar('Training ppo loss', combined_loss.item(), self.global_step_critic)
                self.global_step += 1





