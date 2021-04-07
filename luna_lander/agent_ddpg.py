import torch
import numpy as np
from .replay_memory import ReplayMemory
from .ddpg_actor_critic_networks import Actor, Critic
import torch.optim as optim
import torch.nn as nn
import utils
import os



def au_noise(x, mean=0, std=0.2,  theta=0.15, dt=1e-2):

    noised_x= x+ theta*(mean-x)*dt+std*np.sqrt(dt)*np.random.normal(size=x.shape)

    return noised_x


#gamma=0.99 , batch_size=64, critic_linear_sizes=[256,64], actor_linear_sizes=[256,64], tau=0.005, memory_capacity=100000,
                 #config_optim_critic={'lr':0.003}, config_optim_actor={'lr':0.003}
class DDPG_Agent():
    '''
    This class is an implementation of DDPG algorithms.
    The agent learns the policy by using two networks-an Actor and a critic.
    The Actor is responsible to calculating the policy
    The critic is determines via approximating q values whether the actor policy performs weell.
    The critic is learned through bellman equations on sampled transitions
    The actor is learned by performing gradient ascent on the critic values.
    The agent holds a experience memory buffer.
    The agent is also a policy class- which is implemented by outputting the actor policy.
    The two actor and critic netowrks have two copies.A worked network and a target network.
    The learning is done on the local networks and after each learning there is a soft copy to the target netowrks(averaged copy according to a tau parameter).
    The soft copy and use of target networks reduces the variance of the learning processes
    '''

    def __init__(self,device, state_dim, action_dim, low_bound_actions, high_bound_actions, config_agent, writer, models_dir, checkpoint_file_actor, checkpoint_file_critic):
        '''
        Initzaition of inner attributes. Creating the actor and critic networks, optimizers,loss function
        :param device: device
        :param state_dim: dimension size of states
        :param action_dim: dimension size of actions
        :param low_bound_actions: low bounds values for actions
        :param high_bound_actions: high bounds values for actions
        :param config_agent: configuration dictionary that holds parameters- optimizer params:(learning rate, weight decay), batch_size
        :param writer: writer for tensorboard
        :param models_dir: directory for saving the models
        :param checkpoint_file_actor:
        :param checkpoint_file_critic:
        '''


        super(DDPG_Agent, self).__init__()

        self.state_dim=state_dim
        self.action_dim=action_dim

        self.gamma = config_agent["gamma"]
        self.batch_size = config_agent["batch_size"]
        critic_linear_sizes = config_agent["critic_linear_sizes"]
        actor_linear_sizes = config_agent["actor_linear_sizes"]
        self.memory_capacity = config_agent["memory_capacity"]
        self.tau = config_agent["tau"]
        self.config_optim_actor = config_agent["config_optim_actor"]
        self.config_optim_critic = config_agent["config_optim_critic"]



        self.repley_memory=ReplayMemory(self.memory_capacity)

        self.device=device
        self.low_bound_actions=low_bound_actions
        self.high_bound_actions=high_bound_actions

        self.noise= np.zeros(action_dim) #OUActionNoise(mu=np.zeros(action_dim))  #torch.zeros(action_dim)

        #Creating actor networks
        self.Actor=Actor(state_dim,action_dim, actor_linear_sizes).to(device) #Actor(state_dim, action_dim, actor_linear_sizes).to(device)
        self.Actor_target=  Actor(state_dim,action_dim, actor_linear_sizes).to(device)#Actor(state_dim, action_dim, actor_linear_sizes).to(device)
        #self.Actor_target.eval()

        #copy actor to actor target network
        self.Actor_target.load_state_dict(self.Actor.state_dict())
        #self.update_network_parameters(1)

        # Creating critic networks
        self.Critic = Critic(state_dim,action_dim, critic_linear_sizes).to(device)# CrCritic(state_dim, action_dim,  critic_linear_sizes).to(device)
        self.Critic_target=Critic(state_dim,action_dim, critic_linear_sizes).to(device) #Critic(state_dim, action_dim, critic_linear_sizes).to(device)
        #self.Critic_target.eval()
        # copy actor to critic target network
        self.Critic_target.load_state_dict(self.Critic.state_dict())

        #self.update_network_parameters(1)

        self.optimizer_critic= optim.Adam(self.Critic.parameters(), **self.config_optim_critic)
        self.optimizer_actor = optim.Adam(self.Actor.parameters(), **self.config_optim_actor)

        #loss criterion for critic
        self.criterion_critic_mse=nn.MSELoss()

        #writer and global step initialization for tensorboard
        self.writer=writer
        self.global_step_critic=0
        self.global_step_actor = 0

        self.models_dir=models_dir
        self.checkpoint_actor=checkpoint_file_actor
        self.checkpoint_critic = checkpoint_file_critic


    def update_target_networks(self, tau):
        '''
        Soft copying of local networks to target networks(moving average). target_network=tau*(local_network)+(1-tau)*target_netowrk
        :param tau: value between 1-0 for the moving average
        :return: None
        '''

        #Copy actor
        for actor_target_param, actor_param in zip(self.Actor_target.parameters(), self.Actor.parameters()):
            actor_target_param.data=(tau * actor_param.data + (1.0 - tau) * actor_target_param.data)

        #Copy critic
        for critic_target_param, critic_param in zip(self.Critic_target.parameters(), self.Critic.parameters()):
            critic_target_param.data=(tau * critic_param.data + (1.0 - tau) * critic_target_param.data)




    def store_transition(self, observation, action, observation_next, reward, done):
        '''
        Store transition in the replay buffer. We will first convert the value from outside(numpy) to inner values (tensors)
        :param observation:
        :param action:
        :param observation_next:
        :param reward:
        :param done:
        :return:
        '''

        state_inner=self.convert_inner_types_to_outer(observation, flag_outer_to_inner=True)
        action_inner=self.convert_inner_types_to_outer(action, flag_outer_to_inner=True)
        state_next_inner = self.convert_inner_types_to_outer(observation_next, flag_outer_to_inner=True)
        reward_inner = self.convert_inner_types_to_outer(np.array([reward]), flag_outer_to_inner=True)
        #Pay attention- We change done to not done (becuase we want not done to have 1 values(not terminal transitions) for bellman.
        not_done_inner = self.convert_inner_types_to_outer(np.array([not done]), flag_outer_to_inner=True)

        self.repley_memory.push(state_inner, action_inner, state_next_inner, reward_inner,not_done_inner)


    def convert_inner_types_to_outer(self, object, flag_outer_to_inner=False):
        '''
        Convert from numpy values to tensors and vise versa
        :param object:
        :param flag_outer_to_inner: if True it output from outer to inner (usually we output inner to outer)
        :return:
        '''

        if not flag_outer_to_inner:
            returned_obj=object.detach().cpu().numpy()
        else:
            returned_obj = torch.from_numpy(object).to(torch.float32).to(self.device)

        return returned_obj



    def return_action_by_policy(self, observation, train=False):
        '''
        Return an action according to the current policy(Which is the current actor)
        :param observation:
        :param train:
        :return: action
        '''

        self.Actor.eval()
        #Convert from numpy form to torch
        state=self.convert_inner_types_to_outer(observation, flag_outer_to_inner=True)
        with torch.no_grad():
            action=self.Actor.forward(state.unsqueeze(0)).squeeze().detach().cpu()
            if train:
                #If in train mode we add a noise to induce  exploration
                self.noise=au_noise(self.noise)
                action=self.convert_inner_types_to_outer(action)+self.noise
            else:
                action = self.convert_inner_types_to_outer(action)
        self.Actor.train()
        #Clipping the actions values to the boundaries permitted (although this is not essential since we use the Tanh..)
        cliped_action=np.clip(action, self.low_bound_actions, self.high_bound_actions)

        return cliped_action

    def save_model_and_optimizer(self, models_dir, filename_critic=None, filename_actor=None):
        '''
        Save critic and actor networks and optimizer to a checkpoint file (saving both local and targets)
        :param models_dir:
        :param filename_critic: If not None save critic networks to this file
        :param filename_actor: If not None save actor networks to this file
        :return:
        '''

        if filename_critic:
            utils.save_model(self.Critic, self.optimizer_critic, models_dir, filename_critic, self.Critic_target)

        if filename_actor:
            utils.save_model(self.Actor, self.optimizer_actor, models_dir, filename_actor, self.Actor_target)


    def load_model_and_optimizer(self, models_dir, filename_critic=None, filename_actor=None):
        '''
        Load critic and actor networks and optimizer from a checkpoint file (saving both local and targets)
        :param models_dir:
        :param filename_critic: if not None load from this critic networks
        :param filename_actor: if not None load from this actor networks
        :return:
        '''

        if filename_critic:
            critic_state_dict, optimizer_critic_state_dict, critic_target_state_dict=utils.load_model(models_dir, filename_critic)
            self.Critic.load_state_dict(critic_state_dict)
            self.optimizer_critic.load_state_dict(optimizer_critic_state_dict)
            self.Critic_target.load_state_dict(critic_target_state_dict)

        if filename_actor:
            actor_state_dict, optimizer_actor_state_dict, actor_target_state_dict=utils.load_model(models_dir, filename_actor)
            self.Actor.load_state_dict(actor_state_dict)
            self.optimizer_actor.load_state_dict(optimizer_actor_state_dict)
            self.Actor_target.load_state_dict(actor_target_state_dict)

    def update_auc_noise(self, mean=0, std=0.2,  theta=0.15, dt=1e-2):

        action_dim=torch.ones(self.action_dim)

        mean=mean*action_dim
        std=std*action_dim

        added_noise= self.noise+ theta*(mean-self.noise)*dt+std*np.sqrt(dt)*np.random.normal(size=mean.shape)
        self.noise=added_noise

        return self.noise


    def train(self):
        '''
        Main learning function
        '''

        if len(self.repley_memory)<self.batch_size:
            return

        batch_state, batch_action, batch_state_next, batch_reward, batch_not_done=self.repley_memory.sample_batch(self.batch_size)

        #Target network are used to calcualte the target values. Therefore in eval mode
        #self.Actor_target.eval()
        #self.Critic_target.eval()

        #train critic-train the calulation ove the q values
        #Based on q learning

        #calculating target_ q learning values
        with torch.no_grad():
            batch_action_next=self.Actor_target(batch_state_next).detach()
            batch_q_values_next=self.Critic_target(batch_state_next, batch_action_next).detach()
        #This will be the target for the critic
        batch_target_q_values=batch_reward+self.gamma*batch_q_values_next*batch_not_done

        self.Critic.train()
        self.optimizer_critic.zero_grad()
        batch_q_values_curr = self.Critic(batch_state, batch_action)
        loss_critic=self.criterion_critic_mse(batch_q_values_curr, batch_target_q_values.detach())
        loss_critic.backward()
        #for param in self.Critic.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer_critic.step()
        self.writer.add_scalar('Training critic running loss', loss_critic.item(), self.global_step_critic)
        #if not self.global_step_critic % 100:
        #    print(f"Critic step {self.global_step_critic} Train critic Q values running loss: {loss_critic.item():.2f}")


        #Train Actor- training the policy
        self.Actor.train()
        self.Critic.eval()
        self.optimizer_actor.zero_grad()

        q_values_actor_policy= -self.Critic(batch_state, self.Actor(batch_state)).mean()
        q_values_actor_policy.backward()

        #for param in self.Actor.parameters():
        #    param.grad.data.clamp_(-1, 1)

        self.optimizer_actor.step()
        self.writer.add_scalar('Training actor avg Q values', -(q_values_actor_policy.item()), self.global_step_actor)
        self.Critic.train()

        #if not self.global_step_actor%100:
        #    print(f"Actor step {self.global_step_actor} Training actor avg Q values: {-q_values_actor_policy.item():.2f}")

        self.update_target_networks(self.tau)

        self.global_step_critic+=1
        self.global_step_actor += 1

        #if self.checkpoint_actor is not None:
        #    if not self.global_step_actor%1000:
        #        self.save_model_and_optimizer(self.models_dir, filename_actor=self.checkpoint_actor)

        #if self.checkpoint_critic is not None:
        #    if not self.global_step_critic%1000:
         #       self.save_model_and_optimizer(self.models_dir, filename_critic=self.checkpoint_critic)


