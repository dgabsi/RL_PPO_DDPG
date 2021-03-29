import numpy as np
import matplotlib.pyplot as plt
import u_got_package
import utils
import os

class QLearnAgent(u_got_package.Policy):
    '''
    Agent that acts according to the q learning algorithm. Acting according to an e-greedy policy and calculating q values according to q learning
    The class is also a policy(e-greedy) , inheriting from Policy class
    '''
    def __init__(self, num_observations, num_actions, gamma, step_size, epsilon, epsilon_decay, epsilon_min):
        '''
        Constructor . Initialization of attributes. Creating the q values matrix
        :param num_observation- observation space size(int)
                num_actions
                gamma-discount factor (according to Belmman equations
                step-size- step-size(alpha) in the q learning
                epsilon-exploration rate
                epsilon_decay- decay rate of the epsilon .multiplication fact (between 0-1)
                epsilon_min-minimum value for epsilon
        '''


        super(QLearnAgent, self).__init__()
        self.num_actions=num_actions
        self.epsilon_decay=epsilon_decay
        self.num_observations=num_observations
        self.start_epsilon=epsilon
        self.epsilon = epsilon
        self.epsilon_min=epsilon_min
        self.gamma = gamma
        self.step_size = step_size
        self.q_values=np.zeros((self.num_observations,self.num_actions))
        self.reset_agent()


    def get_action_by_policy(self, observation_index):
        '''
        With probability 1-e Returns the highest value action according to a q-values array,
        and with probability e returns a random action
        :param observation_index (int)
        :return: action_index(int)
        '''

        if np.random.rand() < self.epsilon:  # with e probability pick a random action
            action_index = np.random.randint(self.num_actions)

        else:
            index_max_action = np.argwhere(
                self.q_values[observation_index, :] == np.max(self.q_values[observation_index, :]))

            if len(
                    index_max_action) > 1:  # In case there is more than 1 action with the same max value-return a random action among them.
                action_index = int(np.random.choice(index_max_action.squeeze()))
            else:
                action_index = int(index_max_action.squeeze())

        return action_index

    def decay_epsilon(self):
        '''
        Multiply the epsilon by the decay rate. limited by the minimum value for the epsilon
        :return:None
        '''

        new_epsilon = self.epsilon * self.epsilon_decay
        if new_epsilon > self.epsilon_min:
            self.epsilon = new_epsilon
        else:
            self.epsilon = self.epsilon_min


    def reset_agent(self):
        '''
        Reseting the agent. Zero out q values table and return epsilon to start value
        '''
        #self.reset_episode()
        self.q_values = np.zeros((self.num_observations, self.num_actions))
        self.epsilon = self.start_epsilon


    def train(self, observation, action, next_observation, reward):
        '''
        Update q-values according to the transition parameter. Q learning bootstrapping and calculating q values according to Bellman optimality equation(next max value)
        :param observation:
        :param action:
        :param next_observation:
        :param reward:
        :return:
        '''

        #Updating the previous move
        max_qvalue_observation_next=max(self.q_values[next_observation, :])
        self.q_values[observation, action]+=self.step_size*(reward+self.gamma*max_qvalue_observation_next-self.q_values[observation, action])
        #print(self.q_values[observation, action])
        #Update states agent

        #self.prev_state=state
        #self.prev_action=action


    def save_agent_to_pickle(self, models_dir, filename):

        filename = os.path.join(models_dir, filename)
        utils.save_to_pickle(self, filename)

    @staticmethod
    def load_agent_from_pickle(models_dir, filename):

        filename = os.path.join(models_dir, filename)
        agent=utils.load_from_pickle(filename)

        return agent




