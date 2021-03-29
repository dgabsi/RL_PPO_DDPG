import numpy as np


class Policy():
    '''
    Base class for a policy. The main functinality is to return an next action according to an observation
    '''
    def get_action_by_policy(self, observation_index):
        '''
        Basic function for policy. Returns an action according to an observation
        :param observation_index: index of the observation (int)
        :return: returns and action_index (int)
        '''
        return 0



class RandomPolicy(Policy):
    '''
    Random policy. Returns a random action
    '''
    def __init__(self, num_actions):
        super(RandomPolicy, self).__init__()

        self.num_actions=num_actions

    def get_action_by_policy(self, observation_index):
        '''
        Returns a random action
        :param observation_index (int)
        :return: random action (int)
        '''
        action_index = np.random.randint(4)

        return action_index





class EgreedyPolicy(Policy):
    '''
        e-greedy policy. With probability 1-e Returns the highest value action according to a q-values array,
        and with probability e returns a random action.e decays over time
    '''

    def __init__(self, num_actions, epsilon, decay_rate, epsilon_min, q_values):
        '''
        Initialization of attributes
        :param epsilon: probability of picking a random action
        :param decay_rate: multiplication value of the e. Usually under 1. for decay.
        :param epsilon_min: minimum value of epsilon. limits the decay
        :param q_values: 2 dimensional array of size (num observations, num actions).
                        Values represents the expected value of being in a state and performing the action.
        '''
        self.epsilon=epsilon
        self.epsilon_start=epsilon
        self.decay_rate=decay_rate
        self.epsilon_min=epsilon_min
        self.q_values=q_values
        self.num_actions=num_actions

    def update_q_values(self, q_values):
        '''
        Update the q_values array
        :param q_values:
        :return: None
        '''

        self.q_values=q_values


    def get_action_by_policy(self, observation_index):
        '''
        With probability 1-e Returns the highest value action according to a q-values array,
        and with probability e returns a random action
        :param observation_index (int)
        :return: action_index(int)
        '''

        if np.random.rand()<self.epsilon: #with e probability pick a random action
            action_index=np.random.randint(self.num_actions)

        else:
            index_max_action=np.argwhere(self.q_values[observation_index, :] == np.max(self.q_values[observation_index, :]))

            if len(index_max_action)>1: #In case there is more than 1 action with the same max value-return a random action among them.
                action_index=int(np.random.choice(index_max_action.squeeze()))
            else:
                action_index=int(index_max_action.squeeze())


        return action_index

    def decay_epsilon(self):
        '''
        Multiply the epsilon by the decay rate. limited by the minimum value for the epsilon
        :return:None
        '''

        new_epsilon=self.epsilon*self.decay_rate
        if new_epsilon>self.epsilon_min:
            self.epsilon=new_epsilon
        else:
            self.epsilon=self.epsilon_min