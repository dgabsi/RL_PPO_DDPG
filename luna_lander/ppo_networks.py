import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, linear_sizes):
        '''
                Constructor of the actor network (policy network) in the PPO model. The network has 2 Linear hidden layers, each followed and a reLU activation.
                There is a final output layer is logits linear layer followed by a softmax layer to calculates policy distribution probabilities.
                The output is the distribution.
                :param state_dim:
                :param action_dim:
                :param linear_sizes: list of size 2 with the number of units for each hidden layer.
        '''
        super(Actor, self).__init__()


        self.linear1 = nn.Linear(state_dim, linear_sizes[0])
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(linear_sizes[0], linear_sizes[1])
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(linear_sizes[1], action_dim)
        self.prob = nn.Softmax(dim=-1)

    def forward(self, state_inputs):
        batch_size = state_inputs.size()[0]
        first_layer_outputs = self.relu1(self.linear1(state_inputs))

        second_layer_outputs = self.relu2(self.linear2(first_layer_outputs))
        logits = self.linear3(second_layer_outputs)

        prob = self.prob(logits)

        distr = torch.distributions.Categorical(prob)

        return distr


class Critic(nn.Module):
    def __init__(self, state_dim, linear_sizes):
        '''

        Constructor of the Critic network (value network) in the PPO model.
        The network has 2 Linear hidden layers, each  followed by  a reLU activation.
        This final layer takes the combined outputs from the second layer and outputs one value which is are the q values.

        :param state_dim: state dimensions
         :param linear_sizes: list of size 2 with the number of units for the hidden layers.
        '''


        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim, linear_sizes[0])
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(linear_sizes[0], linear_sizes[1])
        self.relu2 = nn.ReLU()

        self.last_linear_layer = nn.Linear(linear_sizes[1], 1)

    def forward(self, state_inputs):
        first_layer_outputs = self.relu1(self.linear1(state_inputs))

        second_layer_outputs = self.relu2(self.linear2(first_layer_outputs))

        output = self.last_linear_layer(second_layer_outputs)
        return output