import torch
import torch.nn as nn
import collections
import numpy as np
import torch.nn.functional as F


def build_block(input_dim, output_dim, num):
    block = nn.Sequential(collections.OrderedDict([
        ('linear' + str(num), nn.Linear(input_dim, output_dim)),
        ('bn' + str(num), nn.BatchNorm1d(output_dim)),
        ('relu' + str(num), nn.ReLU())
    ]))

    return block



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, linear_sizes):
        '''
        Constructor of the actor network (policy network) in the DDPG model. The network has 2 Linear hidden layers, each followed by batch normalization and a reLU activation.
        There is a final output layer which is linear layer followed by TanH activation that outputs the actions.
        :param state_dim:
        :param action_dim:
        :param linear_sizes: list of size 2 with the number of units for each hidden layer.
        '''

        super(Actor, self).__init__()

        self.linear1 = nn.Linear(state_dim, linear_sizes[0])
        #Initialization according to ddpg paper
        nn.init.uniform(self.linear1.weight, -1/np.sqrt(state_dim), 1/np.sqrt(state_dim))
        nn.init.uniform(self.linear1.bias, -1 / np.sqrt(state_dim), 1 / np.sqrt(state_dim))
        self.bn1 = nn.BatchNorm1d(linear_sizes[0])
        self.relu1=nn.ReLU()

        self.linear2 = nn.Linear(linear_sizes[0], linear_sizes[1])
        self.bn2 = nn.BatchNorm1d(linear_sizes[1])
        # Initialization according to ddpg paper
        nn.init.uniform(self.linear2.weight, -1 / np.sqrt(linear_sizes[0]), 1 / np.sqrt(linear_sizes[0]))
        nn.init.uniform(self.linear2.bias, -1 / np.sqrt(linear_sizes[0]), 1 / np.sqrt(linear_sizes[0]))
        self.relu2 = nn.ReLU()


        self.last_linear_layer=nn.Linear(linear_sizes[1], action_dim)
        # Initialization according to ddpg paper
        nn.init.uniform(self.last_linear_layer.weight, -0.003, 0.003)
        nn.init.uniform(self.last_linear_layer.bias, -0.003, 0.003)
        self.tanh=nn.Tanh()

    def forward(self,state_inputs):
        first_layer_outputs= self.relu1(self.bn1(self.linear1(state_inputs)))


        second_layer_outputs = self.relu2(self.bn2(self.linear2(first_layer_outputs)))


        last_linear_output=self.last_linear_layer(second_layer_outputs)
        output = self.tanh(last_linear_output)
        return output




class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, linear_sizes):
        '''
                Constructor of the Critic network (value network) in the DDPG model.
                The network has 3 Linear hidden layers, the first layer takes the state inputs and  is followed by batch normalization and a reLU activation.
                There are two parallel second hidden layers. The first takes as input the first layer outputs and the second takes as inputs the actions inputs.
                The outptut of the two parallel models are added and passed though ReLU activation .
                This final layer takes the combined outputs from the second layer and outputs one value which is are the q values.

                :param state_dim: state dimensions
                :param action_dim: action dimensions
                :param linear_sizes: list of size 2 with the number of units for the hidden layer. (The second parallel linear layers have the same output size)
        '''

        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim, linear_sizes[0])
        # Initalization according to ddpg paper
        nn.init.uniform(self.linear1.weight, -1 / np.sqrt(state_dim), 1 / np.sqrt(state_dim))
        nn.init.uniform(self.linear1.bias, -1 / np.sqrt(state_dim), 1 / np.sqrt(state_dim))
        self.bn1=nn.BatchNorm1d(linear_sizes[0])
        self.relu1 = nn.ReLU()


        self.linear_actions = nn.Linear(action_dim,linear_sizes[1])
        # Initalization according to ddpg paper
        nn.init.uniform(self.linear1.weight, -1 / np.sqrt(action_dim), 1 / np.sqrt(action_dim))
        nn.init.uniform(self.linear1.bias, -1 / np.sqrt(action_dim), 1 / np.sqrt(action_dim))
        #self.bn_actions = nn.BatchNorm1d(linear_sizes[0])
        #self.relu_actions=nn.ReLU()

        self.linear2 = nn.Linear(linear_sizes[0], linear_sizes[1])
        # Initalization according to ddpg paper
        nn.init.uniform(self.linear2.weight, -1 / np.sqrt(linear_sizes[0]), 1 / np.sqrt(linear_sizes[0]))
        nn.init.uniform(self.linear2.bias, -1 / np.sqrt(linear_sizes[0]), 1 / np.sqrt(linear_sizes[0]))
       # self.bn2 = nn.BatchNorm1d(linear_sizes[1])
        # self.relu2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(linear_sizes[1])
        self.relu2 = nn.ReLU()

        #self.first_layer = build_block(state_dim, linear_sizes[0], 0)

        #self.second_layer=build_block(linear_sizes[0]+action_dim, linear_sizes[1], 1)

        self.last_linear_layer=nn.Linear(linear_sizes[1], 1)
        # Initalization according to ddpg paper
        nn.init.uniform(self.last_linear_layer.weight, -0.003, 0.003)
        nn.init.uniform(self.last_linear_layer.bias, -0.003, 0.003)
        #nn.init.uniform(self.last_linear_layer.weight, -0.004, 0.004)

    def forward(self,state_inputs, action_inputs):

        first_layer_outputs= self.relu1(self.bn1(self.linear1(state_inputs)))

        #first_layer_outputs=self.bn1(first_layer_outputs)

        actions_layer_outputs = self.linear_actions(action_inputs)

        second_layer_outputs = self.linear2(first_layer_outputs)

        #actions_layer_outputs=self.bn_actions(actions_layer_outputs)

        #second_layer_inputs = torch.cat((first_layer_outputs, action_inputs), dim=1)

        added_outputs=torch.add(second_layer_outputs, actions_layer_outputs)


        second_layer_outputs=self.relu2(added_outputs)

        #second_layer_outputs=self.bn2(second_layer_outputs)

        #first_layer_output_and_action=torch.cat(first_layer_outputs, dim=1)
        #second_layer_outputs = self.linear2(first_layer_outputs)

        #third_layer_output= self.relu3(self.bn3(self.linear3(second_layer_outputs)))
        #According to ddpg paper actions only were added in second layer
        #second_outputs=self.second_layer(torch.cat((first_layer_outputs, action_inputs), dim=1))

       # combined_states_actions=self.relu2(self.bn2(torch.add(second_layer_outputs,action_output)))
        output = self.last_linear_layer(second_layer_outputs)

        return output