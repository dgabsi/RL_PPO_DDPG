import torch
import torch.nn as nn



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, linear_sizes,std):

        super(Actor, self).__init__()

        self.std=std
        self.linear1 = nn.Linear(state_dim, linear_sizes[0])
        self.relu1=nn.ReLU()

        self.linear2 = nn.Linear(linear_sizes[0], action_dim)

        self.log_std=nn.Parameter(torch.full(1,action_dim), std)

    def forward(self,state_inputs):

        batch_size=state_inputs.size()[0]
        first_layer_outputs= self.relu1(self.linear1(state_inputs))

        mu_output= self.linear2(first_layer_outputs)

        std_output=self.log_std.exp().unsqueeze(batch_size)
        return mu_output,std_output



class Critic(nn.Module):
    def __init__(self, state_dim, linear_sizes):

        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim, linear_sizes[0])
        self.relu1=nn.ReLU()

        self.linear2 = nn.Linear(linear_sizes[0], linear_sizes[1])
        self.relu2 = nn.ReLU()


        self.last_linear_layer=nn.Linear(linear_sizes[1], 1)

    def forward(self,state_inputs):
        first_layer_outputs= self.relu1(self.linear1(state_inputs))

        second_layer_outputs = self.relu2(self.linear2(first_layer_outputs))

        output=self.last_linear_layer(second_layer_outputs)
        return output