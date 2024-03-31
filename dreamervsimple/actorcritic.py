#actor is largely the same:
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from collections import OrderedDict

class BaseMLP(nn.Module):
    def __init__(self, in_state_shape, out_shape, inter_units, layers, act="relu"):
        super().__init__()
        seq_list = OrderedDict([
            ('fc1', nn.Linear(in_state_shape, inter_units)),
            ('relu1', nn.ReLU()) if act == "relu" else ('elu1', nn.ELU())
        ])
        for _ in range(layers - 1):
            seq_list.update({
                f'fc{len(seq_list)}': nn.Linear(inter_units, inter_units),
                f'norm{len(seq_list)}': nn.LayerNorm(inter_units),
                f'relu{len(seq_list)}': nn.ReLU() if act == "relu" else ('elu{len(seq_list)}', nn.ELU())
            })
        seq_list.update({"outlayer": nn.Linear(inter_units, out_shape)})
        self.seq = nn.Sequential(seq_list)

    def forward(self, x):
        return self.seq(x) 
    
#Actor MLP - output actions that maximize the prediction of long-term future rewards made by the critic. Max the same lambda as the critic 
class ActorCritic():
    def __init__(self, batch_size, in_state_shape, action_shape, inter_units, layers,
                 slow_target=False, alr=1e-4, clr=1e-4):
        self.actor = BaseMLP(in_state_shape, action_shape, inter_units, layers)
        self.critic = BaseMLP(in_state_shape, 1, inter_units, layers)
        if slow_target:
            self.target_critic = BaseMLP(in_state_shape, 1, inter_units, layers) # critic target netwiorks are used for smoothing the updation of the critic networks
        else: self.target_critic = self.critic

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=clr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=alr)
        self.rewnorm = torch.ones((batch_size, 1))

    def rescale_reward(self, r, momentum=0.99, scale=1.0, eps=1e-8):
        #normalize and rescale the reward r across the batch 
        self.rewnorm = momentum * self.rewnorm + (1 - momentum) * r.abs().mean()
        return (r / (self.rewnorm + eps)) * scale
    #say sequence such that 0 is a sequence of states, 1 is the sequence of actions, and 3 is the discount weighted rewards
    def actor_loss(self, seq, target_values):
        #ignoring the dynamics thing for now. just reinforce and entropy regularizer
        # dist = Categorical(logits=actions) #TODO: based on rest of your implementation, change this from logits to probs if needed
        policy = Categorical(self.actor(seq[0][:-2].detach()))
        entropy = policy.entropy()
        loss = -policy.log_prob(seq[1][1:-1]) * (target_values[1:-1].detach() - self.target_critic(seq[0][:-2]).detach())
        loss = loss - 0.01 * 1e-3*entropy
        return (loss * seq[:-2]).detach().mean()
    
    def critic_loss(self, seq, target_values):
        dist = Normal(self.critic(seq["state_rep"][:-1]), 1.0)
        weight = #TODO
        loss = -dist.log_prob(target_values[:-1]) * weight[:-1].detach()
        return loss.mean()

    def target(self, seq):
        reward = seq["reward"].astype(torch.float32)
        disc = seq["discount"].astype(torch.float32)
        value = Normal(self.critic(seq["feat"]), 1.0)
        target = self.lambda_return(reward[:-1], value[:-1], disc[:-1], value[-1], 0.8, axis=0)
        return target

    def lambda_return(self, rewards, value, disc, bootstrap, lambda_, axis=0):
        # Setting lambda=1 gives a discounted Monte Carlo return.
        # Setting lambda=0 gives a fixed 1-step return.

    def plan(self, 
             world_model, # this is equivalent to querying the environment if you dont have a model, so pass in the environment here if no model 
             discount=0.999, 
             a_lambda=0.8, 
             c_lambda=0.8, 
             ep=1000, 
             max_steps=10000,
             target_update=0.8):
        
        for _ in range(ep): # run a2c for these many episodes
            #take some actions on the current policy 
            seq = #TODO
            #calculate the returns of the actions taken
            returns = self.rescale_reward(None) #TODO
            #target critic check value:
            target_values = self.target_critic(seq)
            critic_loss, actor_loss = self.critic_loss(), self.actor_loss()

            #giveb the losses perform backprop
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            #update target critic as moving average 
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - target_update) + param.data * target_update)