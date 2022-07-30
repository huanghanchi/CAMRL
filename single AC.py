import metaworld
import random

ml10 = metaworld.MT10() # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name == name])
  env.set_task(task)
  training_envs.append(env)

for env in training_envs:
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action

envs=training_envs

import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import torch_ac

torch.manual_seed(1337)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Single(nn.Module):

    def __init__(self, tasks = len(envs) ):

        super(Single, self).__init__()

        self.actor = torch.nn.ModuleList ( [ nn.Sequential(
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64,4)
        ) for i in range(tasks) ] )
        self.value_head =  torch.nn.ModuleList ( [nn.Sequential(
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64,1)
        ) for i in range(tasks) ] )

        self.saved_actions = [[] for i in range(tasks)] 
        self.rewards = [[] for i in range(tasks)] 
        self.tasks = tasks 

    def forward(self, x):
        tmp=[]
        
        for i in range(self.tasks) :
            tmp.append( F.softmax(self.actor[i](x)-self.actor[i](x).max()))
        state_values = self.value_head[index](x)
        return tmp , state_values

def select_action(state, tasks,index):
    state=torch.tensor(list(state)).float()
    probs, state_value = model(Variable(state))

    # Obtain the most probable action for each one of the policies
    actions = []
    for i in range(tasks):
        model.saved_actions[i].append(SavedAction(probs[i].log().dot(probs[i]), state_value))

    return probs, state_value

def finish_episode( tasks , alpha , beta , gamma ):

    ### Calculate loss function according to Equation 1
    R = 0
    saved_actions = model.saved_actions[index]
    policy_losses = []
    value_losses = []

    ## Obtain the discounted rewards backwards
    rewards = []
    for r in model.rewards[index][::-1]:
        R = r + gamma * R 
        rewards.insert(0, R)

    ## Standardize the rewards to be unit normal (to control the gradient estimator variance)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))


    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    
    loss.backward()
    optimizer.step()

    #Clean memory
    for i in range(tasks):
        del model.rewards[i][:]
        del model.saved_actions[i][:]

model = Single( )
optimizer = optim.Adam(model.parameters(), lr=3e-2)

file_name="Single"
batch_size=128
alpha = 0.5
beta = 0.5
gamma=0.999
is_plot=False
num_episodes=500
max_num_steps_per_episode=10000
learning_rate=0.001 

#Run each one of the policies for the different environments
#update the policies

tasks = len(envs)
rewardsRec=[[] for _ in range(len(envs))]
for rnd in range(10000):
    for index,env in enumerate(envs): 
            total_reward = 0
            state = env.reset()
            for t in range(200):  # Don't infinite loop while learning
                probs, state_value = select_action(state, tasks,index )
                state, reward, done, _ = env.step(probs[index].detach().numpy())
                if is_plot:
                    env.render()
                model.rewards[index].append(reward)
                total_reward += reward
                if done:
                    break
            print(rnd,index,total_reward)
            rewardsRec[index].append(total_reward)
            finish_episode( tasks , alpha , beta, gamma )
    np.save('mt10_single_rewardsRec.npy',rewardsRec)
    torch.save(model.state_dict(), 'mt10_single_params.pkl')