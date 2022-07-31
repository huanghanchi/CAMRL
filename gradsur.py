import metaworld
import random
import numpy as np
from operator import mul
from functools import reduce
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import torch_ac
import pdb
import gym
from itertools import count
from collections import namedtuple
import argparse, math, os
device = torch.device("cuda")
torch.manual_seed(1337)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def pc_grad_update(gradient_list):
  '''
  PyTorch implementation of PCGrad.
  Gradient Surgery for Multi-Task Learning: https://arxiv.org/pdf/2001.06782.pdf
  Arguments:
    gradient_list (Iterable[Tensor] or Tensor): an iterable of Tensorsthat will 
    have gradients with respect to parameters for each task.
  Returns:
    List of gradients with PCGrad applied.
  '''

  assert type(gradient_list) is list
  assert len(gradient_list) ! =  0
  num_tasks = len(gradient_list)
  num_params = len(gradient_list[0])
  np.random.shuffle(gradient_list)

  def flatten_and_store_dims(grad_task):
    output = []
    grad_dim = []
    for param_grad in grad_task: # TODO(speedup): convert to map since they are faster
      grad_dim.append(tuple(param_grad.shape))
      output.append(torch.flatten(param_grad))
      
    return torch.cat(output), grad_dim

  def restore_dims(grad_task, chunk_dims):
    ## chunk_dims is a list of tensor shapes
    chunk_sizes = [reduce(mul, dims, 1) for dims in chunk_dims]
    
    grad_chunk = torch.split(grad_task, split_size_or_sections = chunk_sizes)
    resized_chunks = []
    for index, grad in enumerate(grad_chunk): # TODO(speedup): convert to map since they are faster
      grad = torch.reshape(grad, chunk_dims[index])
      resized_chunks.append(grad)

    return resized_chunks

  def project_gradients(grad_task):
    """
    Subtracts projected gradient components for each grad in gradient_list
    if it conflicts with input gradient.
    Argument:
      grad_task (Tensor): A tensor for a gradient
    Returns:
      Component subtracted gradient
    """
    grad_task, grad_dim = flatten_and_store_dims(grad_task)

    for k in range(num_tasks): # TODO(speedup): convert to map since they are faster
      conflict_gradient_candidate = gradient_list[k]
      # no need to store dims of candidate since we are not changing it in the array
      conflict_gradient_candidate, _ = flatten_and_store_dims(conflict_gradient_candidate)
      
      inner_product = torch.dot(torch.flatten(grad_task), torch.flatten(conflict_gradient_candidate))
      # TODO(speedup): put conflict check condition here so that we aren't calculating norms for non-conflicting gradients
      proj_direction = inner_product / torch.norm(conflict_gradient_candidate)**2
      
      ## sanity check to see if there's any conflicting gradients
      # if proj_direction < 0.:
      #   print('conflict')
      # TODO(speedup): This is a cumulative subtraction, move to threaded in-memory map-reduce
      grad_task = grad_task - min(proj_direction, 0.) * conflict_gradient_candidate
    
    # get back grad_task
    grad_task = restore_dims(grad_task, grad_dim)
    return grad_task

  flattened_grad_task = list(map(project_gradients, gradient_list))

  yield flattened_grad_task

class ACModel(nn.Module):

    def __init__(self, tasks = 1):

        super(ACModel, self).__init__()

        self.actor = torch.nn.ModuleList ( [ nn.Sequential(
            nn.Linear(12, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, 16), 
            nn.ReLU(),             
            nn.Linear(16, 4)
        ) for i in range(1) ] )
        self.value_head =  torch.nn.ModuleList ( [nn.Sequential(
            nn.Linear(12, 128), 
            nn.ReLU(), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, 16), 
            nn.ReLU(),             
            nn.Linear(16, 1)
        ) for i in range(1) ] )

        self.saved_actions = [[] for i in range(1)] 
        self.rewards = [[] for i in range(1)] 
        self.tasks = 1 

    def forward(self, x):
        tmp = []
        for i in range(1) :
            tmp.append( F.softmax(self.actor[i](x)-self.actor[i](x).max()))
        state_values = self.value_head[0](x)
        return tmp, state_values
        
class REINFORCE:
    def __init__(self):
        self.model = ACModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)
        self.model.train()
        self.saved_actions = [[]]
        self.rewards = [[]]

    def select_action(self, state):
        state = torch.tensor(list(state)).float()
        probs, state_value = self.model(Variable(state))

        # Obtain the most probable action for each one of the policies
        actions = []
        self.saved_actions[0].append(SavedAction(probs[0].log().dot(probs[0]), state_value))

        return probs, state_value

    def update_parameters(self, rewards, log_probs, entropies, gamma, multitask = False):# 更新参数
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]                                # 倒序计算累计期望
            # loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i])).cuda()).sum() - (0.0001*entropies[i].cuda()).sum()
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum() - (0.0001*entropies[i]).sum()
        loss = loss / len(rewards)
        if multitask:
            loss += lossw2(currindex, env_id, loss, w, B)
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)             # 梯度裁剪，梯度的最大L2范数 = 40
        self.optimizer.step()

def finish_episode( tasks, alpha, beta, gamma):
    optimizer  = [ optim.Adam(agents[i].model.parameters(), lr = 3e-2) for i in range(len(envs))]
    losses = []
    grad_list = []
    for env_id in range(len(envs)):
        ### Calculate loss function according to Equation 1
        R = 0
        saved_actions = agents[env_id].saved_actions[0]
        policy_losses = []
        value_losses = []
    
    ## Obtain the discounted rewards backwards
        rewards = []
        for r in agents[env_id].rewards[0][::-1]:
            R = r + gamma * R 
            rewards.insert(0, R)

        ## Standardize the rewards to be unit normal (to control the gradient estimator variance)
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for (log_prob, value), r in zip(saved_actions, rewards):
            reward = r - value.data[0]
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
        optimizer[env_id].zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        optimizer[env_id].step()
        
        for i in range(1):
            del agents[env_id].rewards[i][:]
            del agents[env_id].saved_actions[0][:]
        if rnd! = 0:
            tmp = []
            for p in agents[env_id].model.parameters():
              # Simulate 5 different tasks
              tmp.append(p.grad)
            grad_list.append(tmp)

    if rnd = 0:
        pc_grad_update(grad_list)
    
ml10 = metaworld.MT10() # Construct the benchmark, sampling tasks
envs = []
for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name   ==    name])
  env.set_task(task)
  envs.append(env)

for env in envs:
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action      

rnd = 0
batch_size = 128
alpha = 0.5
beta = 0.5
gamma = 0.999
is_plot = False
num_episodes = 500
max_num_steps_per_episode = 10000
learning_rate = 0.001 
rewardsRec = [[] for _ in range(len(envs))]
tasks = len(envs)
agents = [REINFORCE() for i in range(len(envs))]
optimizer  = [ optim.Adam(agents[i].model.parameters(), lr = 3e-2) for i in range(len(envs))]

for rnd in range(10000):
    for env_id in range(len(envs)):
        rewardRec = []
        for i_episode in range(1):
            visualise = True
            rewardcnt = 0
            observations =  envs[env_id].reset() 
            for t in range(200):
                probs, state_value  = agents[env_id].select_action(observations)
                observations, reward, done, _ = envs[env_id].step(probs[0].detach().numpy())
                agents[env_id].rewards[0].append(reward)
                rewardcnt + =  reward
                if done:
                    break
            rewardRec.append(rewardcnt)
            rewardsRec[env_id].append(rewardcnt)
            np.save('rewardsRec2_gradsur_meta.npy', rewardsRec)
            print(rnd, env_id, rewardcnt)
            
    finish_episode( tasks, alpha, beta, gamma )
    for env_id in range(len(envs)):
        torch.save(agents[env_id].model.state_dict(), str(env_id)+'meta10.pkl')   # 只保存网络中的参数 (速度快, 占内存少)
