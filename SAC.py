import gym
import os
import sys
import yaml
import argparse
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import constopt
from constopt.constraints import LinfBall
from constopt.stochastic import PGD, PGDMadry, FrankWolfe, MomentumFrankWolfe
import torch
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import torch.nn.utils as utils
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import rankdata
from collections import deque
sys.path.insert(0,r'constopt-pytorch/')

class BaseAgent(ABC):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=16,
                 memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, num_eval_steps=1000, max_episode_steps=27000,
                 log_interval=10, eval_interval=1000, cuda=True, seed=0):
        super().__init__()
        self.env = env
        self.test_env = test_env

        # Set seed.
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(2**31-1-seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = 'cpu'#torch.device(
            #"cuda" if cuda and torch.cuda.is_available() else "cpu")

        # LazyMemory efficiently stores FrameStacked states.
        if use_per:
            beta_steps = (num_steps - start_steps) / update_interval
            self.memory = LazyPrioritizedMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.observation_space.shape,
                device=self.device, gamma=gamma, multi_step=multi_step,
                beta_steps=beta_steps)
        else:
            self.memory = LazyMultiStepMemory(
                capacity=memory_size,
                state_shape=self.env.observation_space.shape,
                device=self.device, gamma=gamma, multi_step=multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = 16
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps =200
        self.log_interval = log_interval
        self.eval_interval = eval_interval

    def run(self,rnd):
        self.train_episode(rnd)
        #    if self.steps > self.num_steps:
         #       break

    def is_update(self):
        return self.steps % self.update_interval == 0            and self.steps >= self.start_steps

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def update_target(self):
        pass

    @abstractmethod
    def calc_current_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_target_q(self, states, actions, rewards, next_states, dones):
        pass

    @abstractmethod
    def calc_critic_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_policy_loss(self, batch, weights):
        pass

    @abstractmethod
    def calc_entropy_loss(self, entropies, weights):
        pass

    def train_episode(self,rnd):
        if rnd>0:
            #save_dir=os.path.join('logs_sac_newatari', str(index), f'{name}-seed{args.seed}-{time}')
            self.policy.load_state_dict(torch.load(str(index)+'policysac_newatari.pth'))
            self.online_critic.load_state_dict(torch.load(str(index)+'online_criticsac_newatari.pth'))
            self.target_critic.load_state_dict(torch.load(str(index)+ 'target_criticsac_newatari.pth'))
        for inner_rnd in range(20):
            self.episodes += 1
            episode_return = 0.
            episode_steps = 0
            done = False
            state = self.env.reset()

            while (not done) and episode_steps <= 200-1:

                if self.start_steps > self.steps:
                    action = self.env.action_space.sample()
                    next_state, reward, done, info = self.env.step(action)
                else:
                    action = self.explore(state)
                    next_state, reward, done, info = self.env.step(action.detach().cpu().numpy()[0][0])
                # Clip reward to [-1.0, 1.0].
                clipped_reward = max(min(reward, 1.0), -1.0)

                # To calculate efficiently, set priority=max_priority here.
                self.memory.append(state, action, clipped_reward, next_state, done)

                self.steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

     #           if self.is_update():
      #              self.learn()

       #         if self.steps % self.target_update_interval == 0:
        #            self.update_target()

            if (inner_rnd+1)%5 == 0:
                #self.evaluate()
                print('save')
                self.learn(inner_rnd)
                self.update_target()
                self.save_models(os.path.join(self.model_dir, 'final'))
            rewardsRec[index].append(episode_return)
            np.save('sac_newatari_rewardsRec.npy',rewardsRec)
            np.save('sac_newatari_succeessRec.npy',succeessRec)
            # We log running mean of training rewards.
            self.train_return.append(episode_return)

            if self.episodes % self.log_interval == 0:
                self.writer.add_scalar(
                    'reward/train', self.train_return.get(), self.steps)
    
            print('Env: ',index,'rnd: ',rnd,'Episode: ',self.episodes,'Return: ',episode_return)

    def learn(self,inner_rnd):
        assert hasattr(self, 'q1_optim') and hasattr(self, 'q2_optim') and hasattr(self, 'policy_optim') and hasattr(self, 'alpha_optim')

        self.learning_steps += 1

        if self.use_per:
            batch, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # Set priority weights to 1 when we don't use PER.
            weights = 1.

        q1_loss, q2_loss, errors, mean_q1, mean_q2 =             self.calc_critic_loss(batch, weights)
        policy_loss, entropies = self.calc_policy_loss(batch, weights)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        update_params(self.q1_optim, q1_loss,inner_rnd,True)
        update_params(self.q2_optim, q2_loss,inner_rnd)
        update_params(self.policy_optim, policy_loss,inner_rnd)
        update_params(self.alpha_optim, entropy_loss,inner_rnd)

        self.alpha = self.log_alpha.exp()

        if self.use_per:
            self.memory.update_priority(errors)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <=200-1:
                action = self.exploit(state)
                next_state, reward, done, _ = self.test_env.step(action.view(4).detach().numpy())
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def __del__(self):
        self.env.close()
        self.test_env.close()
        self.writer.close()

def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.image_conv =nn.Sequential(nn.Conv2d(3, 1, (5, 5), (5,5)),nn.Conv2d(1, 1, (3, 3), (3,3)))
        
        
        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(140,16),
                nn.Tanh(),
                nn.Linear(16, envs[0].action_space.n))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(140,16),
                nn.Tanh(),
                nn.Linear(16, envs[0].action_space.n))
            self.v_head = nn.Sequential(
                nn.Linear(140,16),
                nn.Tanh(),
                nn.Linear(16, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if len(states.shape)==3:
          states=states.view([1]+list(states.shape))
        else:
          states=states.permute(0,3,1,2)
        new_x = states
        new_x = self.image_conv(new_x)
        states = new_x.reshape(states.shape[0], -1)   
        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_channels, envs[0].action_space.n, shared, dueling_net)
        self.Q2 = QNetwork(num_channels, envs[0].action_space.n, shared, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2

class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False):
        super().__init__()
        self.image_conv =nn.Sequential(
            nn.Conv2d(3, 1, (5, 5), (5,5)),
            nn.Conv2d(1, 1, (3, 3), (3,3))
        )
                
        self.head = nn.Sequential(
            nn.Linear(140,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),            
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),               
            nn.Linear(16, envs[0].action_space.n))

        self.shared = shared

    def act(self, states):
        if len(states.shape)==3:
          states=states.view([1]+list(states.shape))
        else:
          states=states.permute(0,3,1,2)
        new_x = states
        new_x = self.image_conv(new_x)
        states = new_x.reshape(states.shape[0], -1)          
        action_logits = self.head(states)
        return action_logits

    def sample(self, states):
        if len(states.shape)==3:
          states=states.view([1]+list(states.shape))
        else:
          states=states.permute(0,3,1,2)
        new_x = states
        new_x = self.image_conv(new_x)
        states = new_x.reshape(states.shape[0], -1)          
        action_probs = F.softmax(self.head(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

class SharedSacdAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=100000, batch_size=16,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=1000,
                 use_per=False, dueling_net=False, num_eval_steps=1000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0):
        super().__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size, gamma,
            multi_step, target_entropy_ratio, start_steps, update_interval,
            target_update_interval, use_per, num_eval_steps, max_episode_steps,
            log_interval, eval_interval, cuda, seed)

        # Define networks.
        
        self.policy = CateoricalPolicy(
            12, 4,
            shared=True).to(self.device)
        self.online_critic = TwinnedQNetwork(
            12, 4,
            dueling_net=dueling_net, shared=True).to(device=self.device)
        self.target_critic = TwinnedQNetwork(
            12, 4,
            dueling_net=dueling_net, shared=True).to(device=self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(
            list(self.online_critic.Q1.parameters()), lr=lr)
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy =             -np.log(1.0 / 4) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def explore(self, state):
        # Act with randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action

    def exploit(self, state):
        # Act without randomness.
        state = torch.ByteTensor(
            state[None, ...]).to(self.device).float() / 255.
        with torch.no_grad():
            action = self.policy.act(state)
        return action

    def update_target(self):
        self.target_critic.load_state_dict(self.online_critic.state_dict())

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1 = self.online_critic.Q1(states).gather(1, actions.long())
        curr_q2 = self.online_critic.Q2(
            states.detach()).gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        
        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)
        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies)
            * weights)
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        save_dir=os.path.join('logs_sac_newatari', str(index), f'{name}-seed{args.seed}-{time}')
        self.policy.save( str(index)+'policysac_newatari.pth')
        self.online_critic.save(str(index)+'online_criticsac_newatari.pth')
        self.target_critic.save(str(index)+'target_criticsac_newatari.pth')

class MultiStepBuff:

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.reset()

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self, gamma=0.99):
        assert len(self.rewards) > 0
        state = self.states.popleft()
        action = self.actions.popleft()
        reward = self._nstep_return(gamma)
        return state, action, reward

    def _nstep_return(self, gamma):
        r = np.sum([r * (gamma ** i) for i, r in enumerate(self.rewards)])
        self.rewards.popleft()
        return r

    def reset(self):
        # Buffer to store n-step transitions.
        self.states = deque(maxlen=self.maxlen)
        self.actions = deque(maxlen=self.maxlen)
        self.rewards = deque(maxlen=self.maxlen)

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return len(self.rewards) == self.maxlen

    def __len__(self):
        return len(self.rewards)

class LazyMemory(dict):

    def __init__(self, capacity, state_shape, device):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self.reset()

    def reset(self):
        self['state'] = []
        self['next_state'] = []

        self['action'] = np.empty((self.capacity, 4), dtype=np.int64)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)

        self._n = 0
        self._p = 0

    def append(self, state, action, reward, next_state, done,
               episode_done=None):
        self._append(state, action, reward, next_state, done)

    def _append(self, state, action, reward, next_state, done):
        self['state'].append(state)
        self['next_state'].append(next_state)
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        self.truncate()

    def truncate(self):
        while len(self['state']) > self.capacity:
            del self['state'][0]
            del self['next_state'][0]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        return self._sample(indices, batch_size)

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)

        for i, index1 in enumerate(indices):
            _index = np.mod(index1+bias, self.capacity)
            states[i, ...] = self['state'][_index]
            next_states[i, ...] = self['next_state'][_index]

        states = torch.ByteTensor(states).to(self.device).float() / 255.
        next_states = torch.ByteTensor(
            next_states).to(self.device).float() / 255.
        actions = torch.LongTensor(self['action'][indices]).to(self.device)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self._n

class LazyMultiStepMemory(LazyMemory):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3):
        super(LazyMultiStepMemory, self).__init__(
            capacity, state_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if self.buff.is_full():
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done:
                while not self.buff.is_empty():
                    state, action, reward = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done)
        else:
            self._append(state, action, reward, next_state, done)

def update_params(optim, loss,inner_rnd, retain_graph=False):
  optim.zero_grad()
  w=[]
  for key in agents[0].policy.state_dict().keys():
      w.append(torch.cat([agents[j].policy.state_dict()[key].unsqueeze(-1) for j in range(len(envs))]))            
  pre=loss
  with torch.autograd.set_detect_anomaly(True):
      pre.backward(retain_graph=retain_graph)
  # nn.utils.clip_grad_norm_(model.parameters(), 30)
  optim.step()
  
def disable_gradients(network):
  # Disable calculations of gradients.
  for param in network.parameters():
      param.requires_grad = False

class RunningMeanStats:
  def __init__(self, n=10):
      self.n = n
      self.stats = deque(maxlen=n)

  def append(self, x):
      self.stats.append(x)

  def get(self):
      return np.mean(self.stats)

def loss(rloss,w,B,mu=0.2,lamb=[0.01,0.01,0.01]):
    return torch.tensor([1+mu*(np.linalg.norm(B[t],ord=1)-np.linalg.norm(B[t][t],ord=1)) for t in range(len(envs))]).dot(rloss)+lamb[0]*sum([sum([sum([torch.norm(w[i][t]-sum([B.T[t][j]*w[i][j] for j in range(len(envs))]),p=2)**2]) for i in range(2)]) for t in range(len(envs))])

class parser:
    def __init__(self):
        
        self.config=os.path.join('metaworld-master/', 'sacd.yaml')
        self.shared=True
        self.env_id='MsPacmanNoFrameskip-v4'
        self.cuda=True
        self.seed=0
        
args=parser()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

envs=[]
for env_name in ['YarsRevenge', 'Jamesbond', 'FishingDerby', 'Venture',
       'DoubleDunk', 'Kangaroo', 'IceHockey', 'ChopperCommand', 'Krull',
       'Robotank', 'BankHeist', 'RoadRunner', 'Hero', 'Boxing',
       'Seaquest', 'PrivateEye', 'StarGunner', 'Riverraid',
       'Zaxxon', 'Tennis', 'BattleZone',
       'MontezumaRevenge', 'Frostbite', 'Gravitar',
       'Defender', 'Pitfall', 'Solaris', 'Berzerk',
       'Centipede'][:10]:
    env=gym.make(env_name)  
    envs.append(env)
    
rloss=[0.0 for i in range(len(envs))]
rewardsRec=[[] for i in range(len(envs))]
rewardsRec_nor=[[0] for i in range(len(envs))]
succeessRec=[[] for i in range(len(envs))]

agents=[]
for index in range(len(envs)):
    # Create environments.
    env = envs[index]
    test_env = envs[index]

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    if args.shared:
        name = 'shared-' + name
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir =os.path.join('logs_sac_newatari', str(index), f'{name}-seed{args.seed}-{time}')

    # Create the agent.
    Agent = SacdAgent if not args.shared else SharedSacdAgent
    agent = Agent(
        env=env, test_env=test_env, log_dir=log_dir, cuda=args.cuda,
        seed=args.seed, **config)
    agents.append(agent)

for i_episode in range(10000):
    for index in range(len(envs)):
        rnd=i_episode
        agents[index].run(rnd)
