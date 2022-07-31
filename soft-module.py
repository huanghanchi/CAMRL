import os
import os.path as osp
import sys
import metaworld
from metaworld.envs.mujoco.sawyer_xyz import *
import random
import time
import numpy as np
import math
import copy
from collections import deque
import gym
from gym import Wrapper
from gym.spaces import Box
import torch
import torch.optim as optim
from torch import nn as nn
import torchrl.algo.utils as atu
from torchrl.utils import get_params
from torchrl.env import get_env
from torchrl.utils import Logger
import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.collector.base import BaseCollector
from torchrl.algo import SAC
from torchrl.algo import TwinSAC
from torchrl.algo import TwinSACQ
from torchrl.algo import MTSAC
from torchrl.collector.para import ParallelCollector
from torchrl.collector.para import AsyncParallelCollector
from torchrl.collector.para.mt import SingleTaskParallelCollectorBase
from torchrl.replay_buffers import BaseReplayBuffer
from torchrl.replay_buffers.shared import SharedBaseReplayBuffer
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
from torchrl.env.continuous_wrapper import *
from torchrl.env.get_env import wrap_continuous_env
import matplotlib.pyplot as plt
import constopt
from constopt.constraints import LinfBall
from constopt.stochastic import PGD, PGDMadry, FrankWolfe, MomentumFrankWolfe
import torch
from torch.autograd import Variable
import torch.nn.utils as utils
from scipy.stats import rankdata
os.environ['LD_LIBRARY_PATH'] = '/root/.mujoco/mujoco210/bin:/usr/lib/nvidia'
sys.path.insert(0, './metaworld-master/')
sys.path.append("./")
sys.path.append("../..")
sys.path.insert(0, r'./constopt-pytorch/')

class SingleWrapper(Wrapper):
    def __init__(self, env):
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.train_mode = True
    def reset(self):
        return self._env.reset()

    def seed(self, se):
        self._env.seed(se)

    def reset_with_index(self, task_idx):
        return self._env.reset()

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info

    def train(self):
        self.train_mode = True

    def test(self):
        self.train_mode = False
    def eval(self):
        self.train_mode = False

    def render(self, mode = 'human', **kwargs):
        return self._env.render(mode = mode, **kwargs)

    def close(self):
        self._env.close()

class Normalizer():
    def __init__(self, shape, clip = 10.):
        self.shape = shape
        self._mean = np.zeros(shape)
        self._var = np.ones(shape)
        self._count = 1e-4
        self.clip = clip
        self.should_estimate = True

    def stop_update_estimate(self):
        self.should_estimate = False

    def update_estimate(self, data):
        if not self.should_estimate:
            return
        if len(data.shape)  ==   self.shape:
            data = data[np.newaxis, :]
        self._mean, self._var, self._count = update_mean_var_count(
            self._mean, self._var, self._count, 
            np.mean(data, axis = 0), np.var(data, axis = 0), data.shape[0])

    def inverse(self, raw):
        return raw * np.sqrt(self._var)  +  self._mean

    def inverse_torch(self, raw):
        return raw * torch.Tensor(np.sqrt(self._var)).to(raw.device) \
             +  torch.Tensor(self._mean).to(raw.device)

    def filt(self, raw):
        return np.clip(
            (raw - self._mean) / (np.sqrt(self._var)  +  1e-4), 
            -self.clip, self.clip)

    def filt_torch(self, raw):
        return torch.clamp(
            (raw - torch.Tensor(self._mean).to(raw.device)) / \
            (torch.Tensor(np.sqrt(self._var)  +  1e-4).to(raw.device)), 
            -self.clip, self.clip)

class RLAlgo():
    """
    Base RL Algorithm Framework
    """
    def __init__(self, 
        env = None, 
        replay_buffer = None, 
        collector = None, 
        logger = None, 
        continuous = None, 
        discount = 0.99, 
        num_epochs = 3000, 
        epoch_frames = 200, 
        max_episode_frames = 999, 
        batch_size = 128, 
        device = 'cpu', 
        train_render = False, 
        eval_episodes = 1, 
        eval_render = False, 
        save_interval = 100, 
        save_dir = None
    ):
        self.env = env
        self.total_frames = 0
        self.continuous = isinstance(self.env.action_space, gym.spaces.Box)

        self.replay_buffer = replay_buffer
        self.collector = collector        
        # device specification
        self.device = device

        # environment relevant information
        self.discount = discount
        self.num_epochs = num_epochs
        self.epoch_frames = epoch_frames
        self.max_episode_frames = max_episode_frames

        self.train_render = train_render
        self.eval_render = eval_render

        # training information
        self.batch_size = batch_size
        self.training_update_num = 0
        self.sample_key = None

        # Logger & relevant setting
        self.logger = logger
        self.episode_rewards = deque(maxlen = 30)
        self.training_episode_rewards = deque(maxlen = 30)
        self.eval_episodes = eval_episodes

        self.save_interval = save_interval
        self.save_dir = save_dir
        if not osp.exists( self.save_dir ):
            os.mkdir( self.save_dir )

        self.best_eval = None

    def start_epoch(self):
        pass

    def finish_epoch(self):
        return {}

    def pretrain(self):
        pass
    
    def update_per_epoch(self):
        pass

    def snapshot(self, prefix, epoch):
        for name, network in self.snapshot_networks:
            model_file_name = "model_{}_{}.pth".format(name, epoch)
            model_path = osp.join(prefix, model_file_name)
            torch.save(network.state_dict(), model_path)

    def train(self, epoch):
        if epoch =  = 1:
            self.pretrain()
            self.total_frames = 0
            if hasattr(self, "pretrain_frames"):
                self.total_frames = self.pretrain_frames

            self.start_epoch()

        self.current_epoch = epoch
        start = time.time()

        self.start_epoch()

        explore_start_time = time.time()
        training_epoch_info =  self.collector.train_one_epoch()
        for reward in training_epoch_info["train_rewards"]:
            self.training_episode_rewards.append(reward)
        explore_time = time.time() - explore_start_time

        train_start_time = time.time()
        loss = self.update_per_epoch()
        train_time = time.time() - train_start_time

        finish_epoch_info = self.finish_epoch()
        eval_start_time = time.time()
        eval_infos = self.collector.eval_one_epoch()
        eval_time = time.time() - eval_start_time

        self.total_frames  +=  self.collector.active_worker_nums * self.epoch_frames

        infos = {}

        for reward in eval_infos["eval_rewards"]:
            self.episode_rewards.append(reward)
        # del eval_infos["eval_rewards"]

        if self.best_eval is None or \
            np.mean(eval_infos["eval_rewards"]) > self.best_eval:
            self.best_eval = np.mean(eval_infos["eval_rewards"])
            self.snapshot(self.save_dir, 'best')
        del eval_infos["eval_rewards"]
        infos["eval_avg_success_rate"]  = eval_infos["success"]
        infos["Running_Average_Rewards"] = np.mean(self.episode_rewards)
        infos["Running_success_rate"]  = training_epoch_info["train_success_rate"]
        infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
        infos["Running_Training_Average_Rewards"] = np.mean(
            self.training_episode_rewards)
        infos["Explore_Time"] = explore_time
        infos["Train___Time"] = train_time
        infos["Eval____Time"] = eval_time
        infos.update(eval_infos)
        infos.update(finish_epoch_info)

        self.logger.add_epoch_info(epoch, self.total_frames, 
            time.time() - start, infos )

        if epoch % self.save_interval  ==   0:
            self.snapshot(self.save_dir, epoch)
        if epoch =  = self.num_epochs-1:
            self.snapshot(self.save_dir, "finish")
            self.collector.terminate()
        return loss
      
    def update(self, batch):
        raise NotImplementedError

    def _update_target_networks(self):
        if self.use_soft_update:
            for net, target_net in self.target_networks:
                atu.soft_update_from_to(net, target_net, self.tau)
        else:
            if self.training_update_num % self.target_hard_update_period  ==   0:
                for net, target_net in self.target_networks:
                    atu.copy_model_params_from_to(net, target_net)

    @property
    def networks(self):
        return [
        ]
    
    @property
    def snapshot_networks(self):
        return [
        ]

    @property
    def target_networks(self):
        return [
        ]
    
    def to(self, device):
        for net in self.networks:
            net.to(device)

class OffRLAlgo(RLAlgo):
    """
    Base RL Algorithm Framework
    """
    def __init__(self, 

        pretrain_epochs = 0, 

        min_pool = 0, 

        target_hard_update_period = 1000, 
        use_soft_update = True, 
        tau = 0.001, 
        opt_times = 1, 

        **kwargs
    ):
        super(OffRLAlgo, self).__init__(**kwargs)

        # environment relevant information
        self.pretrain_epochs = pretrain_epochs
        
        # target_network update information
        self.target_hard_update_period = target_hard_update_period
        self.use_soft_update = use_soft_update
        self.tau = tau

        # training information
        self.opt_times = opt_times
        self.min_pool = min_pool

        self.sample_key = [ "obs", "next_obs", "acts", "rewards", "terminals" ]

    def update_per_timestep(self):
        if self.replay_buffer.num_steps_can_sample() > max( self.min_pool, self.batch_size ):
            for _ in range( self.opt_times ):
                batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
                infos = self.update( batch )
                self.logger.add_update_info( infos )

    def update_per_epoch(self):
        loss = []
        for _ in range( self.opt_times ):
            batch = self.replay_buffer.random_batch(self.batch_size, self.sample_key)
            infos = self.update( batch )
            loss.append(infos['Training/policy_loss'])
            self.logger.add_update_info( infos )
        return np.mean(loss)
    
    def pretrain(self):
        total_frames = 0
        self.pretrain_epochs * self.collector.worker_nums * self.epoch_frames
        
        for pretrain_epoch in range( self.pretrain_epochs ):

            start = time.time()

            self.start_epoch()
            
            training_epoch_info =  self.collector.train_one_epoch()
            for reward in training_epoch_info["train_rewards"]:
                self.training_episode_rewards.append(reward)

            finish_epoch_info = self.finish_epoch()

            total_frames  +=  self.collector.active_worker_nums * self.epoch_frames
            
            infos = {}
            
            infos["Train_Epoch_Reward"] = training_epoch_info["train_epoch_reward"]
            infos["Running_Training_Average_Rewards"] = np.mean(self.training_episode_rewards)
            infos.update(finish_epoch_info)
            
            self.logger.add_epoch_info(pretrain_epoch, total_frames, time.time() - start, infos, csv_write = False )
        
        self.pretrain_frames = total_frames

        self.logger.log("Finished Pretrain")

class SAC(OffRLAlgo):
    """
    SAC
    """
    def __init__(
            self, 
            pf, vf, qf, 
            plr, vlr, qlr, 
            optimizer_class = optim.Adam, 
            
            policy_std_reg_weight = 1e-3, 
            policy_mean_reg_weight = 1e-3, 

            reparameterization = True, 
            automatic_entropy_tuning = True, 
            target_entropy = None, 
            **kwargs
    ):
        super(SAC, self).__init__(**kwargs)
        self.pf = pf
        self.qf = qf
        self.vf = vf
        self.target_vf = copy.deepcopy(vf)
        self.to(self.device)

        self.plr = plr
        self.vlr = vlr
        self.qlr = qlr

        self.qf_optimizer = optimizer_class(
            self.qf.parameters(), 
            lr = self.qlr, 
        )

        self.vf_optimizer = optimizer_class(
            self.vf.parameters(), 
            lr = self.vlr, 
        )

        self.pf_optimizer = optimizer_class(
            self.pf.parameters(), 
            lr = self.plr, 
        )
        
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # from rlkit
            self.log_alpha = torch.zeros(1).to(self.device)
            self.log_alpha.requires_grad_()
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha], 
                lr = self.plr, 
            )

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_mean_reg_weight = policy_mean_reg_weight

        self.reparameterization = reparameterization

    def update(self, batch):
        self.training_update_num  +=  1
        
        obs = batch['obs']
        actions = batch['acts']
        next_obs = batch['next_obs']
        rewards = batch['rewards']
        terminals = batch['terminals']

        rewards = torch.Tensor(rewards).to( self.device )
        terminals = torch.Tensor(terminals).to( self.device )
        obs = torch.Tensor(obs).to( self.device )
        actions = torch.Tensor(actions).to( self.device )
        next_obs = torch.Tensor(next_obs).to( self.device )

        """
        Policy operations.
        """
        sample_info = self.pf.explore(obs, return_log_probs = True )

        mean        = sample_info["mean"]
        log_std     = sample_info["log_std"]
        new_actions = sample_info["action"]
        log_probs   = sample_info["log_prob"]
        ent         = sample_info["ent"]

        q_pred = self.qf([obs, actions])
        v_pred = self.vf(obs)

        if self.automatic_entropy_tuning:
            """
            Alpha Loss
            """
            alpha_loss = -(self.log_alpha * (log_probs  +  self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1
            alpha_loss = 0

        """
        QF Loss
        """
        target_v_values = self.target_vf(next_obs)
        q_target = rewards  +  (1. - terminals) * self.discount * target_v_values
        qf_loss = self.qf_criterion( q_pred, q_target.detach())

        """
        VF Loss
        """
        q_new_actions = self.qf([obs, new_actions])
        v_target = q_new_actions - alpha * log_probs
        vf_loss = self.vf_criterion( v_pred, v_target.detach())

        """
        Policy Loss
        """
        if not self.reparameterization:
            log_policy_target = q_new_actions - v_pred
            policy_loss = (
                log_probs * ( alpha * log_probs - log_policy_target).detach()
            ).mean()
        else:
            policy_loss = ( alpha * log_probs - q_new_actions).mean()

        std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean()

        policy_loss  +=  std_reg_loss  +  mean_reg_loss
        
        """
        Update Networks
        """
        self.pf_optimizer.zero_grad()
        
        w = []
        for key in pfs[0].state_dict().keys():
            w.append(torch.cat([pfs[j].state_dict()[key].unsqueeze(0) for j in range(len(envs))]))            
        
        rloss[index] = policy_loss.clone().detach().item()
        
        if multitask:
            pre = rloss[index] + lossw(currindex, index, rloss, w, B)/10   
        else:
            pre = rloss[index]
        # compute gradients

        pre.backward()

        # train the NN
    
        self.pf_optimizer.step()
        rloss[index] = rloss[index].detach().item()
        
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self._update_target_networks()

        # Information For Logger
        info = {}
        info['Reward_Mean'] = rewards.mean().item()

        if self.automatic_entropy_tuning:
            info["Alpha"] = alpha.item()
            info["Alpha_loss"] = alpha_loss.item()
        info['Training/policy_loss'] = policy_loss.item()
        info['Training/vf_loss'] = vf_loss.item()
        info['Training/qf_loss'] = qf_loss.item()

        info['log_std/mean'] = log_std.mean().item()
        info['log_std/std'] = log_std.std().item()
        info['log_std/max'] = log_std.max().item()
        info['log_std/min'] = log_std.min().item()

        info['log_probs/mean'] = log_std.mean().item()
        info['log_probs/std'] = log_std.std().item()
        info['log_probs/max'] = log_std.max().item()
        info['log_probs/min'] = log_std.min().item()

        info['mean/mean'] = mean.mean().item()
        info['mean/std'] = mean.std().item()
        info['mean/max'] = mean.max().item()
        info['mean/min'] = mean.min().item()

        return info

    @property
    def networks(self):
        return [
            self.pf, 
            self.qf, 
            self.vf, 
            self.target_vf
        ]
    
    @property
    def snapshot_networks(self):
        return [
            ["pf", self.pf], 
            ["qf", self.qf], 
            ["vf", self.vf]
        ]

    @property
    def target_networks(self):
        return [
            ( self.vf, self.target_vf )
        ]

class parser:
    def __init__(self): 
        self.config = 'config/sac_ant.json'
        self.id = 'mt10'
        self.worker_nums = 10
        self.eval_worker_nums = 10
        self.seed = 20
        self.vec_env_nums = 1
        self.save_dir = './save/sac_ant'
        self.log_dir = './log/sac_ant'
        self.no_cuda = True
        self.overwrite = True
        self.device = 'cpu'
        self.cuda = False
     
ml10 = metaworld.MT10() # Construct the benchmark, sampling tasks
envs = []
for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name  ==   name])
  env.set_task(task)
  envs.append(env)

for env in envs:
  obs = env.reset()  # Reset environment
  a = env.action_space.sample()  # Sample an action
  obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action      

text = """{
    "env_name" : "mt10", 
    "env":{
        "reward_scale":1, 
        "obs_norm":false
    }, 
    "meta_env":{
        "obs_type": "with_goal_and_id"
    }, 
    "replay_buffer":{
        "size": 1e6
    }, 
    "net":{ 
        "hidden_shapes": [128, 64, 32, 16], 
        "append_hidden_shapes":[]
    }, 
    "general_setting": {
        "discount" : 0.99, 
        "pretrain_epochs" : 20, 
        "num_epochs" : 7500, 
        "epoch_frames" : 200, 
        "max_episode_frames" : 200, 

        "batch_size" : 1280, 
        "min_pool" : 10000, 

        "target_hard_update_period" : 1000, 
        "use_soft_update" : true, 
        "tau" : 0.005, 
        "opt_times" : 200, 

        "eval_episodes" : 3
    }, 
    "sac":{
    }
}"""

!mkdir config
with open('config/sac_ant.json', 'w') as f:
    f.write(text)
                
args = parser()
params = get_params(args.config)
device = torch.device(
    "cuda:{}".format(args.device) if args.cuda else "cpu")
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

pfs = []
qf1s = []
vfs = []
agents = []
epochs = [1 for i in range(len(envs))]    
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
normalizer = Normalizer(env.observation_space.shape)
buffer_param = params['replay_buffer']
experiment_name = os.path.split(
    os.path.splitext(args.config)[0])[-1] if args.id is None \
    else args.id
logger = Logger(
    experiment_name, params['env_name'], args.seed, params, args.log_dir)

for index in range(len(envs)):
    env = SingleWrapper(envs[index])
    params = get_params(args.config)
    params['general_setting']['logger'] =  Logger(
            'mt10', str(index), args.seed, params, './log/mt10_' + str(index) + '/')
    params['env_name'] = str(index)
    params['general_setting']['env'] = env

    replay_buffer = BaseReplayBuffer(
        max_replay_buffer_size = int(buffer_param['size'])
    )
    params['general_setting']['replay_buffer'] = replay_buffer

    params['general_setting']['device'] = device

    params['net']['base_type'] = networks.MLPBase
    params['net']['activation_func'] = torch.nn.ReLU
    
    pf = policies.GuassianContPolicy(
        input_shape = env.observation_space.shape[0], 
        output_shape = 2 * env.action_space.shape[0], 
        **params['net'], 
        **params['sac'])

    qf1 = networks.QNet(
        input_shape = env.observation_space.shape[0]  +  env.action_space.shape[0], 
        output_shape = 1, 
        **params['net'])

    vf = networks.Net(
            input_shape = env.observation_space.shape, 
            output_shape = 1, 
            **params['net']
        )
    pfs.append(pf)
    qf1s.append(qf1)
    vfs.append(vf)
    params['general_setting']['collector'] = BaseCollector(
        env = env, pf = pf, 
        replay_buffer = replay_buffer, device = device, 
        train_render = False
    )
    params['general_setting']['save_dir'] = osp.join(
        './log/', "model10_" + str(index))
    agent = SAC(
            pf = pf, 
            qf = qf1, plr = 3e-4, vlr = 3e-4, qlr = 3e-4, 
            vf = vf, 
            **params["sac"], 
            **params["general_setting"]
        )
    agents.append(agent)

# differentiable ranking loss
def pss(x, points):
    def pss0(x, i):
        return torch.tanh(200*torch.tensor(x-i))/2 + 0.5
    return len(points)-sum([pss0(x, i) for i in points])

def losst(currindex, t, rloss, w, B, mu = 0.2, lamb = [0.01, 0.01, 0.01], U = [13], pi = list(range(len(envs)))):
    new_rloss = [i for i in rloss]
    new_rloss[t] = new_rloss[t] + 1
    rlossRank = 1 + len(envs) - rankdata(new_rloss, method = 'min')
    points = B[t]
    sim = [sum(nn.CosineSimilarity()(pfs[t].state_dict()['last.weight'].view(-1, 1), pfs[i].state_dict()['last.weight'].view(-1, 1))) for i in range(len(envs))] 
    sim[t] = sim[t] + 100
    rlossRank_renew = 1 + len(envs) - rankdata(sim, method = 'min')
    term0 = (1 + mu*sum([torch.norm(torch.tensor(B[t][i]), p = 1)for i in set(list(range(len(envs))))-set([t])]))*rloss[t]
    term1 = sum([sum([sum([torch.norm(w[i][s]-sum([B[pi[j]][s]*w[i][pi[j]] for j in range(currindex-1)])-B[t][s]*w[i][t],p=2)**2]) for i in range(len(pfs[0].state_dict().keys()))]) for s in U])
    term2 = torch.norm(torch.tensor(priors[current])-torch.tensor([pss(torch.tensor(i-0.01),points) for i in points]))**2
    term3 = torch.norm(torch.tensor(rlossRank)-torch.tensor([pss(torch.tensor(i-0.01),points) for i in points]))**2
    term4 = torch.norm(torch.tensor(rlossRank2)-torch.tensor([pss(torch.tensor(i-0.01),points) for i in points]))**2
    terms_history[0].append(term0.detach().numpy())
    terms_history[1].append(term1)
    terms_history[2].append(term2.detach().numpy())
    terms_history[3].append(term3.detach().numpy())
    terms_history[4].append(term4.detach().numpy())
    if len(terms_history[0])<=1:
        return term0+term1+term2+term3+term4
    else:
        return (1/(np.array(terms_history[0]).std())**2)*term0+(1/(np.array(terms_history[1]).std())**2)*term1+(1/(np.array(terms_history[2]).std())**2)*term2+(1/(np.array(terms_history[3]).std())**2)*term3+(1/(np.array(terms_history[4]).std())**2)*term4+np.log((np.array(terms_history[0]).std())*(np.array(terms_history[1]).std())*(np.array(terms_history[2]).std())*(np.array(terms_history[3]).std())*(np.array(terms_history[4]).std()))

def lossb(currindex, t, rloss, w, B, mu = 0.2, lamb = [0.01, 0.01, 0.01], U = [13], pi = list(range(len(envs)))):
    new_rloss = [i for i in rloss]
    new_rloss[t] = new_rloss[t] + 1
    rlossRank = 1 + len(envs) - rankdata(new_rloss, method = 'min')
    points = B[t]
    sim = [sum(nn.CosineSimilarity()(pfs[t].state_dict()['last.weight'].view(-1, 1), pfs[i].state_dict()['last.weight'].view(-1, 1))) for i in range(len(envs))] 
    sim[t] = sim[t] + 100
    rlossRank_renew = 1 + len(envs) - rankdata(sim, method = 'min')
    term0 = (1 + mu*sum([torch.norm(B[t][i], p = 1)for i in set(list(range(len(envs))))-set([t])]))*rloss[t]
    term1 = sum([sum([sum([torch.norm(w[i][s]-sum([B[pi[j]][s]*w[i][pi[j]] for j in range(currindex-1)])-B[t][s]*w[i][t],p=2)**2]) for i in range(len(pfs[0].state_dict().keys()))]) for s in U])
    term2 = torch.norm(torch.tensor(priors[current])-torch.tensor([pss(torch.tensor(i-0.01),points) for i in points]))**2
    term3 = torch.norm(torch.tensor(rlossRank)-torch.tensor([pss(torch.tensor(i-0.01),points) for i in points]))**2
    term4 = torch.norm(torch.tensor(rlossRank2)-torch.tensor([pss(torch.tensor(i-0.01),points) for i in points]))**2
    if len(terms_history[0])<=1:
        return term0+term1+term2+term3+term4
    else:
        return (1/(np.array(terms_history[0]).std())**2)*term0+(1/(np.array(terms_history[1]).std())**2)*term1+(1/(np.array(terms_history[2]).std())**2)*term2+(1/(np.array(terms_history[3]).std())**2)*term3+(1/(np.array(terms_history[4]).std())**2)*term4+np.log((np.array(terms_history[0]).std())*(np.array(terms_history[1]).std())*(np.array(terms_history[2]).std())*(np.array(terms_history[3]).std())*(np.array(terms_history[4]).std()))

def lossw(currindex, t, rloss, w, B, mu = 0.2, lamb = [0.01, 0.01, 0.01], U = [13], pi = list(range(len(envs)))):
    new_rloss = [i for i in rloss]
    new_rloss[t] = new_rloss[t] + 1
    rlossRank = 1 + len(envs) - rankdata(new_rloss, method = 'min')
    points = B[t]
    term0 = (1 + mu*sum([torch.norm(torch.tensor(B[t][i]), p=1)for i in set(list(range(len(envs))))-set([t])]))*rloss[t]
    sim = [sum(nn.CosineSimilarity()(pfs[t].state_dict()['last.weight'].view(-1, 1), pfs[i].state_dict()['last.weight'].view(-1, 1))) for i in range(len(envs))] 
    sim[t] = sim[t] + 100
    rlossRank_renew = 1 + len(envs) - rankdata(sim, method = 'min')

    term1 = sum([sum([torch.norm(w[i][t]-sum([B[pi[j]][t]*w[i][pi[j]] for j in range(currindex-1)]),p=2)**2]) for i in range(len(pfs[0].state_dict().keys()))])
    term2 = torch.norm(torch.tensor(priors[current])-torch.tensor([pss(torch.tensor(i-0.01),points) for i in points]))**2
    term3 = torch.norm(torch.tensor(rlossRank)-torch.tensor([pss(torch.tensor(i-0.01),points) for i in points]))**2
    term4 = torch.norm(torch.tensor(rlossRank2)-torch.tensor([pss(torch.tensor(i-0.01),points) for i in points]))**2
    if len(terms_history[0])<=1:
        return term0+term1+term3+term4
    else:
        return (1/(np.array(terms_history[0]).std())**2)*term0+(1/(np.array(terms_history[1]).std())**2)*term1+(1/(np.array(terms_history[2]).std())**2)*term2+(1/(np.array(terms_history[3]).std())**2)*term3+(1/(np.array(terms_history[4]).std())**2)*term4+np.log((np.array(terms_history[0]).std())*(np.array(terms_history[1]).std())*(np.array(terms_history[2]).std())*(np.array(terms_history[3]).std())*(np.array(terms_history[4]).std()))

# FrankWolfe
OPTIMIZER_CLASSES = [FrankWolfe]
radius = 0.05

def setup_problem(make_nonconvex = False):
    radius2 = radius
    loss_func = lossb
    constraint = LinfBall(radius2)

    return loss_func, constraint
  
def optimize(loss_func, constraint, optimizer_class, iterations = 100):
    for i in range(len(envs)):
        if i! = t:
            B[t][i]  = torch.tensor(B[t][i], requires_grad = True)
    optimizer = [optimizer_class([B[t][i]], constraint) for i in set(list(range(len(envs))))-set([t])]
    iterates = [[B[t][i].data if i! = t else B[t][i] for i in range(len(envs))]]
    losses = []
    # Use Madry's heuristic for step size
    step_size = {
        FrankWolfe.name: None, 
        MomentumFrankWolfe.name: None, 
        PGD.name: 2.5 * constraint.alpha / iterations * 2., 
        PGDMadry.name: 2.5 * constraint.alpha / iterations
    }
    for _ in range(iterations):
        for i in range(len(envs)-1):
            optimizer[i].zero_grad()
        loss = loss_func(currindex, t, rloss, w, B, U = list(set(U)-set(list([t]))))
        loss.backward(retain_graph = True)
        for i in  range(len(envs)-1):
            optimizer[i].step(step_size[optimizer[i].name])
        for i in set(list(range(len(envs))))-set([t]):
            B[t][i].data.clamp_(0, 100)
        losses.append(loss)
        iterates.append([B[t][i].data if i! = t else B[t][i] for i in range(len(envs))])
    loss = loss_func(currindex, t, rloss, w, B, U = list(set(U)-set(list([t])))).detach()
    losses.append(loss)
    B[t] = [B[t][i].data if i! = t else B[t][i] for i in range(len(envs))]
    return losses, iterates

multitask = False
rloss = [0.0 for i in range(len(envs))]
rewardsRec = [[] for i in range(len(envs))]
succeessRec = [[] for i in range(len(envs))]

for i_episode in range(10000):
    for index, env in enumerate(envs):
            agents[index].train(epochs[index])
            epochs[index] += 1
