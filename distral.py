import metaworld
import random
from torch.distributions.normal import Normal
import math

class parser:
    def __init__(self):
        self.gamma=0.99
        self.alpha=0.9
        self.beta=.5
        self.seed=543
        self.render=False
        self.log_interval=10
        self.envs=envs
        
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.num_envs = num_envs
        # shared layer
        # not shared layers

        self.mu_heads = nn.ModuleList ( [ nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, 16),
            nn.ReLU(),             
            nn.Linear(16,4)
        ) for i in range(self.num_envs+1) ] )
        self.sigma2_heads =nn.ModuleList ( [ nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, 16),
            nn.ReLU(),             
            nn.Linear(16,4)
        ) for i in range(self.num_envs+1) ] )
        self.value_heads = nn.ModuleList([nn.Linear(16, 4) for i in range(self.num_envs)])
        self.apply(weights_init)
        # +1 for the distilled policy
        # initialize lists for holding run information
        self.div = [[] for i in range(num_envs)]
        self.saved_actions = [[] for i in range(self.num_envs)]
        #self.entropies = [[] for i in range(num_envs)]
        self.entropies = []
        self.rewards = [[] for i in range(self.num_envs)]

    def forward(self, y, index):
        '''updated to have 5 return values (2 for each action head one for
        value'''
        x = y
        mu =  F.softmax(self.mu_heads[index](x),dim=-1)[0]
        sigma2 = self.sigma2_heads[index](x)
        sigma = F.softplus(sigma2)
        value = self.value_heads[index](x)
        mu_dist =  F.softmax(self.mu_heads[-1](x),dim=-1)[0]
        sigma2_dist = self.sigma2_heads[-1](x)
        sigma_dist = F.softplus(sigma2_dist)
        return mu, sigma, value, mu_dist, sigma_dist

def select_action(state, index):
    '''given a state, this function chooses the action to take
    arguments: state - observation matrix specifying the current model state
               env - integer specifying which environment to sample action
                         for
    return - action to take'''

    state=Variable(state)
    mu, sigma, value, mu_t, sigma_t = model(state, index)

    prob = Normal(args.alpha*mu_t + args.beta*mu, args.alpha*sigma_t.sqrt() + \
                  args.beta*sigma.sqrt())

    entropy = 0.5*(((args.alpha*sigma_t + args.beta*sigma)*2*pi).log()+1)


    new_KL = torch.div(sigma_t.sqrt(),args.alpha*sigma_t.sqrt() + \
                       args.beta*sigma.sqrt()).log() + \
             torch.div((args.alpha*sigma_t.sqrt() + \
             args.beta*sigma.sqrt()).pow(2) + \
             ((args.alpha-1)*mu_t+(args.beta)*mu).pow(2),(2*sigma_t)) - 0.5

    log_prob = prob.loc.log()
    model.saved_actions[index].append(SavedAction(log_prob, value))
    model.entropies.append(entropy)
    model.div[index].append(new_KL)

    # model.div[index].append(torch.div(tsigma.sqrt(),sigma.sqrt()).log() + torch.div(sigma+(tmu-mu).pow(2),tsigma*2) - 0.5)
    return prob.loc

def finish_episode():
    policy_losses = []
    value_losses = []
    entropy_sum = 0
    loss = torch.zeros(1, 1)
    loss = Variable(loss)

    for index in range(num_envs):
        saved_actions = model.saved_actions[index]
        model_rewards = model.rewards[index]
        R = torch.zeros(1, 1)
        R = Variable(R)
        rewards = []
        # compute the reward for each state in the end of a rollout
        for r in model_rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        if rewards.std() != rewards.std() or len(rewards) == 0:
            rewards = rewards - rewards.mean()
        else:
            rewards = (rewards - rewards.mean()) / (rewards.std()+1e-3)

        for i, reward in enumerate(rewards):
            rewards = rewards + args.gamma**i * model.div[index][i].mean()

        for (log_prob, value), r in zip(saved_actions, rewards):
            # reward is the delta param
            value += Variable(torch.randn(value.size()))
            reward = r - value[0].dot(log_prob[0].exp()).item()
            # theta
            # need gradient descent - so negative
            policy_losses.append(-log_prob * reward) #/ length_discount[index])
            # https://pytorch.org/docs/master/nn.html#torch.nn.SmoothL1Loss
            # feeds a weird difference between value and the reward
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))

    loss = (torch.stack(policy_losses).sum() + \
            0.5*torch.stack(value_losses).sum() - \
            torch.stack(model.entropies).sum() * 0.0001) / num_envs

    # Debugging
    if False:
        print(divergence[0].data)
        print(loss, 'loss')
        print()
    # compute gradients
    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 30)

    # Debugging
    if False:
        print('grad')
        for i in range(num_envs):
            print(i)
            print(model.mu_heads[i].weight)
            print('##')
            #print(model.sigma2_head[i].weight.grad)
            print('##')
            #print(model.value_head[i].weight.grad)
            print('##')
            #print(model.affine1.weight.grad)

    # train the NN
    optimizer.step()

    model.div = [[] for i in range(num_envs)]
    model.saved_actions = [[] for i in range(num_envs)]
    model.entropies = []
    model.rewards = [[] for i in range(model.num_envs)]

ml10 = metaworld.MT10() # Construct the benchmark, sampling tasks
envs = []
for name, env_cls in ml10.train_classes.items():
  env = env_cls()
  task = random.choice([task for task in ml10.train_tasks
                        if task.env_name  ==   name])
  env.set_task(task)
  envs.append(env)

pi = Variable(torch.FloatTensor([math.pi]))    
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
test = False
trained = False 
args=parser()
model = Policy()
# learning rate - might be useful to change
optimizer = optim.Adam(model.parameters(), lr=1e-3)

running_reward = 10
run_reward = np.array([10 for i in range(num_envs)])
roll_length = np.array([0 for i in range(num_envs)])
trained_envs = np.array([False for i in range(num_envs)])
rewardsRec=[[] for i in range(num_envs)]
for i_episode in range(6000):
    p = np.random.random()
    # roll = np.random.randint(2)
    length = 0
    for index, env in enumerate(envs):
        # Train each environment simultaneously with the distilled policy
        state = env.reset()
        r=0
        done = False
        for t in range(200):  # Don't infinite loop while learning
            action = select_action(state, index)
            state, reward, done, _ = env.step(Categorical(action).sample())
            r+= reward
            model.rewards[index].append(reward)
            if args.render:
                env.render()
            if done or t==199:
                print(i_episode ,index,r)
                rewardsRec[index].append(r)
                length += t
                roll_length[index] = t
                break
    np.save('distral_rewardsRec.npy',rewardsRec)
    torch.save(model.state_dict(), 'distral_params.pkl')
    finish_episode()
