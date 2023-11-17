import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gymnasium
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0)
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic

def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.get_wrapper_attr('state')).float()
    values, logprobs, rewards = [], [], []
    done = False
    j = 0
    while (not done):
        j += 1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, truncated, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()

        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)

    return values, logprobs, rewards


def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):
    # reverse the order of the rewards, logprobs and values
    # .view(-1) is called to make sure that they are flat
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = torch.Tensor([0])

    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)

    # we need to detach the values tensor from the graph to prevent
    # back propagating through the critic head
    actor_loss = -1 * logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns, 2)
    # sum the actor and critic losses to get an overall loss
    # we scale down the critic loss because we want the actor to learn
    # faster than the critic
    loss = actor_loss.sum() + clc * critic_loss.sum()
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)


# the main training loop
def worker(t, worker_model, counter, params):
    worker_env = gymnasium.make('CartPole-v1')
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()

    for i in range(params['epochs']):
        clear_console()
        print('training at %d / %d epoch' % (i + 1, params['epochs']))

        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, worker_model)
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards)
        counter.value = counter.value + 1

def test_model(model):
    # env = gymnasium.make("CartPole-v1", render_mode='human')
    env = gymnasium.make("CartPole-v1")
    env.reset()

    episodes = 200
    episode_lens = []

    for episode in range(episodes):
        counter = 0
        for i in range(200):
            counter += 1
            state_ = np.array(env.env.get_wrapper_attr('state'))
            state = torch.from_numpy(state_).float()
            logits,value = MasterNode(state)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            state2, reward, done, truncated, info = env.step(action.detach().numpy())
            if done:
                env.reset()
                break

            # state_ = np.array(env.env.get_wrapper_attr('state'))
            # state = torch.from_numpy(state_).float()
            # env.render()

        episode_lens.append(counter)

    return episode_lens

if __name__ == '__main__':
    MasterNode = ActorCritic()
    MasterNode.share_memory()
    processes = []
    params = {
        'epochs': 1000,
        'n_workers': 7,
    }

    counter = mp.Value('i', 0)
    for i in range(params['n_workers']):
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for p in processes:
        p.terminate()

    print(counter.value, processes[1].exitcode)

    episode_lens = test_model(MasterNode)

    plt.xlabel('episode')
    plt.ylabel('episode length')
    plt.scatter(np.arange(len(episode_lens)), episode_lens)
    plt.show()
