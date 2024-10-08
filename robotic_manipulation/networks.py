import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super(Actor, self).__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), nn.ReLU, nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, x, deterministic=False, return_D=False):
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        if deterministic:
            action = mu
            log_prob = None
        else:
            m = Normal(mu, std)
            if return_D:
                return m
            action = m.rsample()

            log_prob = m.log_prob(action).sum(axis=-1)

            log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)

        action = torch.tanh(action)
        return action, log_prob


class Critic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super(Critic, self).__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)

    def forward(self, state, action):
        q = self.q(torch.cat([state, action], dim=-1))
        return torch.squeeze(q, -1)


class Vf(nn.Module):

    def __init__(self, obs_dim, hidden_sizes=(256, 256)):
        super(Vf, self).__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [1], nn.ReLU)

    def forward(self, state):
        q = self.q(state)
        return torch.squeeze(q, -1)