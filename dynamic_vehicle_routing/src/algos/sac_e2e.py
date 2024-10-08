import numpy as np
import torch
from torch import batch_norm, nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from collections import namedtuple
import random
from collections import namedtuple
import json

"""_summary_
Online SAC E2E agent for the AMoD control problem.
"""

args = namedtuple("args", ("render", "gamma", "log_interval"))
args.render = True
args.gamma = 0.99
args.log_interval = 10

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class PairData(Data):
    def __init__(
        self,
        edge_index_s=None,
        x_s=None,
        reward=None,
        action=None,
        edge_index_t=None,
        x_t=None,
    ):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ReplayData:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, device):
        self.device = device
        self.data_list = []
        self.rewards = []

    def store(self, data1, action, reward, data2):
        self.data_list.append(
            PairData(
                data1.edge_index,
                data1.x,
                torch.as_tensor(reward),
                torch.as_tensor(action),
                data2.edge_index,
                data2.x,
            )
        )
        self.rewards.append(reward)

    def size(self):
        return len(self.data_list)

    def sample_batch(self, batch_size=32, norm=False):
        data = random.sample(self.data_list, batch_size)
        if norm:
            mean = np.mean(self.rewards)
            std = np.std(self.rewards)
            batch = Batch.from_data_list(data, follow_batch=["x_s", "x_t"])
            batch.reward = (batch.reward - mean) / (std + 1e-16)
            return batch.to(self.device)
        else:
            return Batch.from_data_list(data, follow_batch=["x_s", "x_t"]).to(
                self.device
            )


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant


#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t)
    """

    def __init__(
        self,
        in_channels,
        hidden_size=32,
        act_dim=14,
        edges=None,
        low=0,
        high=480,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(
            2 * in_channels, hidden_size
        )  # Input is from 2 nodes (edge = node1 + node2)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.h_to_mu = nn.Linear(hidden_size, 1)
        self.h_to_sigma = nn.Linear(hidden_size, 1)
        self.edges = edges
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.low = low
        self.high = high

    def _squash(self, raw_values):
        # Returned values are within [low, high] (including `low` and `high`).
        # assert raw_values.shape[-1] == self.high.shape[0]
        squashed = ((torch.tanh(raw_values) + 1.0) / 2.0) * (
            self.high - self.low
        ) + self.low
        # transform squashed to Floattensor
        squashed = squashed.type(torch.FloatTensor)

        return torch.clamp(squashed, self.low, self.high).float()

    def forward(
        self,
        state,
        edge_index,
        deterministic=False,
    ):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        edges = torch.tensor(self.edges)
        # Obtain edge features using 'out' for the updated node features
        edges_src = edges[:, 0]  # [E]
        edges_dst = edges[:, 1]  # [E]

        # Obtain features for each node involved in an edge
        edge_features_src = x[:, edges_src, :]  # [#batch, E, #features]
        edge_features_dst = x[:, edges_dst, :]  # [#batch, E, #features]

        # Concatenate features from source and destination nodes
        edge_features = torch.cat([edge_features_src, edge_features_dst], dim=2)

        x = F.leaky_relu(self.lin1(edge_features))
        x = F.leaky_relu(self.lin2(x))
        log_std = self.h_to_sigma(x)

        mu = self.h_to_mu(x)

        mu, sigma = F.softplus(mu), F.softplus(log_std)

        mu = mu.squeeze(-1)
        std = sigma.squeeze(-1)

        pi_distribution = Normal(mu, std)

        if deterministic:
            action = mu
            log_prob = None
            action = self._squash(action)

        else:
            action = pi_distribution.rsample()

            log_prob = pi_distribution.log_prob(action).sum(axis=-1)

        return action, log_prob


#########################################
############## CRITIC ###################
#########################################
class GNNCritic(nn.Module):
    """
    Critic
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=14, edges=None):
        super().__init__()
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(
            2 * in_channels + 1, hidden_size
        )  # Input is from 2 nodes (edge = node1 + node2)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.edges = edges
        self.act_dim = act_dim
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        edges = torch.tensor(self.edges)

        edges_src = edges[:, 0]  # [E]
        edges_dst = edges[:, 1]  # [E]

        # Obtain features for each node involved in an edge
        edge_features_src = x[:, edges_src, :]  # [#batch, E, #features]
        edge_features_dst = x[:, edges_dst, :]  # [#batch, E, #features]

        # Concatenate features from source and destination nodes
        edge_features = torch.cat([edge_features_src, edge_features_dst], dim=2)

        concat = torch.cat([edge_features, action.unsqueeze(-1)], dim=-1)

        x = F.leaky_relu(self.lin1(concat))
        x = F.leaky_relu(self.lin2(x))
        x = torch.sum(x, dim=1)
        x = self.lin3(x).squeeze(-1)

        return x


#########################################
############## SAC AGENT ################
#########################################
class SAC(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        env,
        input_size,
        hidden_size=32,
        alpha=0.2,
        gamma=0.99,
        polyak=0.995,
        batch_size=128,
        p_lr=3e-4,
        q_lr=1e-3,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=-1,
        min_q_weight=1,
        deterministic_backup=False,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
        load_memory=False,
        min_q_version=3,
        clip=200,
        critic_version=4,
    ):
        super(SAC, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.path = None
        self.act_dim = env.nregion

        # SAC parameters
        self.alpha = alpha
        self.polyak = polyak
        self.env = env
        self.BATCH_SIZE = batch_size
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.gamma = gamma
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.min_q_version = min_q_version
        self.clip = clip

        # conservative Q learning parameters
        self.num_random = 10
        self.temp = 1.0
        self.min_q_weight = min_q_weight
        if lagrange_thresh == -1:
            self.with_lagrange = False
        else:
            print("using lagrange")
            self.with_lagrange = True
        self.deterministic_backup = deterministic_backup
        self.step = 0
        self.nodes = env.nregion

        self.replay_buffer = ReplayData(device=device)
        # nnets
        self.act_dim = self.env.nregion
        self.actor = GNNActor(
            self.input_size, self.hidden_size, act_dim=self.act_dim, edges=env.edges
        ).to(self.device)
        print(self.actor)
        self.critic1 = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim, edges=env.edges
        ).to(self.device)
        self.critic2 = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim, edges=env.edges
        ).to(self.device)
        assert self.critic1.parameters() != self.critic2.parameters()
        print(self.critic1)

        self.critic1_target = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim, edges=env.edges
        ).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim, edges=env.edges
        ).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh  # lagrange treshhold
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = torch.optim.Adam(
                self.log_alpha_prime.parameters(),
                lr=self.p_lr,
            )

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(self.act_dim).item()
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=1e-3
            )

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, data, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(data.x, data.edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy()[0]
        return list(a)

    def compute_loss_q(self, data):
        (
            state_batch,
            edge_index,
            next_state_batch,
            edge_index2,
            reward_batch,
            action_batch,
        ) = (
            data.x_s,
            data.edge_index_s,
            data.x_t,
            data.edge_index_t,
            data.reward,
            data.action.reshape(-1, len(self.env.edges)),
        )
        q1 = self.critic1(state_batch, edge_index, action_batch)
        q2 = self.critic2(state_batch, edge_index, action_batch)
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor(next_state_batch, edge_index2)
            q1_pi_targ = self.critic1_target(next_state_batch, edge_index2, a2)
            q2_pi_targ = self.critic2_target(next_state_batch, edge_index2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            backup = reward_batch + self.gamma * (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        return loss_q1, loss_q2

    def compute_loss_pi(self, data):
        state_batch, edge_index = (
            data.x_s,
            data.edge_index_s,
        )

        actions, logp_a = self.actor(state_batch, edge_index)
        q1_1 = self.critic1(state_batch, edge_index, actions)
        q2_a = self.critic2(state_batch, edge_index, actions)
        q_a = torch.min(q1_1, q2_a)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (logp_a + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()

        loss_pi = (self.alpha * logp_a - q_a).mean()

        return loss_pi

    def update(self, data, only_q=False):
        loss_q1, loss_q2 = self.compute_loss_q(data)

        self.optimizers["c1_optimizer"].zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.optimizers["c2_optimizer"].step()

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        if not only_q:
            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.critic1.parameters():
                p.requires_grad = False
            for p in self.critic2.parameters():
                p.requires_grad = False

            # one gradient descent step for policy network
            self.optimizers["a_optimizer"].zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.optimizers["a_optimizer"].step()

            # Unfreeze Q-networks
            for p in self.critic1.parameters():
                p.requires_grad = True
            for p in self.critic2.parameters():
                p.requires_grad = True

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=self.q_lr)

        return optimizers

    def test_agent(self, test_episodes, env, cplexpath, directory, parser):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        for _ in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            obs = env.reset()
            actions = []
            done = False
            while not done:
                obs, paxreward, done, info, _, _ = env.pax_step(
                    CPLEXPATH=cplexpath, PATH="scenario_nyc4_test", directory=directory
                )
                eps_reward += paxreward

                o = parser.parse_obs(obs)

                action_rl = self.select_action(o, deterministic=True)
                rebAction = [max(int(i.item()), 0) for i in action_rl]

                _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                eps_reward += rebreward
                eps_served_demand += info["served_demand"]
                eps_rebalancing_cost += info["rebalancing_cost"]
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)

        return (
            np.mean(episode_reward),
            np.mean(episode_served_demand),
            np.mean(episode_rebalancing_cost),
        )

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["model"].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
