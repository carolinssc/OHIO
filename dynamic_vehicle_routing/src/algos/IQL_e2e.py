import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GCNConv
from collections import namedtuple
import json
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, Batch

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])
args = namedtuple("args", ("render", "gamma", "log_interval"))
args.render = True
args.gamma = 0.97
args.log_interval = 10

"""
IQL E2E implementation for the vehicle routing problem
"""


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant


class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T=10, json_file=None, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs, device):
        x = (
            torch.cat(
                (
                    torch.tensor(
                        [obs[0][n][self.env.time + 1] * self.s for n in self.env.region]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    torch.tensor(
                        [
                            [
                                (obs[0][n][self.env.time + 1] + self.env.dacc[n][t])
                                * self.s
                                for n in self.env.region
                            ]
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nregion)
                    .float(),
                    torch.tensor(
                        [
                            [
                                sum(
                                    [
                                        (self.env.scenario.demand_input[i, j][t])
                                        * (self.env.price[i, j][t])
                                        * self.s
                                        for j in self.env.region
                                    ]
                                )
                                for i in self.env.region
                            ]
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nregion)
                    .float(),
                ),
                dim=1,
            )
            .squeeze(0)
            .view(1 + self.T + self.T, self.env.nregion)
            .T
        )
        if self.json_file is not None:
            edge_index = torch.vstack(
                (
                    torch.tensor(
                        [edge["i"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                    torch.tensor(
                        [edge["j"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                )
            ).long()
        else:
            edge_index = torch.cat(
                (
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                ),
                dim=0,
            ).long()
        data = Data(x, edge_index).to(device)
        return data


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
        high=120,
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

    def forward(self, state, edge_index, deterministic=False, return_D=False):
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

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        mu = self.h_to_mu(x)

        mu = F.softplus(mu + 1e-10)

        mu = mu.squeeze(-1)
        std = std.squeeze(-1)

        pi_distribution = Normal(mu, std)

        if deterministic:
            action = mu
            return action

        else:
            if return_D:
                return pi_distribution
            else:
                action = pi_distribution.sample()

                return action


#########################################
############## CRITIC ###################
#########################################
class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(a_t, s_t).
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
        # Obtain edge features using 'out' for the updated node features
        # edge_features = torch.cat([x[edges[:, 0]], x[edges[:, 1]]], dim=1)

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


class VF(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """

    def __init__(self, in_channels, hidden_size=64, act_dim=16):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = torch.sum(x, dim=1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x).squeeze(-1)
        return x


class IQL(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        env,
        input_size,
        hidden_size=32,
        gamma=0.97,
        polyak=0.995,
        batch_size=128,
        p_lr=3e-4,
        q_lr=1e-3,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
        quantile=0.5,
        clip=200,
        json_file=None,
        temperature=1.0,
        clip_score=100,
    ):
        super(IQL, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.path = None
        self.first = True

        # IQL parameters
        self.polyak = polyak
        self.env = env
        self.BATCH_SIZE = batch_size
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.gamma = gamma
        self.temperature = temperature
        self.clip_score = clip_score

        self.act_dim = self.env.nregion
        # conservative Q learning parameters
        self.num_random = 10
        self.clip = clip
        self.quantile = quantile

        self.policy_update_period = (1,)
        self.q_update_period = (1,)

        # nnets
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

        self.vf = VF(in_channels=self.input_size, act_dim=self.act_dim).to(self.device)

        self.critic1_target = (
            copy.deepcopy(self.critic1).requires_grad_(False).to(device)
        )
        self.critic2_target = (
            copy.deepcopy(self.critic2).requires_grad_(False).to(device)
        )

        self.obs_parser = GNNParser(self.env, json_file=json_file, T=6)

        self.optimizers = self.configure_optimizers()
        self.actor_lr_schedule = CosineAnnealingLR(
            self.optimizers["a_optimizer"], 10000
        )
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def parse_obs(self, obs, device):
        state = self.obs_parser.parse_obs(obs, device)
        return state

    def select_action(self, state, edge_index, deterministic=False):
        with torch.no_grad():
            a = self.actor(state, edge_index, deterministic=deterministic)
            a = a.detach().cpu().numpy()[0]
        return list(a)

    def bc_update(self, data):
        """
        Behavior cloning update
        """
        state_batch, edge_index, action_batch = (
            data.x_s,
            data.edge_index_s,
            data.action.reshape(-1, len(self.env.edges)),
        )

        m = self.actor(state_batch, edge_index, return_D=True)

        policy_logpp = m.log_prob(action_batch).sum(dim=-1)

        loss_pi = (-policy_logpp).mean()

        self.optimizers["a_optimizer"].zero_grad()
        loss_pi.backward()
        self.optimizers["a_optimizer"].step()
        self.actor_lr_schedule.step()

        return

    def update(self, data, only_q=False, only_p=False):
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
        """
        Q LOSS
        """

        q1_pred = self.critic1(state_batch, edge_index, action_batch)
        q2_pred = self.critic2(state_batch, edge_index, action_batch)

        target_vf_pred = self.vf(next_state_batch, edge_index2).detach()
        with torch.no_grad():
            q_target = reward_batch + self.gamma * target_vf_pred

        q_target = q_target.detach()

        loss_q1 = self.qf_criterion(q1_pred, q_target)
        loss_q2 = self.qf_criterion(q2_pred, q_target)

        """
        VF Loss
        """
        q_pred = torch.min(
            self.critic1_target(state_batch, edge_index, action_batch),
            self.critic2_target(state_batch, edge_index, action_batch),
        ).detach()

        vf_pred = self.vf(state_batch, edge_index)

        adv = q_pred - vf_pred

        weight = torch.where(adv > 0, self.quantile, (1 - self.quantile))
        vf_loss = weight * (adv**2)
        vf_loss = vf_loss.mean()

        if not only_q:
            """
            Policy Loss
            """

            m = self.actor(state_batch, edge_index, return_D=True)

            policy_logpp = m.log_prob(action_batch).sum(dim=-1)

            exp_adv = torch.exp(self.temperature * adv.detach()).clamp(
                max=self.clip_score
            )

            loss_pi = -(policy_logpp * exp_adv).mean()

        if not only_p:
            self.optimizers["c1_optimizer"].zero_grad()
            loss_q1.backward()
            self.optimizers["c1_optimizer"].step()

            self.optimizers["c2_optimizer"].zero_grad()
            loss_q2.backward()
            self.optimizers["c2_optimizer"].step()

            self.optimizers["v_optimizer"].zero_grad()
            vf_loss.backward()
            self.optimizers["v_optimizer"].step()

        if not only_q:
            self.optimizers["a_optimizer"].zero_grad()
            loss_pi.backward()
            self.optimizers["a_optimizer"].step()

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

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())
        v_params = list(self.vf.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=self.q_lr)

        optimizers["v_optimizer"] = torch.optim.Adam(v_params, lr=self.q_lr)
        return optimizers

    def test_agent(self, test_episodes, env, cplexpath, directory):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        for _ in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            obs = env.reset()

            done = False
            while not done:
                obs, paxreward, done, info, _, _ = env.pax_step(
                    CPLEXPATH=cplexpath, PATH="scenario_nyc4_test", directory=directory
                )
                eps_reward += paxreward

                o = self.parse_obs(obs, self.device)

                rebAction = self.select_action(o.x, o.edge_index, deterministic=True)

                rebAction = [max(0, int(i)) for i in rebAction]

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
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model"])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
