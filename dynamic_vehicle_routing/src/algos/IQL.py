import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from collections import namedtuple
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
import json


"""
IQL bi-level policy implementation for the vehicle routing problem
"""

args = namedtuple("args", ("render", "gamma", "log_interval"))
args.render = True
args.gamma = 0.97
args.log_interval = 10


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
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1) + 1e-20

        return concentration

    def logprob(self, state, edge_index, actions):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1) + 1e-10

        m = Dirichlet(concentration)

        log_prob = m.log_prob(actions)
        return log_prob, concentration


#########################################
############## CRITIC ###################
#########################################
class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator Q(a_t, s_t).
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=16):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + 1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B, 1)
        return x


#########################################
########## VALUE FUNCTION ###############
#########################################
class VF(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """

    def __init__(self, in_channels, act_dim=16):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)

    def forward(self, state, edge_index):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = torch.sum(x, dim=1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x).squeeze(-1)
        return x


#########################################
############## A2C AGENT ################
#########################################


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

        # SAC parameters
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
            self.input_size, self.hidden_size, act_dim=self.act_dim
        ).to(self.device)
        print(self.actor)
        self.critic1 = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim
        ).to(self.device)
        self.critic2 = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim
        ).to(self.device)
        assert self.critic1.parameters() != self.critic2.parameters()
        print(self.critic1)

        self.vf = VF(in_channels=self.input_size, act_dim=self.act_dim).to(self.device)

        self.critic1_target = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim
        ).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim
        ).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.obs_parser = GNNParser(self.env, json_file=json_file, T=6)

        self.optimizers = self.configure_optimizers()

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
            concentration = self.actor(state, edge_index)
        if deterministic:
            a = (concentration) / (concentration.sum() + 1e-20)
            a = a.squeeze(-1)
            a = a.detach().cpu().numpy()[0]
        else:
            m = Dirichlet(concentration)
            a = m.sample()
            a = a.squeeze(-1)
            a = a.detach().cpu().numpy()[0]
        return list(a)

    def bc_update(self, data):
        state_batch, edge_index, action_batch = (
            data.x_s,
            data.edge_index_s,
            data.action.reshape(-1, self.env.nregion),
        )

        concentration = self.actor(state_batch, edge_index)

        m = torch.distributions.dirichlet.Dirichlet(concentration)

        policy_logpp = m.log_prob(action_batch)

        loss_pi = (-policy_logpp).mean()

        self.optimizers["a_optimizer"].zero_grad()
        loss_pi.backward()
        self.optimizers["a_optimizer"].step()

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
            data.action.reshape(-1, self.env.nregion),
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

        vf_err = q_pred - vf_pred

        weight = torch.where(vf_err > 0, self.quantile, (1 - self.quantile))
        vf_loss = weight * (vf_err**2)
        vf_loss = vf_loss.mean()

        if not only_q:
            """
            Policy Loss
            """

            concentration = self.actor(state_batch, edge_index)

            m = torch.distributions.dirichlet.Dirichlet(concentration + 1e-8)

            policy_logpp = m.log_prob(action_batch)

            adv = q_pred - vf_pred

            exp_adv = torch.exp(adv * self.temperature)

            exp_adv = torch.clamp(exp_adv, max=self.clip_score)

            weights = exp_adv.detach()

            loss_pi = -(policy_logpp * weights).mean()

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
            actions = []
            done = False
            while not done:
                obs, paxreward, done, info, _, _ = env.pax_step(
                    CPLEXPATH=cplexpath, PATH="scenario_nyc4_test", directory=directory
                )
                eps_reward += paxreward

                o = self.parse_obs(obs, self.device)

                action_rl = self.select_action(o.x, o.edge_index, deterministic=True)
                actions.append(action_rl)

                desiredAcc = {
                    env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }

                rebAction = solveRebFlow(
                    env, "scenario_nyc4_test", desiredAcc, cplexpath, directory
                )

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
