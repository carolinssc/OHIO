import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Beta
from torch_geometric.data import Data, Batch
import random
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from lcp_solver_cap import solveLCP

"""
IQL for bi-level polic
"""

SMALL_NUMBER = 1e-6


class PairData(Data):
    def __init__(
        self,
        edge_index_s=None,
        edge_attr_s=None,
        x_s=None,
        reward=None,
        action=None,
        done=None,
        edge_index_t=None,
        edge_attr_t=None,
        x_t=None,
    ):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.edge_attr_s = edge_attr_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.done = done
        self.edge_index_t = edge_index_t
        self.edge_attr_t = edge_attr_t
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

    def store(self, data1, action, reward, data2, done):
        self.data_list.append(
            PairData(
                data1.edge_index,
                data1.edge_attr,
                data1.x,
                torch.as_tensor(reward),
                torch.as_tensor(action),
                torch.as_tensor(done),
                data2.edge_index,
                data2.edge_attr,
                data2.x,
            )
        )
        self.rewards.append(reward)

    def size(self):
        return len(self.data_list)

    def sample_batch(self, batch_size=32, norm=False, return_list=False):
        data = random.sample(self.data_list, batch_size)
        if norm:
            mean = np.mean(self.rewards)
            std = np.std(self.rewards)
            batch = Batch.from_data_list(data, follow_batch=["x_s", "x_t"])
            batch.reward = (batch.reward - mean) / (std + 1e-16)
            return batch.to(self.device)
        else:
            if return_list:
                return data
            else:
                return Batch.from_data_list(data, follow_batch=["x_s", "x_t"]).to(
                    self.device
                )


class EdgeConv(MessagePassing):
    def __init__(self, node_size=4, edge_size=0, out_channels=4):
        super().__init__(aggr="add", flow="target_to_source")  #  "Max" aggregation.
        self.mlp = Seq(
            Linear(2 * node_size + edge_size, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        tmp = torch.cat(
            [x_i, x_j, edge_attr], dim=1
        )  # tmp has shape [E, 2 * in_channels]

        return self.mlp(tmp)


#########################################
############## ACTOR ###################
#########################################
class Actor(torch.nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(
        self,
        node_size=4,
        edge_size=0,
        hidden_dim=32,
        out_channels=1,
        num_factories=3,
        mode="gaussian",
        low=None,
        high=None,
        nnodes=4,
    ):
        super(Actor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_factories = num_factories
        self.node_size = node_size
        self.mode = mode

        self.conv1 = EdgeConv(node_size, edge_size, hidden_dim)
        self.h_to_concentration = nn.Linear(node_size + hidden_dim, out_channels)

        self.h_to_mu = nn.Linear(node_size + hidden_dim, out_channels)
        self.h_to_sigma = nn.Linear(node_size + hidden_dim, out_channels)
        self.low = low
        self.high = high
        self.nnodes = nnodes

    def forward(self, x, edge_index, edge_attr, deterministic=False, return_D=False):
        x_pp = self.conv1(x, edge_index, edge_attr)
        x_pp = torch.cat([x, x_pp], dim=1)
        x_pp = x_pp.reshape(-1, self.nnodes, self.node_size + self.hidden_dim)

        alpha = self.h_to_mu(x_pp[:, -self.num_factories :]).reshape(-1)
        alpha = F.softplus(alpha + 1e-20)
        beta = self.h_to_sigma(x_pp[:, -self.num_factories :]).reshape(-1)
        beta = F.softplus(beta + 1e-20)

        concentration = F.softplus(self.h_to_concentration(x_pp) + 1e-10)

        concentration = concentration.squeeze(-1)

        if deterministic:
            inventory_act = (concentration) / (concentration.sum() + 1e-20)

            alpha += 1e-20
            beta += 1
            order_act = alpha / (alpha + beta)
            order_act = order_act * self.high.float()
            return inventory_act, order_act

        else:

            m = Dirichlet(concentration + 1e-20)
            inventory_act = m.sample()

            m1 = Beta(alpha + 1e-20, beta + 1e-20)
            order_act = m1.sample()
            order_act = order_act * self.high.float()

            if return_D:
                return (
                    m,
                    m1,
                )
            else:
                return inventory_act, order_act


#########################################
############## CRITIC ###################
#########################################
class Critic(torch.nn.Module):
    def __init__(
        self,
        node_size=4,
        edge_size=3,
        hidden_dim=32,
        out_channels=1,
        low=None,
        high=None,
        nnodes=4,
    ):
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_size = node_size
        self.conv1 = EdgeConv(node_size, edge_size, hidden_dim)
        self.lin1 = nn.Linear(node_size + hidden_dim + 2, hidden_dim)

        self.low = low
        self.high = high
        self.g_to_v = nn.Linear(hidden_dim, out_channels)
        self.nnodes = nnodes


    def forward(self, x, edge_index, edge_attr, action):
        # demand = x[:, -6:]
        # demand = self.lind(demand)
        # x = torch.cat([x[:, :-6], demand], dim=1)

        x_pp = self.conv1(x, edge_index, edge_attr)
        x_pp = torch.cat([x, x_pp], dim=1)

        x_pp = x_pp.reshape(-1, self.nnodes, self.node_size + self.hidden_dim)

        concat = torch.cat([x_pp, action], dim=-1)  # (B,N,22)
        concat = F.relu(self.lin1(concat))
        v = self.g_to_v(concat)
        v = torch.sum(v, dim=1)
        return v.squeeze(-1)


#########################################
######### VALUE FUNCTION ################
#########################################
class VF(torch.nn.Module):
    def __init__(
        self,
        node_size=4,
        edge_size=3,
        hidden_dim=32,
        out_channels=1,
        nnodes=4,
    ):
        super(VF, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_size = node_size
        self.conv1 = EdgeConv(node_size, edge_size, hidden_dim)
        self.lin1 = nn.Linear(node_size + hidden_dim, hidden_dim)
        self.g_to_v = nn.Linear(hidden_dim, out_channels)
        self.nnodes = nnodes

    def forward(self, x, edge_index, edge_attr):
        x_pp = self.conv1(x, edge_index, edge_attr)
        x_pp = torch.cat([x, x_pp], dim=1)
        x_pp = x_pp.reshape(-1, self.nnodes, self.node_size + self.hidden_dim)
        v = torch.sum(x_pp, dim=1)
        v = F.relu(self.lin1(v))
        v = self.g_to_v(v)
        return v.squeeze(-1)


class IQL(nn.Module):
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
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
        clip=10,
        quantile=0.5,
        temperature=1.0,
        clip_score=100,
        edge_size=2,
    ):
        super(IQL, self).__init__()
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.path = None
        self.nnodes = len(env.nodes)
        self.env = env

        # SAC parameters
        self.alpha = alpha
        self.polyak = polyak
        self.batch_size = batch_size
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.gamma = gamma

        self.num_random = 10
        self.temp = 1.0
        self.clip = clip
        self.clip_score = clip_score
        self.quantile = quantile
        self.temperature = temperature

        self.low = torch.tensor([0 for _ in env.factory]).to(self.device)
        self.high = torch.tensor([env.graph.nodes[(i)]["C"] for i in env.factory]).to(
            self.device
        )
        self.step = 0

        self.replay_buffer = ReplayData(device=device)
        self.low
        self.num_factories = len(env.factory)
        # nnets
        self.actor = Actor(
            node_size=self.input_size,
            hidden_dim=self.hidden_size,
            out_channels=1,
            num_factories=self.num_factories,
            low=self.low,
            high=self.high,
            edge_size=edge_size,
            nnodes=self.nnodes,
        ).to(self.device)
        print(self.actor)
        self.critic1 = Critic(
            node_size=self.input_size,
            hidden_dim=self.hidden_size,
            out_channels=1,
            low=self.low,
            high=self.high,
            edge_size=edge_size,
            nnodes=self.nnodes,
        ).to(self.device)
        self.critic2 = Critic(
            node_size=self.input_size,
            hidden_dim=self.hidden_size,
            out_channels=1,
            low=self.low,
            high=self.high,
            edge_size=edge_size,
            nnodes=self.nnodes,
        ).to(self.device)
        assert self.critic1.parameters() != self.critic2.parameters()
        print(self.critic1)

        self.critic1_target = Critic(
            node_size=self.input_size,
            hidden_dim=self.hidden_size,
            out_channels=1,
            low=self.low,
            high=self.high,
            edge_size=edge_size,
            nnodes=self.nnodes,
        ).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Critic(
            node_size=self.input_size,
            hidden_dim=self.hidden_size,
            out_channels=1,
            low=self.low,
            high=self.high,
            edge_size=edge_size,
            nnodes=self.nnodes,
        ).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.vf = VF(
            node_size=self.input_size,
            edge_size=edge_size,
            hidden_dim=self.hidden_size,
            nnodes=self.nnodes,
        ).to(self.device)
        print(self.vf)
        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.optimizers = self.configure_optimizers()
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def preprocess_act(self, inventory_act, order_act):
        order_act = torch.cat(
            (
                torch.zeros(order_act.shape[0], len(self.env.distrib)).to(self.device),
                order_act.unsqueeze(-1),
            ),
            dim=1,
        )
        return torch.cat((inventory_act.unsqueeze(-1), order_act.unsqueeze(-1)), dim=-1)

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, data, deterministic=False):
        with torch.no_grad():
            inventory_act, order_act = self.actor(
                data.x, data.edge_index, data.edge_attr, deterministic
            )

        combined_action = self.preprocess_act(
            inventory_act.detach(), order_act.detach()
        )
        inventory_act = inventory_act.detach().squeeze().cpu().numpy()
        order_act = order_act.detach().cpu().squeeze().numpy()
        return inventory_act, order_act, combined_action

    def update(self, data, only_q=False):
        (
            state_batch,
            edge_index,
            edge_attr,
            next_state_batch,
            edge_index2,
            edge_attr2,
            reward_batch,
            action_batch,
            done_batch,
        ) = (
            data.x_s,
            data.edge_index_s,
            data.edge_attr_s,
            data.x_t,
            data.edge_index_t,
            data.edge_attr_t,
            data.reward,
            data.action.reshape(-1, self.nnodes, 2),
            data.done.float(),
        )

        q1_pred = self.critic1(state_batch, edge_index, edge_attr, action_batch)
        q2_pred = self.critic2(state_batch, edge_index, edge_attr, action_batch)

        target_vf_pred = self.vf(next_state_batch, edge_index2, edge_attr2).detach()

        with torch.no_grad():
            q_target = reward_batch + (1 - done_batch) * self.gamma * target_vf_pred

        q_target = q_target.detach()

        loss_q1 = F.mse_loss(q1_pred, q_target)
        loss_q2 = F.mse_loss(q2_pred, q_target)

        q1_pi_targ = self.critic1_target(
            next_state_batch, edge_index2, edge_attr2, action_batch
        )
        q2_pi_targ = self.critic2_target(
            next_state_batch, edge_index2, edge_attr2, action_batch
        )
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).detach()
        vf_pred = self.vf(state_batch, edge_index, edge_attr)

        vf_err = q_pi_targ - vf_pred

        weight = torch.where(vf_err > 0, self.quantile, (1 - self.quantile))
        vf_loss = weight * (vf_err**2)
        vf_loss = vf_loss.mean()
        if not only_q:
            D, G = self.actor(state_batch, edge_index, edge_attr, return_D=True)

            order_act = action_batch[:, :, 1][:, -self.num_factories :].squeeze(-1)

            inventory_act = action_batch[:, :, 0]

            order_act = order_act / self.high.float()

            order_act = torch.clamp(order_act, 0.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER)

            order_log_prob = G.log_prob(order_act)

            Dirichlet_log_prob = D.log_prob(inventory_act.squeeze(-1))

            log_prob = order_log_prob + Dirichlet_log_prob

            adv = q_pi_targ - vf_pred

            exp_adv = torch.exp(adv * self.temperature)

            exp_adv = torch.clamp(exp_adv, max=self.clip_score)

            weights = exp_adv.detach()

            loss_pi = -(log_prob * weights).mean()

        self.optimizers["c1_optimizer"].zero_grad()
        loss_q1.backward()
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward()
        self.optimizers["c2_optimizer"].step()

        self.optimizers["v_optimizer"].zero_grad()
        vf_loss.backward()
        self.optimizers["v_optimizer"].step()

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

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # one gradient descent step for policy network
        if not only_q:
            self.optimizers["a_optimizer"].zero_grad()
            loss_pi.backward(retain_graph=False)
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
        v_params = list(self.vf.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=self.q_lr)
        optimizers["v_optimizer"] = torch.optim.Adam(v_params, lr=self.q_lr)

        return optimizers

    def test_agent(self, test_episodes, env, cplexpath, directory):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        e = []
        for _ in epochs:
            eps_reward = 0
            state = env.reset()
            done = False
            errors = []
            while not done:
                inventory_act, order_act, _ = self.select_action(
                    state.to(self.device), deterministic=True
                )
                main_edges_act, prod, error = solveLCP(
                    env,
                    desiredDistrib=inventory_act,
                    desiredProd=order_act,
                    CPLEXPATH=cplexpath,
                    directory=directory,
                )

                prod_action = {i: int(prod[i]) for i in env.factory}
                next_state, reward, done, info = env.step(
                    prod_action=prod_action, distr_action=main_edges_act
                )

                state = next_state

                eps_reward += reward

            episode_reward.append(eps_reward)
            e.append(np.abs(errors).sum())

        return (
            np.mean(episode_reward),
            np.mean(e),
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

        # self.load_state_dict(checkpoint["model"])
        self.load_state_dict(model_dict)
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
