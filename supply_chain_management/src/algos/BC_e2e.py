import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch_geometric.data import Data, Batch

import random
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

"""
BC E2E
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
############## ACTOR ####################
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
        low=None,
        high=None,
        edges=None,
        nnodes=4,
    ):
        super(Actor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_factories = num_factories
        self.node_size = node_size

        self.conv1 = EdgeConv(node_size, edge_size, hidden_dim)

        self.h_to_mu = nn.Linear((node_size + hidden_dim) * 2, out_channels)
        self.h_to_sigma = nn.Linear((node_size + hidden_dim) * 2, out_channels)

        self.h_to_mu_2 = nn.Linear(node_size + hidden_dim, out_channels)
        self.h_to_sigma_2 = nn.Linear(node_size + hidden_dim, out_channels)

        self.low = low.float()
        self.high = high.float()
        self.edges = edges
        self.num_factories = num_factories
        self.nnodes = nnodes

    def forward(self, x, edge_index, edge_attr, deterministic=False, return_D=False):
        x_pp = self.conv1(x, edge_index, edge_attr)
        x_pp = torch.cat([x, x_pp], dim=1)
        x_pp = x_pp.reshape(-1, self.nnodes, self.node_size + self.hidden_dim)

        edges = torch.tensor(self.edges)
        edges_src = edges[:, 0]  # [E]
        edges_dst = edges[:, 1]  # [E]

        edge_features_src = x_pp[:, edges_src, :]  # [#batch, E, #features]
        edge_features_dst = x_pp[:, edges_dst, :]  # [#batch, E, #features]

        edge_features = torch.cat([edge_features_src, edge_features_dst], dim=2)

        alpha = self.h_to_mu(edge_features)

        alpha = F.softplus(alpha + 1e-20).squeeze(-1)

        beta = F.softplus(self.h_to_sigma(edge_features) + 1e-20).squeeze(-1)

        alpha2 = self.h_to_mu_2(x_pp[:, -self.num_factories :, :]).squeeze(-1)
        alpha2 = F.softplus(alpha2 + 1e-20)
        beta2 = self.h_to_sigma_2(x_pp[:, -self.num_factories :, :]).squeeze(-1)
        beta2 = F.softplus(beta2 + 1e-20)

        if deterministic:
            alpha += 1e-20
            beta += 1e-20

            dis_action = alpha / (alpha + beta)

            dis_action = dis_action * self.high[: -self.num_factories]

            alpha2 += 1e-20
            beta2 += 1e-20

            order_act = alpha2 / (alpha2 + beta2)
            order_act = order_act * self.high[-self.num_factories :]

            act = torch.cat([dis_action, order_act], dim=-1)

        else:
            if return_D:
                m1 = Beta(alpha + 1e-20, beta + 1e-20)
                m2 = Beta(alpha2 + 1e-20, beta2 + 1e-20)
                return m1, m2
            else:
                dis_action = m1.sample()
                dis_action = dis_action * self.high[: -self.num_factories]
                order_act = m2.sample()
                order_act = order_act * self.high[-self.num_factories :]
                act = torch.cat([dis_action, order_act], dim=-1)

        return act


class BC(nn.Module):
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
        super(BC, self).__init__()
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.path = None

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

        self.low = torch.tensor(env.action_space.low).to(device)
        self.high = torch.tensor(env.action_space.high).to(device)
        self.act_dim = env.action_space.shape[0]
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
            edges=env.edge_list,
            nnodes=len(env.nodes),
        ).to(self.device)
        print(self.actor)

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def preprocess_act(self, inventory_act, order_act):

        order_act = torch.cat(
            (
                torch.zeros(order_act.shape[0], len(self.env.distrib)).to(self.device),
                order_act,
            ),
            dim=1,
        )
        return torch.cat((inventory_act.unsqueeze(-1), order_act.unsqueeze(-1)), dim=-1)

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, data, deterministic=False):
        with torch.no_grad():
            action = self.actor(data.x, data.edge_index, data.edge_attr, deterministic)
        return action.detach().cpu().squeeze().numpy()

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
            data.action.reshape(-1, self.act_dim),
            data.done.float(),
        )

        D, G = self.actor(state_batch, edge_index, edge_attr, return_D=True)

        order_act = action_batch[:, -self.num_factories :]
        inventory_act = action_batch[:, : -self.num_factories]

        order_act = order_act / self.high[-self.num_factories :]
        inventory_act = inventory_act / self.high[: -self.num_factories]

        order_act = torch.clamp(order_act, 0.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER)
        inventory_act = torch.clamp(
            inventory_act, 0.0 + SMALL_NUMBER, 1.0 - SMALL_NUMBER
        )

        order_log_prob = G.log_prob(order_act).sum(axis=-1)

        Dirichlet_log_prob = D.log_prob(inventory_act).sum(axis=-1)

        log_prob = order_log_prob + Dirichlet_log_prob

        loss_pi = -(log_prob).mean()
        # one gradient descent step for policy network
        self.optimizers["a_optimizer"].zero_grad()
        loss_pi.backward(retain_graph=False)

        self.optimizers["a_optimizer"].step()

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)

        return optimizers

    def test_agent(self, test_episodes, env, cplexpath, directory):
        epochs = range(test_episodes)
        episode_reward = []
        e = []
        for _ in epochs:
            eps_reward = 0
            state = env.reset()
            actions = []
            done = False
            errors = []
            VC, OO = 0, 0
            while not done:
                action_output = self.select_action(
                    state.to(self.device), deterministic=True
                )
                flow = {
                    env.reorder_links[i]: max(0, action_output[i])
                    for i in range(len(env.reorder_links))
                }
                order_act = action_output[-len(env.factory) :]
                prod_action = {
                    i: int(order_act[i - len(env.distrib) - 1]) for i in env.factory
                }
                next_state, reward, done, info = env.step(
                    prod_action=prod_action, distr_action=flow
                )

                state = next_state

                eps_reward += reward

                VC += info["violated_capacity"]
                OO += info["over_order"]

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

        self.load_state_dict(model_dict)
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
