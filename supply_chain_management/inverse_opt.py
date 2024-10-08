import argparse
import numpy as np
import torch

from src.envs.supply_chain_env import (
    NetInvMgmtBacklogEnv as env,
)
import pickle
import numpy as np
import torch
from gurobipy import Model, quicksum, Env

"""
This script is used to create the training dataset for the higher-level policy given a dataset collected on the lower level 
"""


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, edge_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.edge_attr_buff = np.zeros(combined_shape(size, edge_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, edge_attr, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.edge_attr_buff[self.ptr] = edge_attr
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_all(self, idxs):
        print(f"sample {idxs} samples from {len(self.obs_buf)} Dataset")
        if idxs > self.ptr:
            print("idxs > self.ptr")
            print(self.ptr)
        batch = dict(
            obs=self.obs_buf[:idxs],
            obs2=self.obs2_buf[:idxs],
            act=self.act_buf[:idxs],
            rew=self.rew_buf[:idxs],
            edge_attr=self.edge_attr_buff[:idxs],
            done=self.done_buf[:idxs],
        )
        return {k: torch.as_tensor(v) for k, v in batch.items()}


def flow_to_d(flows, acc):

    desiredAcc = acc.copy()
    for i, j in flows.keys():
        desiredAcc[i] -= flows[i, j]
        desiredAcc[j] += flows[i, j]
    return desiredAcc


def inverse_opt(env, observed_flow, inventory):
    gb_env = Env(empty=True)
    gb_env.setParam("OutputFlag", 0)
    gb_env.start()
    # Initialize the model
    model = Model("NetworkFlow", env=gb_env)

    nodes = env.nodes  # List of nodes

    edges = env.reorder_links
    # Add variables
    desiredInv = model.addVars(nodes, name="desiredInv")

    # Set the objective - dummy objective if you are just interested in feasibility
    model.setObjective(0)

    # Add constraints
    for i in nodes:
        inflow = quicksum(observed_flow[k, j] for k, j in edges if j == i)
        outflow = quicksum(observed_flow[k, j] for k, j in edges if k == i)

        model.addConstr(
            inflow - outflow + inventory[i] == desiredInv[i],
            f"node_balance_{i}",
        )
        model.addConstr(outflow <= inventory[i], f"inventory_{i}")

    model.optimize()

    d = {}

    for i in nodes:
        try:
            d[i] = desiredInv[i].x
        except:
            print("infeasible")
            print(inventory)
            print(observed_flow)
    return d


def flow_to_dist(data, replay_buffer=None):
    """
    This function takes the flows and the current inventory levels and returns the desired next state that leads to the observed flows.
    """
    distributions = []
    zero_counter = 0
    for j in range(len(data["act"])):

        state = data["obs"][j]
        action = data["act"][j]

        flows = {}
        for (i, k), flow in zip(reorder_links, action):
            flows[i, k] = flow.item()
        acc = {}
        for i in env.distrib:
            acc[i] = 0
        for i in env.factory:
            acc[i] = state[:, 4][i - 1].item() + state[:, 3][i - 1].item()

        desiredAcc = flow_to_d(flows, acc)
        desiredInv = inverse_opt(env, flows, acc)

        for i in desiredInv.keys():
            # print(desiredInv[i], desiredAcc[i])
            assert (
                desiredInv[i] == desiredAcc[i]
            ), f"desiredInv is not equal to desiredAcc, {desiredInv}, {desiredAcc}"

        for i in desiredInv.keys():
            if desiredInv[i] < 0:
                assert desiredInv[i] >= 0, f"desiredInv is not positive, {desiredInv}"

        if sum(acc.values()) == 0:
            zero_counter += 1
            a = [1e-16] * len(env.distrib) + [1]
            a = np.array(a)
            a /= a.sum()
        else:
            inventoray_act_cleaned = []
            for i in range(1, len(env.nnodes) + 1):
                inventoray_act_cleaned.append(desiredInv[i] / sum(desiredInv.values()))

            a = np.array(inventoray_act_cleaned)
            a[a == 0] = 1e-16
            a /= a.sum()
            assert np.abs(a.sum() - 1) < 1e-8, f"a does not sum up to one_{a.sum()}"

        distributions.append(a)

        if replay_buffer != None:
            inventory_act_cleaned = torch.tensor(a)
            order_act = action[-1]
            order_act = torch.cat(
                (
                    torch.zeros(len(env.distrib)),
                    torch.tensor(order_act).unsqueeze(-1),
                ),
                dim=0,
            ).unsqueeze(-1)

            combined_act_cleaned = torch.cat(
                (inventory_act_cleaned.unsqueeze(-1), order_act), dim=-1
            ).type(torch.float32)

            replay_buffer.store(
                data["obs"][j],
                np.asarray(combined_act_cleaned),
                data["rew"][j],
                data["obs2"][j],
                data["edge_attr"][j],
                data["done"][j],
            )

    return distributions, zero_counter, replay_buffer


epsilon = 1e-10

parser = argparse.ArgumentParser(description="Inverse_opt")


parser.add_argument(
    "--version",
    type=int,
    default=2,
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="MPC",
)


args = parser.parse_args()
w = open(f"/datasets/{args.checkpoint_path}.pkl", "rb")
replay_buffer = pickle.load(w)

data = replay_buffer.sample_all(20000)


env = env(version=args.version)

reorder_links = env.reorder_links

replay_buffer_distr = ReplayBuffer(
    obs_dim=(len(env.nnodes), 18),
    act_dim=(len(env.nnodes), 2),
    size=20000,
    edge_dim=(len(env.edge_list) * 2, 1),
)

mpc_distributions, zero_counter, replay_buffer_distr = flow_to_dist(
    data, replay_buffer=replay_buffer_distr
)

w = open(f"/datasets/{args.checkpoint_path}_distr.pkl", "wb")
pickle.dump(replay_buffer_distr, w)
w.close()
