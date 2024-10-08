import argparse
import numpy as np
from src.algos.mpc_fore import solveMCP

from src.envs.supply_chain_env import (
    NetInvMgmtBacklogEnv as env,
)
from s_type import s_type_policy
import pickle

"""
This script is used to collect the dataset (on the lower level)
"""
epsilon = 1e-10

parser = argparse.ArgumentParser(description="SAC-SC")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
)
parser.add_argument(
    "--directory",
    type=str,
    default="saved_files",
    help="defines directory where to save files",
)
parser.add_argument(
    "--collection",
    type=str,
    default="MPC",
    help="Choice between MPC and SPolicy",
)
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
parser.add_argument(
    "--cplexpath",
    type=str,
    default="/opt/opl/bin/x86-64_linux/",
    help="defines directory of the CPLEX installation",
)

args = parser.parse_args()

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


env = env(version=args.version)

state, ep_ret, ep_len = env.reset(), 0, 0
SR, PC, TC, OC, HC, UP, UD = 0, 0, 0, 0, 0, 0, 0
inventory_actions = []
training_rewards = []

order_actions = []
errors = []
demand_per_t = []
episode_reward = []
actions = []
inventory = {
    "inventory_1": [],
    "inventory_2": [],
    "inventory_3": [],
    "inventory_4": [],
    "inventory_5": [],
    "inventory_6": [],
}
demand = []


replay_buffer = ReplayBuffer(
    obs_dim=(4, 17 + env.lt_max),
    act_dim=4,
    size=20000,
    edge_dim=(6, 1),
)


if args.collection == "SPolicy":

    if args.version == 2:
        combination = [46, 3, 9, 13]
        s_policy = s_type_policy(
            factory_s=combination[0],
            warehouses_s=[
                combination[1],
                combination[1],
                combination[2],
                combination[3],
            ],
        )
    elif args.version == 3:
        combination = [79, 2, 6, 10]
        s_policy = s_type_policy(
            factory_s=combination[0],
            warehouses_s=[
                combination[1],
                combination[1],
                combination[1],
                combination[1],
                combination[2],
                combination[2],
                combination[2],
                combination[3],
                combination[3],
                combination[3],
            ],
        )
rew = 0
rewards = []
VC = []
for t in range(100):
    if args.collection == "MPC":
        flowMPC, productionMPC, results_MPC = solveMCP(
            env,
            CPLEXPATH=args.cplexpath,
            res_path="scim",
            directory=args.directory,
            T=10,
        )

        flow = [int(flowMPC[(i, j)]) for i, j in env.reorder_links]
        prod = [int(productionMPC[i]) for i in env.factory]

        action = flow + prod
        flowMPC = {k: int(v) for k, v in flowMPC.items() if k in env.reorder_links}

        productionMPC = {
            k: int(v) for k, v in productionMPC.items() if k in env.factory
        }

        next_state, reward, done, info = env.step(
            prod_action=productionMPC, distr_action=flowMPC
        )

        action = [int(i) for i in action]
        replay_buffer.store(
            state.x, np.asarray(action), reward, next_state.x, state.edge_attr, done
        )

        state = next_state
        rew += reward

    elif args.collection == "SPolicy":
        prod, ship = s_policy.select_action(env)
        next_state, reward, done, info = env.step(prod_action=prod, distr_action=ship)

        action = [ship[k] for k in env.reorder_links]
        action += [prod[i] for i in env.factory]

        replay_buffer.store(
            state.x, action, reward, next_state.x, state.edge_attr, done
        )

        state = next_state
        rew += reward

    if done:
        state, ep_ret, ep_len = env.reset(), 0, 0
        print("Steps", t)
        print("reward", rew)
        rewards.append(rew)
        rew = 0


print(replay_buffer.size)
w = open(f"/datasets/{args.checkpoint_path}.pkl", "wb")
pickle.dump(replay_buffer, w)
w.close()
