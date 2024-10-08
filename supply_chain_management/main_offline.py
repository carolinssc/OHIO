import argparse
from tqdm import trange
import numpy as np
import torch
from src.algos.mpc_fore import solveMCP
from src.algos.sac import SAC
from src.algos.sac_e2e import SAC as SAC_e2e
from src.algos.IQL import IQL
from src.algos.BC import BC

from src.algos.IQL_e2e import IQL as IQL_e2e
from src.algos.BC_e2e import BC as BC_e2e

import random
import pickle
from torch_geometric.data import Data, Batch

from src.algos.lcp_solver_cap import solveLCP
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.supply_chain_env import (
    NetInvMgmtBacklogEnv as env,
)

"""
Script for offline training. Choice from: 
Hierarchical: CQL, IQL, BC
End-to-end: CQL_e2e, IQL_e2e, BC_e2e
"""


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

    def create_dataset(self, edge_index, memory_path, size=50000):
        w = open(f"/datasets/{memory_path}.pkl", "rb")
        replay_buffer = pickle.load(w)
        data = replay_buffer.sample_all(size)

        print(data["rew"].min())
        print(data["rew"].max())

        (
            state_batch,
            edge_attr,
            next_state_batch,
            action_batch,
            reward_batch,
            done_batch,
        ) = (
            data["obs"],
            data["edge_attr"],
            data["obs2"],
            data["act"],
            args.rew_scale * data["rew"],
            data["done"].float(),
        )

        for i in range(len(state_batch)):
            self.data_list.append(
                PairData(
                    edge_index,
                    edge_attr[i],
                    state_batch[i],
                    reward_batch[i],
                    action_batch[i],
                    done_batch[i],
                    edge_index,
                    edge_attr[i],
                    next_state_batch[i],
                )
            )

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
        batch = dict(
            obs=self.obs_buf[:idxs],
            obs2=self.obs2_buf[:idxs],
            act=self.act_buf[:idxs],
            rew=self.rew_buf[:idxs],
            edge_attr=self.edge_attr_buff[:idxs],
            done=self.done_buf[:idxs],
        )
        return {k: torch.as_tensor(v) for k, v in batch.items()}


parser = argparse.ArgumentParser(description="Offline-RL-SC")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
)
# Model parameters
parser.add_argument(
    "--test", type=bool, default=False, help="activates test mode for agent evaluation"
)
parser.add_argument(
    "--cplexpath",
    type=str,
    default="/opt/opl/bin/x86-64_linux/",
    help="defines directory of the CPLEX installation",
)
parser.add_argument(
    "--directory",
    type=str,
    default="saved_files",
    help="defines directory where to save files",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=10000,
    metavar="N",
    help="number of episodes to train agent (default: 16k)",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=50,
    metavar="N",
    help="number of steps per episode (default: T=20)",
)
parser.add_argument(
    "--no-cuda", type=bool, default=False, help="disables CUDA training"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="SAC",
)
parser.add_argument(
    "--clip",
    type=int,
    default=10,
)
parser.add_argument(
    "--p_lr",
    type=float,
    default=1e-4,
)
parser.add_argument(
    "--q_lr",
    type=float,
    default=3e-4,
)

parser.add_argument(
    "--rew_scale",
    type=float,
    default=0.1,
)

parser.add_argument(
    "--memory_path",
    type=str,
    default="SAC",
)

parser.add_argument(
    "--min_q_weight",
    type=float,
    default=1,
)
parser.add_argument(
    "--samples_buffer",
    type=int,
    default=20000,
)
parser.add_argument(
    "--lagrange_thresh",
    type=float,
    default=-1,
)

parser.add_argument(
    "--min_q_version",
    type=int,
    default=3,
)

parser.add_argument(
    "--algo",
    type=str,
    default="CQL",
)

parser.add_argument(
    "--version",
    type=int,
    default=2,
)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(device)


if not args.test:

    env = env(version=args.version)
    print(env.edge_list)
    if args.algo == "CQL":
        model = SAC(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            edge_size=1,
            p_lr=args.p_lr,
            q_lr=args.q_lr,
            alpha=args.alpha,
            batch_size=args.batch_size,
            clip=args.clip,
            min_q_weight=args.min_q_weight,
            lagrange_thresh=args.lagrange_thresh,
            device=device,
        ).to(device)
    elif args.algo == "CQL_e2e":
        model = SAC_e2e(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            edge_size=1,
            p_lr=args.p_lr,
            q_lr=args.q_lr,
            alpha=args.alpha,
            batch_size=args.batch_size,
            clip=args.clip,
            min_q_weight=args.min_q_weight,
            lagrange_thresh=args.lagrange_thresh,
            device=device,
        ).to(device)
    elif args.algo == "IQL":
        print("load IQL")
        model = IQL(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            edge_size=1,
            p_lr=args.p_lr,
            q_lr=args.q_lr,
            batch_size=args.batch_size,
            device=device,
            temperature=3,
            quantile=0.9,
            clip_score=100,
            clip=args.clip,
        ).to(device)

    elif args.algo == "IQL_e2e":
        print("load IQL")
        print(device)
        model = IQL_e2e(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            edge_size=1,
            p_lr=args.p_lr,
            q_lr=args.q_lr,
            batch_size=args.batch_size,
            device=device,
            temperature=3,
            quantile=0.9,
            clip_score=100,
            clip=args.clip,
        ).to(device)
    elif args.algo == "BC_e2e":
        model = BC_e2e(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            edge_size=1,
            p_lr=args.p_lr,
            batch_size=args.batch_size,
            device=device,
            clip_score=100,
            clip=args.clip,
        ).to(device)
    elif args.algo == "BC":
        model = BC(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            edge_size=1,
            p_lr=args.p_lr,
            batch_size=args.batch_size,
            device=device,
            clip_score=100,
            clip=args.clip,
        ).to(device)

    edge_list_bidirectional = env.edge_list + [(j, i) for (i, j) in env.edge_list]
    edge_index = torch.tensor(edge_list_bidirectional).T.long()

    #######################################
    #############Training Loop#############
    #######################################
    # Initialize lists for logging
    Dataset = ReplayData(device=device)
    Dataset.create_dataset(
        edge_index=edge_index, memory_path=args.memory_path, size=args.samples_buffer
    )

    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length

    best_reward = -np.inf  # set best reward
    best_reward_test = -np.inf  # set best reward
    model.train()  # set model in train mode

    total_steps = train_episodes * T  # compute total number of training steps
    epochs = trange(total_steps)  # epoch iterator
    state, ep_ret, ep_len = env.reset(), 0, 0
    SR, PC, TC, OC, HC, UP, UD, VC = 0, 0, 0, 0, 0, 0, 0, 0
    inventory_actions = []
    training_rewards = []
    max_ep_len = T
    order_actions = []
    errors = []
    demand_per_t = []

    for step in range(total_steps):
        batch = Dataset.sample_batch(args.batch_size)
        if "CQL" in args.algo:
            model.update(data=batch, conservative=True)
        else:
            model.update(data=batch)

        if step % 400 == 0:
            (episode_reward, episode_error) = model.test_agent(
                1, env, args.cplexpath, args.directory
            )
            epochs.set_description(f"Step {step} | Reward: {episode_reward:.2f}")
            # Checkpoint best performing model
            if episode_reward >= best_reward and step != 0:
                model.save_checkpoint(path=f"/ckpt/{args.checkpoint_path}_test.pth")
                best_reward = episode_reward

            model.save_checkpoint(path=f"/ckpt/{args.checkpoint_path}_running.pth")

else:
    env = env(version=args.version)
    if args.algo == "CQL":
        model = SAC(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            p_lr=args.p_lr,
            q_lr=args.q_lr,
            alpha=args.alpha,
            batch_size=args.batch_size,
            clip=args.clip,
            min_q_weight=args.min_q_weight,
            lagrange_thresh=args.lagrange_thresh,
            device=device,
            edge_size=1,
        ).to(device)
        model.load_checkpoint(path=f"/ckpt/{args.checkpoint_path}_running.pth")
        model.eval()

    elif args.algo == "IQL":
        print("load IQL")
        model = IQL(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            edge_size=1,
            p_lr=args.p_lr,
            q_lr=args.q_lr,
            batch_size=args.batch_size,
            device=device,
            temperature=3,
            quantile=0.9,
            clip_score=100,
            clip=args.clip,
        ).to(device)

        model.load_checkpoint(path=f"/ckpt/{args.checkpoint_path}_running.pth")
        model.eval()
    elif args.algo == "BC":
        model = BC(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            edge_size=1,
            p_lr=args.p_lr,
            batch_size=args.batch_size,
            device=device,
            clip_score=100,
            clip=args.clip,
        ).to(device)
        model.load_checkpoint(path=f"/ckpt/{args.checkpoint_path}_running.pth")
        model.eval()
    elif args.algo == "BC_e2e":
        model = BC_e2e(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            edge_size=1,
            p_lr=args.p_lr,
            batch_size=args.batch_size,
            device=device,
            clip_score=100,
            clip=args.clip,
        ).to(device)
        model.load_checkpoint(path=f"/ckpt/{args.checkpoint_path}_running.pth")
        model.eval()

    elif args.algo == "IQL_e2e":
        print("load IQL")
        print(device)
        model = IQL_e2e(
            env=env,
            input_size=17 + env.lt_max,
            hidden_size=args.hidden_size,
            edge_size=1,
            p_lr=args.p_lr,
            q_lr=args.q_lr,
            batch_size=args.batch_size,
            device=device,
            temperature=3,
            quantile=0.9,
            clip_score=100,
            clip=args.clip,
        ).to(device)
        model.load_checkpoint(path=f"/ckpt/{args.checkpoint_path}_running.pth")

        model.eval()
    actions = []
    state, ep_ret, ep_len = env.reset(), 0, 0
    episode_rewards = []
    total_steps = 30 * 10
    inventory_actions = []
    VC = []
    OO = []
    errors = []

    comb = []

    inventory = {
        "inventory_1": [],
        "inventory_2": [],
        "inventory_3": [],
        "inventory_4": [],
    }
    diffs = []
    demand = []
    DIFF = 0
    for t in range(total_steps):
        if "e2e" in args.algo:
            action_output = model.select_action(state, deterministic=True)
            main_edges_act = {
                env.reorder_links[i]: max(0, action_output[i])
                for i in range(len(env.reorder_links))
            }
            order_act = action_output[-len(env.factory) :]

            prod_action = {i: int(order_act) for i in env.factory}

        elif "MPC" in args.algo:
            flowMPC, productionMPC, results_MPC = solveMCP(
                env,
                CPLEXPATH=args.cplexpath,
                res_path="scim",
                directory=args.directory,
                T=10,
            )
            flow = [flowMPC[(i, j)] for i, j in env.reorder_links]

            main_edges_act = {
                k: v for k, v in flowMPC.items() if k in env.reorder_links
            }

            prod_action = {k: v for k, v in productionMPC.items() if k in env.factory}

        else:
            inventory_act, order_act, _ = model.select_action(state, deterministic=True)
            main_edges_act, prod, error = solveLCP(
                env,
                desiredDistrib=inventory_act,
                desiredProd=order_act,
                CPLEXPATH=args.cplexpath,
                directory=args.directory,
            )
            prod_action = {i: int(prod[i]) for i in env.factory}
        next_state, reward, done, info = env.step(
            prod_action=prod_action, distr_action=main_edges_act
        )

        VC.append(info["violated_capacity"])

        state = next_state

        ep_ret += reward

        if done:
            episode_rewards.append(ep_ret)
            print(f"Episode reward: {ep_ret}")
            state, ep_ret, ep_len = env.reset(), 0, 0

    print(np.mean(episode_rewards), np.std(episode_rewards))
    print("violated capacity", np.mean(VC), np.std(VC), np.max(VC))
