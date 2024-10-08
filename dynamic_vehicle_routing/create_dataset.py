import argparse
from tqdm import trange
import numpy as np
import torch
import pickle
from src.envs.amod_env import Scenario, AMoD
from src.algos.reb_flow_solver import INF, DTV
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
from src.algos.heuristic import PROP
import copy
from torch_geometric.data import Data

import json
import numpy as np
import torch

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

    def parse_obs(self, obs):
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
        data = Data(x, edge_index)
        return data


def flow_to_dist(flows, acc):
    """AMoD
    Convert flow to distribution
    :param flows: rebflow decision in time step t (i,j)
    :param acc: idle vehicle distribtution in time step t+1
    :return: distribution
    """
    desiredAcc = acc.copy()
    for i, j in flows.keys():
        desiredAcc[i] -= flows[i, j]
        desiredAcc[j] += flows[i, j]

    total_acc = sum(acc.values())
    if total_acc == 0:
        distribution = {i: 1 / len(acc) for i in acc}
    else:
        distribution = {i: desiredAcc[i] / total_acc for i in desiredAcc}
    return distribution


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
        )
        return {k: torch.as_tensor(v) for k, v in batch.items()}


demand_ratio = {
    "san_francisco": 2,
    "nyc_brooklyn": 9,
    "shenzhen_downtown_west": 2.5,
}
json_hr = {
    "san_francisco": 19,
    "nyc_brooklyn": 19,
    "shenzhen_downtown_west": 8,
}
beta = {
    "san_francisco": 0.2,
    "nyc_brooklyn": 0.5,
    "shenzhen_downtown_west": 0.5,
}

test_tstep = {"san_francisco": 3, "nyc_brooklyn": 4, "shenzhen_downtown_west": 3}

parser = argparse.ArgumentParser(description="A2C-GNN")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
)
parser.add_argument(
    "--demand_ratio",
    type=int,
    default=0.5,
    metavar="S",
    help="demand_ratio (default: 0.5)",
)
parser.add_argument(
    "--json_hr", type=int, default=7, metavar="S", help="json_hr (default: 7)"
)
parser.add_argument(
    "--json_tstep",
    type=int,
    default=3,
    metavar="S",
    help="minutes per timestep (default: 3min)",
)
parser.add_argument(
    "--beta",
    type=int,
    default=0.5,
    metavar="S",
    help="cost of rebalancing (default: 0.5)",
)

parser.add_argument(
    "--city",
    type=str,
    default="nyc_brooklyn",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="_",
)
parser.add_argument(
    "--max_reb",
    type=float,
    default=6,
    help="parameter for INF model"
)
parser.add_argument(
    "--roh",
    type=float,
    default=4,
    help="parameter for INF model"
)
parser.add_argument(
    "--Heuristic",
    type=str, 
    default="LP3", 
    help="Choose from: INF, DTV, PROP, DISP")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

city = args.city

scenario = Scenario(
    json_file=f"data/scenario_{city}.json",
    demand_ratio=demand_ratio[city],
    json_hr=json_hr[city],
    sd=args.seed,
    json_tstep=3,
    tf=args.max_steps,
)

env = AMoD(scenario, beta=beta[city])

parser = GNNParser(env, T=6, json_file=f"data/scenario_{city}.json")

test_episodes = args.max_episodes  # set max number of training episodes
T = args.max_steps  # set episode length
epochs = trange(test_episodes)  # epoch iterator
# Initialize lists for logging
log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}

obs_dim = (env.nregion, 13)

act_dim = len(env.edges)
replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=10000)
replay_buffer_distr = ReplayBuffer(obs_dim=obs_dim, act_dim=env.nregion, size=10000)

heuristic = PROP(horizon=6)

rewards = []
demands = []
costs = []
epsilon = 1e-6

for episode in range(510):
    episode_reward = 0
    episode_served_demand = 0
    episode_rebalancing_cost = 0
   
    obs = env.reset()
    done = False
    # while (not done):
    current_eps = []
    step = 0
    while not done:
        # take matching step (Step 1 in paper)
        if step > 0:
            obs1 = copy.deepcopy(o)
        obs, paxreward, _, info, _, paxAction = env.pax_step(
            CPLEXPATH=args.cplexpath, PATH="scenario_nyc4", directory=args.directory
        )
        o = parser.parse_obs(obs)

        episode_reward += paxreward
        open_requests = {}

        for i in env.region:
            open_requests[i] = 0

        for i, j in env.scenario.demand_input:
            for t in range(env.time + 1, env.time + 7):
                open_requests[i] += env.scenario.demand_input[i, j][t]

        for i in env.region:
            open_requests[i] = round(open_requests[i] / 6)

        if step > 0:
            rl_reward = paxreward + rebreward
            replay_buffer.store(obs1.x, rebAction, rl_reward, o.x)
            replay_buffer_distr.store(obs1.x, action_rl, rl_reward, o.x)
       
        if args.Heuristic == "DISP":
            m = torch.distributions.dirichlet.Dirichlet(
                torch.tensor([1.0] * env.nregion)
            )
            action_rl = m.sample().numpy()
            
        if args.Heuristic == "INF": 
            rebAction, flows = INF(
                env, open_requests, max_reb=args.max_reb, roh=args.roh
            )
            acc = {n: env.acc[n][env.time + 1] for n in env.region}
            action_rl = flow_to_dist(flows, acc=acc)
        
        if args.Heuristic == "DTV":
            rebAction, flows = DTV(env, open_requests)
            acc = {n: env.acc[n][env.time + 1] for n in env.region}
            action_rl = flow_to_dist(flows, acc=acc)
       
        if args.Heuristic == "PROP":
            action_rl = heuristic.next_action(env)

        if args.Heuristic == "PROP" or args.Heuristic == "DISP":
            desiredAcc = {
                env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                for i in range(len(env.region))
            }
            rebAction = solveRebFlow(
                env,
                "scenario_nyc4",
                desiredAcc,
                args.cplexpath,
                directory=args.directory,
            )

        a = np.array([action_rl[n] for n in env.region])
        non_zero_elements = a > 0
        zero_elements = a == 0
        num_non_zero = np.sum(non_zero_elements)
        num_zero = np.sum(zero_elements)

        # Subtract epsilon from non-zero elements and add to zero elements
        a[non_zero_elements] -= num_zero * epsilon / num_non_zero
        a[zero_elements] = epsilon
        action_rl = list(a)

        new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
        episode_reward += rebreward

        episode_served_demand += info["served_demand"]
        episode_rebalancing_cost += info["rebalancing_cost"]
        step += 1

    rewards.append(episode_reward)
    demands.append(episode_served_demand)
    costs.append(episode_rebalancing_cost)
    # stop episode if terminating conditions are met
    # Send current statistics to screen
    epochs.set_description(
        f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}"
    )


w = open(f"/Replaymemories/Replaymemory_{args.city}_{args.Heuristic}_flows.pkl", "wb")
pickle.dump(replay_buffer, w)
w.close()
print("replay_buffer", replay_buffer.size)

w = open(f"/Replaymemories/Replaymemory_{args.city}_{args.Heuristic}_distr.pkl", "wb")
pickle.dump(replay_buffer_distr, w)
w.close()
print("replay_buffer", replay_buffer_distr.size)
