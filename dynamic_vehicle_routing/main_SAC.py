from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.amod_env import Scenario, AMoD
from src.algos.sac import SAC
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
import json
from torch_geometric.data import Data
import copy


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


# Define calibrated simulation parameters
demand_ratio = {
    "nyc_brooklyn": 9,
    "shenzhen_downtown_west": 2.5,
}
json_hr = {
    "nyc_brooklyn": 19,
    "shenzhen_downtown_west": 8,
}
beta = {
    "nyc_brooklyn": 0.5,
    "shenzhen_downtown_west": 0.5,
}

test_tstep = {"nyc_brooklyn": 4, "shenzhen_downtown_west": 3}

parser = argparse.ArgumentParser(description="SAC-GNN")

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
    default=20,
    metavar="N",
    help="number of steps per episode (default: T=20)",
)
parser.add_argument("--no-cuda", type=bool, default=True, help="disables CUDA training")
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="batch size for training (default: 100)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.3,
    help="entropy coefficient (default: 0.3)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    help="hidden size of neural networks (default: 256)",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="SAC",
    help="name of checkpoint file to save/load (default: SAC)",
)
parser.add_argument(
    "--clip",
    type=int,
    default=500,
    help="clip value for gradient clipping (default: 500)",
)
parser.add_argument(
    "--p_lr",
    type=float,
    default=1e-3,
    help="learning rate for policy network (default: 1e-4)",
)
parser.add_argument(
    "--q_lr",
    type=float,
    default=1e-3,
    help="learning rate for Q networks (default: 4e-3)",
)
parser.add_argument(
    "--city",
    type=str,
    default="shenzhen_downtown_west",
    help="city to train on",
)
parser.add_argument(
    "--rew_scale",
    type=float,
    default=0.1,
    help="reward scaling factor (default: 0.1)",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
city = args.city


if not args.test:
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio[city],
        json_hr=json_hr[city],
        sd=args.seed,
        json_tstep=args.json_tstep,
        tf=args.max_steps,
    )

    env = AMoD(scenario, beta=beta[city])

    parser = GNNParser(
        env, T=6, json_file=f"data/scenario_{city}.json"
    )  # Timehorizon T=6 (K in paper)

    model = SAC(
        env=env,
        input_size=13,
        hidden_size=args.hidden_size,
        p_lr=args.p_lr,
        q_lr=args.q_lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        use_automatic_entropy_tuning=False,
        clip=args.clip,
        critic_version=args.critic_version,
    ).to(device)

    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(train_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    best_reward_test = -np.inf  # set best reward
    model.train()  # set model in train mode

    for i_episode in epochs:
        obs = env.reset()  # initialize environment
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        actions = []

        current_eps = []
        done = False
        step = 0
        while not done:
            # take matching step (Step 1 in paper)
            if step > 0:
                obs1 = copy.deepcopy(o)

            obs, paxreward, done, info, _, _ = env.pax_step(
                CPLEXPATH=args.cplexpath, PATH="scenario_nyc4", directory=args.directory
            )

            o = parser.parse_obs(obs=obs)
            episode_reward += paxreward
            if step > 0:
                # store transition in memroy
                rl_reward = paxreward + rebreward
                model.replay_buffer.store(
                    obs1, action_rl, args.rew_scale * rl_reward, o
                )

            action_rl = model.select_action(o)

            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {
                env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                for i in range(len(env.region))
            }
            # solve minimum rebalancing distance problem (Step 3 in paper)
            rebAction = solveRebFlow(
                env,
                "scenario_nyc4",
                desiredAcc,
                args.cplexpath,
                directory=args.directory,
            )
            # Take action in environment
            new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
            episode_reward += rebreward
            # track performance over episode
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]

            step += 1
            if i_episode > 10:
                # sample from memory and update model
                batch = model.replay_buffer.sample_batch(args.batch_size, norm=False)
                model.update(data=batch)

        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
        )
        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(path=f"/ckpt/{args.checkpoint_path}_sample.pth")
            best_reward = episode_reward
        model.save_checkpoint(path=f"/ckpt/{args.checkpoint_path}_running.pth")
        if i_episode % 10 == 0:
            test_reward, test_served_demand, test_rebalancing_cost = model.test_agent(
                1, env, args.cplexpath, args.directory, parser=parser
            )
            if test_reward >= best_reward_test:
                best_reward_test = test_reward
                model.save_checkpoint(path=f"/ckpt/{args.checkpoint_path}_test.pth")
else:
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio[city],
        json_hr=json_hr[city],
        sd=args.seed,
        json_tstep=test_tstep[city],
        tf=args.max_steps,
    )

    env = AMoD(scenario, beta=beta[city])
    parser = GNNParser(env, T=6, json_file=f"data/scenario_{city}.json")

    model = SAC(
        env=env,
        input_size=13,
        hidden_size=256,
        p_lr=1e-3,
        q_lr=1e-3,
        alpha=0.3,
        batch_size=100,
        use_automatic_entropy_tuning=False,
        critic_version=args.critic_version,
    ).to(device)

    print("load model")
    model.load_checkpoint(path=f"/ckpt/{args.checkpoint_path}.pth")

    test_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(test_episodes)  # epoch iterator
    # Initialize lists for logging
    log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}

    rewards = []
    demands = []
    costs = []

    for episode in range(10):
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        obs = env.reset()
        done = False
        k = 0
        pax_reward = 0
        while not done:
            # take matching step (Step 1 in paper)
            obs, paxreward, done, info, _, _ = env.pax_step(
                CPLEXPATH=args.cplexpath,
                PATH="scenario_nyc4_test",
                directory=args.directory,
            )

            episode_reward += paxreward
            pax_reward += paxreward
            # use GNN-RL policy (Step 2 in paper)
            o = parser.parse_obs(obs=obs)
            action_rl = model.select_action(o, deterministic=False)

            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {
                env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + 1))
                for i in range(len(env.region))
            }
            # solve minimum rebalancing distance problem (Step 3 in paper)
            rebAction = solveRebFlow(
                env, "scenario_nyc4_test", desiredAcc, args.cplexpath, args.directory
            )

            _, rebreward, done, info, _, _ = env.reb_step(rebAction)

            episode_reward += rebreward
            # track performance over episode
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            k += 1
        # Send current statistics to screen
        epochs.set_description(
            f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}"
        )
        # Log KPIs

        rewards.append(episode_reward)
        demands.append(episode_served_demand)
        costs.append(episode_rebalancing_cost)

    print("Rewards (mean, std):", np.mean(rewards), np.std(rewards))
    print("Served demand (mean, std):", np.mean(demands), np.std(demands))
    print("Rebalancing cost (mean, std):", np.mean(costs), np.std(costs))
