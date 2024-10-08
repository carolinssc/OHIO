from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch

from src.envs.amod_env import Scenario, AMoD
from src.algos.CQL import SAC
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
import random
import pickle
import copy
from src.algos.reb_flow_solver import solveRebFlow
from torch_geometric.data import Data, Batch
import json


def return_to_go(rewards):
    """
    Calculate the return-to-go for a given trajectory.
    """
    gamma = 0.97
    return_to_go = [0] * len(rewards)
    prev_return = 0
    for i in range(len(rewards)):
        return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return
        prev_return = return_to_go[-i - 1]
    return np.array(return_to_go, dtype=np.float32)


class PairData(Data):
    def __init__(
        self,
        edge_index_s=None,
        x_s=None,
        reward=None,
        action=None,
        mc_returns=None,
        edge_index_t=None,
        x_t=None,
    ):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.mc_returns = mc_returns
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
    ReplayBuffer for Pytorch Geometric Data
    """

    def __init__(self, device, rew_scale):
        self.device = device
        self.data_list = []
        self.rew_scale = rew_scale
        self.episode_data = {}
        self.episode_data["obs"] = []
        self.episode_data["act"] = []
        self.episode_data["rew"] = []
        self.episode_data["obs2"] = []

    def create_dataset(self, edge_index, memory_path, size=60000, st=False, sc=False):
        """
        edge_index: Adjaency matrix of the graph
        memory_path: Path to the replay memory
        size: Size of the replay memory
        st: Standardize the rewards
        sc: min-max scaling of rewards
        """
        w = open(f"Replaymemories/{memory_path}.pkl", "rb")

        replay_buffer = pickle.load(w)
        data = replay_buffer.sample_all(size)

        if st:
            mean = data["rew"].mean()
            std = data["rew"].std()
            data["rew"] = (data["rew"] - mean) / (std + 1e-16)
        elif sc:
            data["rew"] = (data["rew"] - data["rew"].min()) / (
                data["rew"].max() - data["rew"].min()
            )

        (state_batch, action_batch, reward_batch, next_state_batch, mc_returns) = (
            data["obs"],
            data["act"],
            args.rew_scale * data["rew"],
            data["obs2"],
            args.rew_scale * data["mc_returns"],
        )
        for i in range(len(state_batch)):
            self.data_list.append(
                PairData(
                    edge_index,
                    state_batch[i],
                    reward_batch[i],
                    action_batch[i],
                    mc_returns[i],
                    edge_index,
                    next_state_batch[i],
                )
            )

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

    def sample_batch(self, batch_size=32, return_list=False):
        data = random.sample(self.data_list, batch_size)
        if return_list:
            return data
        else:
            return Batch.from_data_list(data, follow_batch=["x_s", "x_t"]).to(
                self.device
            )

    def store_episode_data(self, obs, action, reward, obs2, terminal=False):
        """
        store new transitions
        """
        self.episode_data["obs"].append(obs.x)
        self.episode_data["act"].append(torch.as_tensor(action))
        self.episode_data["rew"].append(torch.as_tensor(reward))
        self.episode_data["obs2"].append(obs2.x)
        if terminal:
            mc_returns = return_to_go(self.episode_data["rew"])
            for i in range(len(self.episode_data["obs"])):
                self.data_list.append(
                    PairData(
                        edge_index,
                        self.episode_data["obs"][i],
                        args.rew_scale * self.episode_data["rew"][i],
                        self.episode_data["act"][i],
                        args.rew_scale * torch.as_tensor(mc_returns[i]),
                        edge_index,
                        self.episode_data["obs2"][i],
                    )
                )

            self.episode_data["obs"] = []
            self.episode_data["act"] = []
            self.episode_data["rew"] = []
            self.episode_data["obs2"] = []


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    Necessary to load the offline datasets
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.mc_returns = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def sample_all(self, samples):
        if samples > self.size:
            samples = self.ptr
        batch = dict(
            obs=self.obs_buf[:samples],
            obs2=self.obs2_buf[:samples],
            act=self.act_buf[:samples],
            rew=self.rew_buf[:samples],
            mc_returns=self.mc_returns[:samples],
        )
        return {k: torch.as_tensor(v) for k, v in batch.items()}


parser = argparse.ArgumentParser(description="Cal-CQL-GNN")

demand_ratio = {
    "san_francisco": 2,
    "washington_dc": 4.2,
    "nyc_brooklyn": 9,
    "shenzhen_downtown_west": 2.5,
}
json_hr = {
    "san_francisco": 19,
    "washington_dc": 19,
    "nyc_brooklyn": 19,
    "shenzhen_downtown_west": 8,
}
beta = {
    "san_francisco": 0.2,
    "washington_dc": 0.5,
    "nyc_brooklyn": 0.5,
    "shenzhen_downtown_west": 0.5,
}

test_tstep = {"san_francisco": 3,
              "nyc_brooklyn": 4, "shenzhen_downtown_west": 3}

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

# Training specifics
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
    help="number of steps per episode (default: T=60)",
)
parser.add_argument("--cuda", type=bool, default=True,
                    help="disables CUDA training")

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
    help="entropy regularization coefficient (default: 0.3)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    help="hidden size of the networks (default: 256)",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="Cal_CQL",
    help="name of the checkpoint file (default: SAC)",
)
parser.add_argument(
    "--memory_path",
    type=str,
    default="Replaymemory_shenzhen_downtown_west_M",
    help="name of the offline dataset file",
)
parser.add_argument(
    "--min_q_weight",
    type=float,
    default=5,
    help="CQL coefficient (eta in paper)",
)
parser.add_argument(
    "--samples_buffer",
    type=int,
    default=10000,
    help="size of the replay buffer",
)
parser.add_argument(
    "--lagrange_thresh",
    type=float,
    default=-1,
    help="threshold for the lagrange tuning of entropy (default: -1 =disabled)",
)
parser.add_argument(
    "--city",
    type=str,
    default="shenzhen_downtown_west",
    help="city to train on (default: nyc_brooklyn)",
)
parser.add_argument(
    "--rew_scale",
    type=float,
    default=0.1,
    help="scaling factor for the rewards (default: 0.1)",
)
parser.add_argument(
    "--st",
    type=bool,
    default=False,
    help="standardize the rewards (default: False)",
)
parser.add_argument(
    "--sc",
    type=bool,
    default=False,
    help="min-max scale the rewards (default: False)",
)
parser.add_argument(
    "--fine_tune",
    type=bool,
    default=False,
    help="fine tune the offline trained model (default: False)",
)
parser.add_argument(
    "--enable_calql",
    type=bool,
    default=True,
    help="enable the Cal-CQL (default: False)",
)
parser.add_argument(
    "--load_ckpt",
    type=str,
    default="Cal_CQL",
    help="name of the checkpoint file to load for online-fine-tuning(default: SAC)",
)
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

city = args.city
# Define AMoD Simulator Environment

if not args.test and not args.fine_tune:
    # Train offline
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio[city],
        json_hr=json_hr[city],
        sd=args.seed,
        json_tstep=args.json_tstep,
        tf=args.max_steps,
    )

    env = AMoD(scenario, beta=beta[city])

    model = SAC(
        env=env,
        input_size=13,
        hidden_size=args.hidden_size,
        p_lr=1e-4,
        q_lr=3e-4,
        alpha=args.alpha,
        batch_size=args.batch_size,
        deterministic_backup=True,
        min_q_weight=args.min_q_weight,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=args.lagrange_thresh,
        device=device,
        json_file=f"data/scenario_{city}.json",
        min_q_version=3,
    ).to(device)

    with open(f"data/scenario_{city}.json", "r") as file:
        data = json.load(file)

    edge_index = torch.vstack(
        (
            torch.tensor([edge["i"]
                         for edge in data["topology_graph"]]).view(1, -1),
            torch.tensor([edge["j"]
                         for edge in data["topology_graph"]]).view(1, -1),
        )
    ).long()
    #######################################
    ############# Training Loop#############
    #######################################
    # Initialize lists for logging
    Dataset = ReplayData(device=device, rew_scale=args.rew_scale)
    Dataset.create_dataset(
        edge_index=edge_index,
        memory_path=args.memory_path,
        size=args.samples_buffer,
        st=args.st,
        sc=args.sc,
    )

    log = {"train_reward": [], "train_served_demand": [], "train_reb_cost": []}
    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    training_steps = train_episodes * 20
    best_reward = -np.inf  # set best reward
    model.train()  # set model in train mode

    for step in range(training_steps):
        if step % 50 == 0:
            print(f"Step {step}")

        batch = Dataset.sample_batch(args.batch_size)
        model.update(data=batch, conservative=True,
                     enable_calql=args.enable_calql)
        model.save_checkpoint(path=f"ckpt/" + args.checkpoint_path + ".pth")


elif args.fine_tune:
    # Online fine-tuning
    with open(f"data/scenario_{city}.json", "r") as file:
        data = json.load(file)

    edge_index = torch.vstack(
        (
            torch.tensor([edge["i"]
                         for edge in data["topology_graph"]]).view(1, -1),
            torch.tensor([edge["j"]
                         for edge in data["topology_graph"]]).view(1, -1),
        )
    ).long()

    Dataset = ReplayData(device="cpu", rew_scale=args.rew_scale)
    Dataset.create_dataset(
        edge_index=edge_index,
        memory_path=args.memory_path,
        size=args.samples_buffer,
        st=args.st,
        sc=args.sc,
    )
    replay_buffer_online = ReplayData(device="cpu", rew_scale=args.rew_scale)

    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio[city],
        json_hr=json_hr[city],
        sd=args.seed,
        json_tstep=args.json_tstep,
        tf=args.max_steps,
    )

    env = AMoD(scenario, beta=beta[city])

    model = SAC(
        env=env,
        input_size=13,
        hidden_size=args.hidden_size,
        p_lr=1e-3,
        q_lr=1e-3,
        alpha=args.alpha,
        batch_size=args.batch_size,
        deterministic_backup=True,
        min_q_weight=args.min_q_weight,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=args.lagrange_thresh,
        device=device,
        json_file=f"data/scenario_{city}.json",
        min_q_version=3,
    ).to(device)
    print("load model")
    model.load_checkpoint(f"ckpt/{args.load_ckpt}.pth")

    log = {"train_reward": [], "train_served_demand": [], "train_reb_cost": []}
    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(train_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    best_reward_test = -np.inf
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

            o = model.parse_obs(obs=obs, device=device)
            episode_reward += paxreward
            if step > 0:
                rl_reward = paxreward + rebreward
                # model.replay_buffer.store(obs1, action_rl, args.rew_scale * rl_reward, o)
                if step == T - 1:
                    replay_buffer_online.store_episode_data(
                        obs1, action_rl, rl_reward, o, terminal=True
                    )
                else:
                    replay_buffer_online.store_episode_data(
                        obs1, action_rl, rl_reward, o, terminal=False
                    )

            action_rl = model.select_action(
                o.x, o.edge_index, deterministic=False)

            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {
                env.region[i]: int(
                    action_rl[i] * dictsum(env.acc, env.time + 1))
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
            # rebreward *=10
            episode_reward += rebreward
            # track performance over episode
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]

            step += 1
            if i_episode > 10:
                batch1 = Dataset.sample_batch(
                    int(args.batch_size * 0.25), return_list=True
                )
                batch2 = replay_buffer_online.sample_batch(
                    int(args.batch_size * 0.75), return_list=True
                )
                batch = batch1 + batch2
                batch = Batch.from_data_list(
                    batch, follow_batch=["x_s", "x_t"])
                model.update(data=batch, conservative=False,
                             enable_calql=False)
        
        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
        )
        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(
                path=f"/ckpt/{args.checkpoint_path}_sample.pth")
            best_reward = episode_reward
        model.save_checkpoint(path=f"/ckpt/{args.checkpoint_path}_running.pth")
        if i_episode % 10 == 0:
            test_reward, test_served_demand, test_rebalancing_cost = model.test_agent(
                1, env, args.cplexpath, args.directory
            )
            if test_reward >= best_reward_test:
                best_reward_test = test_reward
                model.save_checkpoint(
                    path=f"/ckpt/{args.checkpoint_path}_test.pth")


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

    model = SAC(
        env=env,
        input_size=13,
        hidden_size=args.hidden_size,
        p_lr=1e-4,
        q_lr=3e-4,
        alpha=args.alpha,
        batch_size=args.batch_size,
        deterministic_backup=True,
        min_q_weight=args.min_q_weight,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=args.lagrange_thresh,
        device=device,
        json_file=f"data/scenario_{city}.json",
    ).to(device)

    model.load_checkpoint(path=f"/ckpt/" + args.checkpoint_path + ".pth")
    print("load model")
    model.eval()
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

            o = model.parse_obs(obs, device=device)
            action_rl = model.select_action(
                o.x, o.edge_index, deterministic=True)

            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {
                env.region[i]: int(
                    action_rl[i] * dictsum(env.acc, env.time + 1))
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
