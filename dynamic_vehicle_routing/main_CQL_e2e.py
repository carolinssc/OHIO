from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.amod_env import Scenario, AMoD
from src.algos.CQL_e2e import SAC
import random
import pickle
from torch_geometric.data import Data, Batch
import json


def return_to_go(rewards):
    gamma = 0.97
    # calculate the true value using rewards returned from the environment
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


class PairData(Data):
    def __init__(
        self,
        edge_index_s=None,
        x_s=None,
        reward=None,
        action=None,
        edge_index_t=None,
        x_t=None,
    ):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
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
    A simple FIFO experience replay buffer for SAC agents.
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
        w = open(f"/Replaymemories/{memory_path}.pkl", "rb")

        replay_buffer = pickle.load(w)
        data = replay_buffer.sample_all(size)
        # data["act"] = data["act"] + 1e-10
        # data["act"] = data["act"] / data["act"].sum(axis=1, keepdims=True)

        if st:
            mean = data["rew"].mean()
            std = data["rew"].std()
            data["rew"] = (data["rew"] - mean) / (std + 1e-16)
        elif sc:
            data["rew"] = (data["rew"] - data["rew"].min()) / (
                data["rew"].max() - data["rew"].min()
            )

        print(data["rew"].min())
        print(data["rew"].max())

        # (state_batch, action_batch,reward_batch,next_state_batch, mc_returns) = (data["obs"],data["act"], args.rew_scale*data["rew"],data["obs2"], args.rew_scale*data["mc_returns"])
        # for i in range(len(state_batch)):
        #    self.data_list.append(PairData(edge_index, state_batch[i], reward_batch[i],action_batch[i], mc_returns[i], edge_index, next_state_batch[i]))
        (state_batch, action_batch, reward_batch, next_state_batch) = (
            data["obs"],
            0.1 * data["act"],
            args.rew_scale * data["rew"],
            data["obs2"],
        )
        for i in range(len(state_batch)):
            self.data_list.append(
                PairData(
                    edge_index,
                    state_batch[i],
                    reward_batch[i],
                    action_batch[i],
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
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.mc_returns = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, mc_returns):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.mc_returns[self.ptr] = mc_returns
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_mc_returns(self, mc_returns):
        self.mc_returns = mc_returns

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            mc_returns=self.mc_returns[idxs],
        )
        return {k: torch.as_tensor(v) for k, v in batch.items()}

    def sample_all(self, samples):
        if samples > self.size:
            print(f"Only {self.ptr} samples in buffer, returning all samples")
            samples = self.ptr
        batch = dict(
            obs=self.obs_buf[:samples],
            obs2=self.obs2_buf[:samples],
            act=self.act_buf[:samples],
            rew=self.rew_buf[:samples],
            # mc_returns=self.mc_returns[:samples]
        )
        return {k: torch.as_tensor(v) for k, v in batch.items()}


parser = argparse.ArgumentParser(description="A2C-GNN")

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

checkpoints_50 = {
    "san_francisco": 7.2,
    "shenzhen_downtown_west": 30660,
    "nyc_brooklyn": 23233,
}
checkpoints_75 = {
    "san_francisco": 10.8,
    "shenzhen_downtown_west": 45990,
    "nyc_brooklyn": 34850,
}
checkpoints_90 = {
    "san_francisco": 12.96,
    "shenzhen_downtown_west": 58254,
    "nyc_brooklyn": 41820,
}

test_tstep = {"san_francisco": 3, "nyc_brooklyn": 4, "shenzhen_downtown_west": 3}

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
    help="number of steps per episode (default: T=60)",
)
parser.add_argument("--cuda", type=bool, default=True, help="disables CUDA training")

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
    "--memory_path",
    type=str,
    default="SAC",
)
parser.add_argument(
    "--min_q_weight",
    type=float,
    default=5,
)
parser.add_argument(
    "--samples_buffer",
    type=int,
    default=60000,
)
parser.add_argument(
    "--lagrange_thresh",
    type=float,
    default=-1,
)
parser.add_argument(
    "--n",
    type=int,
    default=1,
)
parser.add_argument(
    "--city",
    type=str,
    default="nyc_brooklyn",
)
parser.add_argument(
    "--rew_scale",
    type=float,
    default=0.1,
)
parser.add_argument(
    "--st",
    type=bool,
    default=False,
)
parser.add_argument(
    "--sc",
    type=bool,
    default=False,
)
parser.add_argument(
    "--enable_calql",
    type=bool,
    default=False,
)
parser.add_argument(
    "--load_ckpt",
    type=str,
    default="SAC",
)
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

city = args.city
# Define AMoD Simulator Environment

config = {
    "learning_rate_p": 1e-4,
    "learning_rate_q": 3e-4,
    "batch_size": args.batch_size,
    "alpha": args.alpha,
    "use_automatic_entropy_tuning": False,
    "hidden_size": args.hidden_size,
    "min_q_weight": args.min_q_weight,
    "lagrange_thresh": args.lagrange_thresh,
    "directory": args.directory + args.checkpoint_path,
    "samples": args.samples_buffer,
}


if not args.test:
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
        n=args.n,
        device=device,
        json_file=f"data/scenario_{city}.json",
        min_q_version=3,
    ).to(device)

    with open(f"data/scenario_{city}.json", "r") as file:
        data = json.load(file)

    edge_index = torch.vstack(
        (
            torch.tensor([edge["i"] for edge in data["topology_graph"]]).view(1, -1),
            torch.tensor([edge["j"] for edge in data["topology_graph"]]).view(1, -1),
        )
    ).long()
    #######################################
    #############Training Loop#############
    #######################################

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
    epochs = trange(train_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    model.train()  # set model in train mode

    for step in range(train_episodes * 20):
        if step % 400 == 0:
            model.eval()
            (
                episode_reward,
                episode_served_demand,
                episode_rebalancing_cost,
            ) = model.test_agent(2, env, args.cplexpath, args.directory)
            model.train()
            epochs.set_description(
                f"Episode {step/20} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
            )
            # Checkpoint best performing model
            if episode_reward >= best_reward and step > 1000:
                model.save_checkpoint(
                    path=f"/ckpt/SAC_" + args.checkpoint_path + ".pth"
                )
                best_reward = episode_reward
            model.save_checkpoint(
                path=f"/ckpt/SAC_" + args.checkpoint_path + "_running.pth"
            )

        batch = Dataset.sample_batch(args.batch_size)
        model.update(data=batch, conservative=True, enable_calql=args.enable_calql)

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
        n=args.n,
        device=device,
        json_file=f"data/scenario_{city}.json",
    ).to(device)

    model.load_checkpoint(path=f"/ckpt/SAC_" + args.checkpoint_path + "_running.pth")
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
    actions = []
    paxs = []
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

            o = model.parse_obs(obs, device=device)

            rebAction = model.select_action(o.x, o.edge_index, deterministic=False)
            rebAction = [max(int(i.item()), 0) for i in rebAction]

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
        paxs.append(pax_reward)
        import pickle
