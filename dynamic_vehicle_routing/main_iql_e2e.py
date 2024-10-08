from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch

from src.envs.amod_env import Scenario, AMoD
from src.algos.IQL_e2e import IQL
import random
import pickle
from torch_geometric.data import Data, Batch
import json
import copy


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

        rows_of_zeros = torch.any(data["act"] != 0, dim=1) == False
        data["act"] = data["act"][rows_of_zeros == False]
        data["obs"] = data["obs"][rows_of_zeros == False]
        data["obs2"] = data["obs2"][rows_of_zeros == False]
        data["rew"] = data["rew"][rows_of_zeros == False]

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

        (state_batch, action_batch, reward_batch, next_state_batch) = (
            data["obs"],
            data["act"],
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

    def sample_batch(self, batch_size=32, return_list=False):
        data = random.sample(self.data_list, batch_size)
        if return_list:
            return data
        else:
            return Batch.from_data_list(data, follow_batch=["x_s", "x_t"]).to(
                self.device
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

    def size(self):
        return len(self.data_list)

    def store_episode_data(self, obs, action, reward, obs2, terminal=False):
        self.episode_data["obs"].append(obs.x)
        self.episode_data["act"].append(torch.as_tensor(action))
        self.episode_data["rew"].append(torch.as_tensor(reward))
        self.episode_data["obs2"].append(obs2.x)
        if terminal:
            for i in range(len(self.episode_data["obs"])):
                self.data_list.append(
                    PairData(
                        edge_index,
                        self.episode_data["obs"][i],
                        args.rew_scale * self.episode_data["rew"][i],
                        self.episode_data["act"][i],
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

    def sample_all(self, idxs):
        print(f"sample {idxs} from {len(self.act_buf)}")
        print(idxs)
        print(self.ptr)
        print(len(self.act_buf))
        # assert idxs <= len(self.act_buf)
        batch = dict(
            obs=self.obs_buf[:idxs],
            obs2=self.obs2_buf[:idxs],
            act=self.act_buf[:idxs],
            rew=self.rew_buf[:idxs],
        )
        return {k: torch.as_tensor(v) for k, v in batch.items()}


parser = argparse.ArgumentParser(description="A2C-GNN")

demand_ratio = {
    "san_francisco": 2,
    "washington_dc": 4.2,
    "chicago": 1.8,
    "nyc_man_north": 1.8,
    "nyc_man_middle": 1.8,
    "nyc_man_south": 1.8,
    "nyc_brooklyn": 9,
    "porto": 4,
    "rome": 1.8,
    "shenzhen_baoan": 2.5,
    "shenzhen_downtown_west": 2.5,
    "shenzhen_downtown_east": 3,
    "shenzhen_north": 3,
}
json_hr = {
    "san_francisco": 19,
    "washington_dc": 19,
    "chicago": 19,
    "nyc_man_north": 19,
    "nyc_man_middle": 19,
    "nyc_man_south": 19,
    "nyc_brooklyn": 19,
    "porto": 8,
    "rome": 8,
    "shenzhen_baoan": 8,
    "shenzhen_downtown_west": 8,
    "shenzhen_downtown_east": 8,
    "shenzhen_north": 8,
}
beta = {
    "san_francisco": 0.2,
    "washington_dc": 0.5,
    "chicago": 0.5,
    "nyc_man_north": 0.5,
    "nyc_man_middle": 0.5,
    "nyc_man_south": 0.5,
    "nyc_brooklyn": 0.5,
    "porto": 0.1,
    "rome": 0.1,
    "shenzhen_baoan": 0.5,
    "shenzhen_downtown_west": 0.5,
    "shenzhen_downtown_east": 0.5,
    "shenzhen_north": 0.5,
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

test_tstep = {"nyc_brooklyn": 4, "shenzhen_downtown_west": 3}

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
    "--samples_buffer",
    type=int,
    default=10000,
)
parser.add_argument(
    "--city",
    type=str,
    default="nyc_brooklyn",
)
parser.add_argument(
    "--rew_scale",
    type=float,
    default=0.01,
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
    "--temperature",
    type=float,
    default=3,
)
parser.add_argument(
    "--quantile",
    type=float,
    default=0.9,
)
parser.add_argument(
    "--finetune",
    type=bool,
    default=False,
)
parser.add_argument(
    "--bc_steps",
    type=int,
    default=0,
)
parser.add_argument(
    "--load_checkpoint",
    type=str,
    default="SAC",
)
parser.add_argument(
    "--Heuristic",
    type=str,
    default="RL",
)
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
# device = torch.device("cuda" if args.cuda else "cpu")
# device = torch.device("cpu" if args.finetune else device)
# device= 'cpu'
device = torch.device("cuda")
city = args.city
# Define AMoD Simulator Environment

config = {
    "learning_rate_p": 1e-4,
    "learning_rate_q": 3e-4,
    "batch_size": args.batch_size,
    "hidden_size": args.hidden_size,
    "directory": args.directory + args.checkpoint_path,
    "samples": args.samples_buffer,
    "temperature": args.temperature,
    "quantile": args.quantile,
    "memory_path": args.memory_path,
}

print(device)
if not args.test and not args.finetune:
    print("start training")

    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio[city],
        json_hr=json_hr[city],
        sd=args.seed,
        json_tstep=args.json_tstep,
        tf=args.max_steps,
    )

    env = AMoD(scenario, beta=beta[city])

    model = IQL(
        env=env,
        input_size=13,
        hidden_size=args.hidden_size,
        p_lr=1e-4,
        q_lr=3e-4,
        batch_size=args.batch_size,
        device=device,
        json_file=f"data/scenario_{city}.json",
        temperature=args.temperature,
        quantile=args.quantile,
        clip_score=100,
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
    epochs = trange(train_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    model.train()  # set model in train mode

    for step in range(args.bc_steps):
        batch = Dataset.sample_batch(args.batch_size)
        model.bc_update(batch)

        if step % 400 == 0:
            (
                episode_reward,
                episode_served_demand,
                episode_rebalancing_cost,
            ) = model.test_agent(1, env, args.cplexpath, args.directory)

            epochs.set_description(
                f"Episode {step/20} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
            )

            # Checkpoint best performing model
            if episode_reward >= best_reward and step > 1000:
                model.save_checkpoint(path=f"/ckpt/BC_" + args.checkpoint_path + ".pth")
                best_reward = episode_reward
        model.save_checkpoint(path=f"/ckpt/BC_" + args.checkpoint_path + "_running.pth")

    # episode_reward, episode_served_demand, episode_rebalancing_cost = model.test_agent(
    #    2, env, args.cplexpath, args.directory
    # )

    # model.save_checkpoint(
    #    path=f"/ckpt/SAC_" + args.checkpoint_path + "_bh_.pth"
    # )

    # epochs.set_description(
    #    f"Episode {0} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
    # )
    if args.bc_steps != 0:
        train_episodes = 0
    for step in range(train_episodes * T):
        if step % 400 == 0:
            (
                episode_reward,
                episode_served_demand,
                episode_rebalancing_cost,
            ) = model.test_agent(1, env, args.cplexpath, args.directory)

            epochs.set_description(
                f"Episode {step/20} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
            )
            # Checkpoint best performing model
            if episode_reward >= best_reward and step > 1000:
                model.save_checkpoint(
                    path=f"/ckpt/IQL_" + args.checkpoint_path + ".pth"
                )
                best_reward = episode_reward
            model.save_checkpoint(
                path=f"/ckpt/IQL_" + args.checkpoint_path + "_running.pth"
            )

        batch = Dataset.sample_batch(args.batch_size)

        model.update(data=batch)

elif args.finetune:
    device = torch.device("cpu")
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio[city],
        json_hr=json_hr[city],
        sd=args.seed,
        json_tstep=test_tstep[city],
        tf=args.max_steps,
    )

    env = AMoD(scenario, beta=beta[city])

    model = IQL(
        env=env,
        input_size=13,
        hidden_size=args.hidden_size,
        p_lr=1e-4,
        q_lr=3e-4,
        batch_size=args.batch_size,
        device=device,
        json_file=f"data/scenario_{city}.json",
        temperature=args.temperature,
        quantile=args.quantile,
        clip_score=100,
    ).to(device)

    model.load_checkpoint(path=f"/ckpt/" + args.checkpoint_path + ".pth")
    print("load model")

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
            if step > 0:
                obs1 = copy.deepcopy(o)
            # take matching step (Step 1 in paper)
            obs, paxreward, done, info, _, _ = env.pax_step(
                CPLEXPATH=args.cplexpath, PATH="scenario_nyc4", directory=args.directory
            )

            o = model.parse_obs(obs=obs, device=device)
            episode_reward += paxreward
            if step > 0:
                rl_reward = paxreward + rebreward
                Dataset.store(obs1, action_rl, args.rew_scale * rl_reward, o)

            action_rl = model.select_action(o.x, o.edge_index)
            # Take action in environment
            rebAction = [max(0, int(i)) for i in action_rl]

            new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
            episode_reward += rebreward
            # track performance over episode
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            # stop episode if terminating conditions are met
            step += 1
            if i_episode > 10:
                batch = Dataset.sample_batch(args.batch_size)

                model.update(data=batch)  # update model

        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}"
        )
        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(
                path=f"/ckpt/SAC_{args.checkpoint_path}_sample_finetune.pth"
            )
            best_reward = episode_reward

        if i_episode % 10 == 0:  # test model every 10th episode
            test_reward, test_served_demand, test_rebalancing_cost = model.test_agent(
                1, env, args.cplexpath, args.directory
            )
            if test_reward >= best_reward_test:
                best_reward_test = test_reward
                model.save_checkpoint(
                    path=f"/ckpt/SAC_{args.checkpoint_path}_test_finetune.pth"
                )

else:
    # Load pre-trained model

    device = torch.device("cpu")
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio[city],
        json_hr=json_hr[city],
        sd=args.seed,
        json_tstep=test_tstep[city],
        tf=args.max_steps,
    )

    env = AMoD(scenario, beta=beta[city])

    model = IQL(
        env=env,
        input_size=13,
        hidden_size=args.hidden_size,
        p_lr=1e-4,
        q_lr=3e-4,
        batch_size=args.batch_size,
        device=device,
        json_file=f"data/scenario_{city}.json",
        temperature=args.temperature,
        quantile=args.quantile,
        clip_score=100,
    ).to(device)

    model.load_checkpoint(path=f"/ckpt/" + args.checkpoint_path + "_running.pth")
    print("load model")
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
    rebactions = []
    # print(env.scenario.demand_input)
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
            # paxreward *= 10
            episode_reward += paxreward
            pax_reward += paxreward
            # use GNN-RL policy (Step 2 in paper)
            # action_rl, concentration, value = model.select_action(obs)
            # print(action_rl)
            # action_rl = [0,0.45,0.28,0,0,0,0,0,0,0,0,0,0.27,0,0,0]
            o = model.parse_obs(obs, device=device)
            action_rl = model.select_action(o.x, o.edge_index, deterministic=False)

            rebAction = [max(0, int(i)) for i in action_rl]
            # print(rebAction)
            # sample random ints between 0 and 2 for len(action_rl) with higher probability of 0s
            # rebAction = [random.choices([0,1,2], weights=[0.8,0.1,0.1])[0] for i in range(len(action_rl))]
            # rebAction = [random.randint(0, 2) for i in range(len(action_rl))]

            # print(sum(rebAction))
            # print([env.acc[i][env.time] for i in env.region])
            # solve minimum rebalancing distance problem (Step 3 in paper)
            # print(rebAction)
            _, rebreward, done, info, _, _ = env.reb_step(rebAction)
            rebactions.append(rebAction)
            # rebreward *= 10
            episode_reward += rebreward
            # track performance over episode
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            # print(info["diff"])
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

        actions_array = np.asarray(actions)
        # w = open('actions_SAC_nyc_M.pkl', "wb")
        # pickle.dump(actions_array, w)

        # log['test_reward'].append(episode_reward)
        # log['test_served_demand'].append(episode_served_demand)
        # log['test_reb_cost'].append(episode_rebalancing_cost)
    # model.log(log, path=f"./{args.directory}/rl_logs/nyc4/a2c_gnn_test.pth")

    rebactions = np.array(rebactions).reshape(-1, len(env.edges))
    # print(np.sum(np.array(rebactions), axis=1))
    print(np.sum(rebactions, axis=0).reshape(env.nregion, env.nregion))
    # print(np.mean(actions, axis=0))
    print(np.mean(rewards), np.std(rewards))
    print("demand", np.mean(demands), np.std(demands))
    print("cost", np.mean(costs), np.std(costs))
    print(np.mean(paxs), np.std(paxs))
