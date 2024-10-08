from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch

from src.envs.supply_chain_env import (
    NetInvMgmtLostSalesEnv as env,
)
from src.algos.sac import SAC
from src.algos.sac_e2e import SAC as SAC_e2e
from supply_chain.src.algos.lcp_solver_cap import solveLCP

"""
This script is used to train the online SAC agent
"""

parser = argparse.ArgumentParser(description="SAC-SC")
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

parser.add_argument("--no-cuda", type=bool, default=True, help="disables CUDA training")
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
    "--algo",
    type=str,
    default="sac",
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
    if args.algo == "sac":
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
            device=device,
        ).to(device)
    if args.algo == "sac_e2e":
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
            device=device,
        ).to(device)

    else:
        raise ValueError("Invalid algorithm")

    train_episodes = args.max_episodes  # set max number of training episodes

    epochs = trange(train_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    best_reward_test = -np.inf  # set best reward
    model.train()  # set model in train mode

    total_steps = train_episodes * 50  # compute total number of training steps

    state, ep_ret, ep_len = env.reset(), 0, 0
    SR, PC, TC, OC, HC, UP, UD, VC, OO = 0, 0, 0, 0, 0, 0, 0, 0, 0
    inventory_actions = []
    training_rewards = []
    max_ep_len = T
    order_actions = []
    errors = []
    demand_per_t = []

    for t in range(total_steps):
        inventory_act, order_act, combined_action = model.select_action(state)

        inventory_actions.append(inventory_act)
        order_actions.append(order_act)

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
        reward = torch.tensor(reward, dtype=torch.float32)

        ep_ret += reward
        ep_len += 1
        SR += info["sales_revenue"]
        PC += info["purchasing_costs"]
        TC += info["transportation_costs"]
        OC += info["operating_costs"]
        HC += info["holding_costs"]
        UP += info["unfulfilled_penalty"]
        VC += info["violated_capacity"]
        OO += info["over_order"]

        demand_per_t.append(info["demand"])

        model.rewards.append(reward)

        model.replay_buffer.store(
            state, combined_action, args.rew_scale * reward, next_state, done
        )

        state = next_state

        if t > 2 * args.batch_size:
            batch = model.replay_buffer.sample_batch(args.batch_size)

            model.update(data=batch)

        if done:  # end of episode

            if ep_ret >= best_reward:
                model.save_checkpoint(path=f"/ckpt/{args.checkpoint_path}_sample.pth")
                best_reward = ep_ret
            model.save_checkpoint(path=f"/ckpt/{args.checkpoint_path}_running.pth")

            epochs.set_description(f"Step {t} | Reward: {ep_ret:.2f}")
            inventory_actions = np.asarray(inventory_actions).reshape(-1, 6)
            order_actions = np.asarray(order_actions).reshape(-1, 2)

            SR, PC, TC, OC, HC, UP, UD, VC, OO = 0, 0, 0, 0, 0, 0, 0, 0, 0
            inventory_actions = []
            order_actions = []
else:
    env = env(version=args.version)
    if args.algo == "sac":
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
            device=device,
        ).to(device)
    if args.algo == "sac_e2e":
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
            device=device,
        ).to(device)

    else:
        raise ValueError("Invalid algorithm")

    model.load_checkpoint(path=f"/ckpts/{args.checkpoint_path}_running.pth")

    actions = []
    state, ep_ret, ep_len = env.reset(), 0, 0
    episode_rewards = []
    total_steps = 30 * 10
    inventory_actions = []
    SR, PC, TC, OC, HC, UP, UD, SU = 0, 0, 0, 0, 0, 0, 0, 0

    for t in range(total_steps):
        inventory_act, order_act, combined_action = model.select_action(
            state, deterministic=True
        )

        main_edges_act, error = solveLCP(
            env, inventory_act, CPLEXPATH=args.cplexpath, directory=args.directory
        )

        prod_action = {i: int(order_act) for i in env.factory}

        next_state, reward, done, info = env.step(
            prod_action=prod_action, distr_action=main_edges_act
        )

        state = next_state

        ep_ret += reward

        SR += info["sales_revenue"]
        PC += info["purchasing_costs"]
        TC += info["transportation_costs"]
        OC += info["operating_costs"]
        HC += info["holding_costs"]
        UP += info["unfulfilled_penalty"]
        UD += info["unfulfilled_demand"]

        if done:
            episode_rewards.append(ep_ret)
            print(f"Episode reward: {ep_ret}")
            state, ep_ret, ep_len = env.reset(), 0, 0

    print("Profit: ", SR - PC)
    print("purchasing costs", PC)
    print("Revenue ", SR)
    print("Transportation Costs: ", TC)
    print("Operating Costs: ", OC)
    print("Holding Costs: ", HC)
    print("Unfulfilled Penalty: ", UP)
    print("Unfulfilled Demand: ", UD)
    print(np.mean(episode_rewards))
