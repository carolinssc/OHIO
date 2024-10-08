import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import Actor, Critic, Vf
import argparse
import pickle
from dm_control import suite

class ReplayData:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, device):
        self.device = device
        self.data_list = []
        self.rewards = []

    def create_dataset(self, path):
        path = f"data/{path}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)

        states = np.array(data["state"])
        actions = np.array(data["action"])
        rewards = np.array(data["reward"])
        next_states = np.array(data["next_state"])

        self.data = dict()
        self.data["states"] = states
        self.data["next_states"] = next_states
        self.data["actions"] = actions
        self.data["rewards"] = rewards  

        self.size = len(self.data["rewards"])
        print(f"created dataset with {self.size} samples")
    
    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=self.data["states"][idxs],
            next_state=self.data["next_states"][idxs],
            action=self.data["actions"][idxs],
            reward=self.data["rewards"][idxs],
        )
        return batch

class IQL(nn.Module):
    """
    Soft Actor Critic algorithm
    """

    def __init__(
        self,
        env,
        act_limit,
        alpha=0.3,
        gamma=0.99,
        polyak=0.995,
        batch_size=256,
        lr=1e-3,
        device=torch.device("cuda"),
    ):
        super(IQL, self).__init__()
        self.env = env
        self.gamma = gamma
        self.device = device
        self.alpha = alpha
        self.polyak = polyak
        self.env = env
        self.BATCH_SIZE = batch_size
        self.lr = lr
        self.num_test_episodes = 1
        self.num_random = 10
        self.temperature = 3
        self.quantile = 0.7
        
        obs_dim =6
        act_dim = env.action_spec().shape[0]

        self.act_limit = act_limit.to(self.device)
        # Networks
        print(obs_dim)
        print(act_dim)
        self.actor = Actor(
            obs_dim, act_dim, self.act_limit, hidden_sizes=(256, 256)
        ).to(self.device)
        print(self.actor)
        self.critic1 = Critic(obs_dim, act_dim, hidden_sizes=(256, 256)).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim, hidden_sizes=(256, 256)).to(self.device)
        assert self.critic1.parameters() != self.critic2.parameters()
        self.vf = Vf(obs_dim, hidden_sizes=(256, 256)).to(self.device)
        # Target networks
        self.critic1_target = Critic(obs_dim, act_dim, hidden_sizes=(256, 256)).to(
            self.device
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Critic(obs_dim, act_dim, hidden_sizes=(256, 256)).to(
            self.device
        )
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.optimizers = self.configure_optimizers()
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def select_action(self, state, deterministic=False):
        state = torch.as_tensor(
            state, device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            action, _ = self.actor(state, deterministic)
        return action.detach().cpu().numpy()[0]


    def update(self, batch):
        """
        IQL update
        """
        state_batch = torch.FloatTensor(batch["state"]).to(self.device)
        next_state_batch = torch.FloatTensor(batch["next_state"]).to(self.device)
        action_batch = torch.FloatTensor(batch["action"]).to(self.device)
        reward_batch = torch.FloatTensor(batch["reward"]).to(self.device)

        q1_pred = self.critic1(state_batch, action_batch)
        q2_pred = self.critic2(state_batch, action_batch)

        target_vf_pred = self.vf(next_state_batch).detach()

        with torch.no_grad():
            q_target = reward_batch + self.gamma * target_vf_pred

        q_target = q_target.detach()

        loss_q1 = F.mse_loss(q1_pred, q_target)
        loss_q2 = F.mse_loss(q2_pred, q_target)

        q1_pi_targ = self.critic1_target(next_state_batch, action_batch)
        q2_pi_targ = self.critic2_target(next_state_batch, action_batch)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).detach()
        vf_pred = self.vf(state_batch)

        vf_err = q_pi_targ - vf_pred

        weight = torch.where(vf_err > 0, self.quantile, (1 - self.quantile))
        vf_loss = weight * (vf_err**2)
        vf_loss = vf_loss.mean()

        G = self.actor(state_batch, return_D=True)

        log_prob = G.log_prob(action_batch).sum(axis=-1)

        adv = q_pi_targ - vf_pred

        exp_adv = torch.exp(adv * self.temperature)

        exp_adv = torch.clamp(exp_adv, max=100)

        weights = exp_adv.detach()

        loss_pi = -(log_prob * weights).mean()

        self.optimizers["c1_optimizer"].zero_grad()
        loss_q1.backward()
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward()
        self.optimizers["c2_optimizer"].step()

        self.optimizers["v_optimizer"].zero_grad()
        vf_loss.backward()
        self.optimizers["v_optimizer"].step()

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False


        self.optimizers["a_optimizer"].zero_grad()
        loss_pi.backward(retain_graph=False)
        self.optimizers["a_optimizer"].step()

        # Unfreeze Q-networks
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True  
    
    def update_bc(self, batch): 
        """behavioral cloning update
        """
        state_batch = torch.FloatTensor(batch["state"]).to(self.device)
        action_batch = torch.FloatTensor(batch["action"]).to(self.device)

        G = self.actor(state_batch, return_D=True)
        
        log_prob = G.log_prob(action_batch).sum(axis=-1)
    
        loss_pi = -(log_prob).mean()

        self.optimizers["a_optimizer"].zero_grad()
        loss_pi.backward()
        self.optimizers["a_optimizer"].step()

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())
        vf_params = list(self.vf.parameters())
        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=self.lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=self.lr)
        optimizers["v_optimizer"] = torch.optim.Adam(vf_params, lr=self.lr)
        return optimizers

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model"])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)

    def flatten_observation(self, observation):

        if isinstance(observation, collections.OrderedDict):
            keys = observation.keys()
        else:
            # Keep a consistent ordering for other mappings.
            keys = sorted(observation.keys())

        observation_arrays = [observation[key].ravel() for key in keys]
        return np.concatenate(observation_arrays)
    
    def test_agent(self, env):
        
        time_step = env.reset()
        ep_ret = 0
    
        state = self.flatten_observation(time_step.observation)
      
        while not time_step.last():

            rl_action = rl_agent.select_action(state, deterministic=True)

            time_step = env.step(rl_action)

            reward = time_step.reward

            next_state = self.flatten_observation(time_step.observation)
            state = next_state
            ep_ret += reward

        return ep_ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--path",
        type=str,
        default="reacher",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data_e2e_reacher_expert" 
    )

    args = parser.parse_args()
    num_episodes = 1000
    max_ep_len = 1000
    steps_per_epoch = 1000
    start_steps = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    PATH = f"ckpt/BC_{args.path}.pth"

    env = suite.load("reacher", "hard")
    
    obs_dim = 6
    act_dim = 2

    act_limit = torch.tensor([1,1])
    
    print("Observation space:", obs_dim)
    print("Action space:", act_dim)
    print("Action limits:", act_limit)

    replay_buffer = ReplayData(device)
    replay_buffer.create_dataset(args.data_path)
    if not args.test:
        config = {"directory": args.path}
       
        rl_agent = IQL(
            env, act_limit, batch_size=100, device=device
        )

        total_steps = steps_per_epoch * num_episodes
        rew = 0
        max_ep_len = 1000
        print("start training")
        for i in range(total_steps):
            
            data = replay_buffer.sample_batch(100)
            rl_agent.update_bc(data)

            if i % 10 == 0 and i != 0:
                test_rew = rl_agent.test_agent(env=env)
                print(i, test_rew)
                rl_agent.save_checkpoint(PATH)
    
    else:
        num_episodes = 30
        
        rl_agent = IQL(
            env, act_limit, batch_size=100, device=device
        )

        rl_agent.load_checkpoint(PATH)

        rewards_test = []

        for _ in range(num_episodes):
            ep_ret = rl_agent.test_agent(env=env)
            print(ep_ret)
            rewards_test.append(ep_ret)
       
       print('Mean Reward:', np.mean(rewards_test), 'Std Reward:', np.std(rewards_test))