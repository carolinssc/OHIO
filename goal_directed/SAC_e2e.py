import random
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import Actor, Critic
import argparse
from dm_control import suite
import collections

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SAC(nn.Module):
    """
    Soft Actor Critic algorithm
    """

    def __init__(
        self,
        env,
        alpha=0.3,
        gamma=0.99,
        polyak=0.995,
        batch_size=256,
        lr=1e-4,
        device=torch.device("cuda"),
        obs_dim=1,
        act_dim=1,
    ):  
        super(SAC, self).__init__()
        self.env = env
        self.gamma = gamma
        self.device = device
        #self.alpha = alpha
        self.polyak = polyak
        self.env = env
        self.BATCH_SIZE = batch_size
        self.lr = lr
        self.num_test_episodes = 1
        self.test_env = suite.load("reacher", "hard")
         
        act_dim = act_dim
        obs_dim = obs_dim
        act_limit = torch.tensor([1,1]).to(self.device)
        self.target_entropy = -act_dim
        # Networks
        self.actor = Actor(obs_dim, act_dim, act_limit, hidden_sizes=(256, 256)).to(
            self.device
        )
        print(self.actor)
        self.critic1 = Critic(obs_dim, act_dim, hidden_sizes=(256, 256)).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim, hidden_sizes=(256, 256)).to(self.device)
        assert self.critic1.parameters() != self.critic2.parameters()

        # Target networks
        self.critic1_target = Critic(obs_dim, act_dim, hidden_sizes=(256, 256)).to(
            self.device
        )
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Critic(obs_dim, act_dim, hidden_sizes=(256, 256)).to(
            self.device
        )
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.log_alpha = torch.tensor(np.log(0.3)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=1e-4)
    
        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.optimizers = self.configure_optimizers()

        self.memory = ReplayMemory(int(1e6))

        self.saved_actions = []
        self.rewards = []
        self.to(self.device)    
        self.learnable_temperature = False

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.as_tensor(
            state, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(state, deterministic)
        return action.detach().cpu().numpy()[0]

    def compute_loss_q(self, batch):

        state_batch = torch.FloatTensor(batch.state).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        action_batch = torch.FloatTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        q1 = self.critic1(state_batch, action_batch)
        q2 = self.critic2(state_batch, action_batch)

        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor(next_state_batch)

            q1_pi_targ = self.critic1_target(next_state_batch, a2)
            q2_pi_targ = self.critic2_target(next_state_batch, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            backup = reward_batch + self.gamma * (1 - done_batch) * (
                q_pi_targ - self.alpha.detach() * logp_a2
            )


        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, batch):
        state_batch = torch.FloatTensor(batch.state).to(self.device)

        actions, logp_a = self.actor(state_batch, False)

        q1_1 = self.critic1(state_batch, actions)
        q2_a = self.critic2(state_batch, actions)
        q_a = torch.min(q1_1, q2_a)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha.detach() * logp_a - q_a).mean()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-logp_a - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()


        return loss_pi

    
    def update(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        # get a batch from the replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        loss_q = self.compute_loss_q(batch)

        self.optimizers["c_optimizer"].zero_grad()
        loss_q.backward()
        self.optimizers["c_optimizer"].step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # one gradient descent step for policy network
        self.optimizers["a_optimizer"].zero_grad()
        loss_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.optimizers["a_optimizer"].step()

        # Unfreeze Q-networks
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

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

    def test_agent(self):
        rewards_test = []
        #for _ in range(self.num_test_episodes):
        time_step = self.test_env.reset()
        o = flatten_observation(time_step.observation)
        ep_ret, ep_len = 0, 0

        while not time_step.last():
            # Take deterministic actions at test time
            time_step = self.test_env.step(self.select_action(o, True))
            r = time_step.reward
            o = flatten_observation(time_step.observation)
            ep_ret += r
            ep_len += 1
        rewards_test.append(ep_ret)
        return np.mean(rewards_test)
    
    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())
        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.lr)
        optimizers["c_optimizer"] = torch.optim.Adam(
            critic1_params + critic2_params, lr=self.lr
        )
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

# %%
def flatten_observation(observation):

    if isinstance(observation, collections.OrderedDict):
        keys = observation.keys()
    else:
        # Keep a consistent ordering for other mappings.
        keys = sorted(observation.keys())

    observation_arrays = [observation[key].ravel() for key in keys]
    return np.concatenate(observation_arrays)

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
        default="SAC",
    )
    args = parser.parse_args()

    num_episodes = 10000
    max_ep_len = 1000
    steps_per_epoch = 1000
    start_steps = 10000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    PATH = f"ckpt/{args.path}.pth"

    env = suite.load("reacher", "hard")
    
    time_step = env.reset()
    obs = flatten_observation(time_step.observation)
    obs_dim = obs.shape[0]
    act_dim = env.action_spec().shape[0]

    saved = 0
    print("Observation space:", obs_dim)
    print("Action space:", act_dim)
  

    if not args.test:

        model = SAC(
            env,
            batch_size=256,
            device=device,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )
       
        total_steps = 1000
        
        time_step = env.reset()
        ep_ret, ep_len = 0, 0

        training_rewards = []

        for step in range(total_steps):
            time_step = env.reset()
            ep_ret, ep_len = 0, 0
    
            state = flatten_observation(time_step.observation)
        
            while not time_step.last():
                
                action = model.select_action(state)

                time_step = env.step(action)

                reward = time_step.reward
                next_state = flatten_observation(time_step.observation)

                ep_ret += reward
                ep_len += 1
                
                model.memory.push(state, action, next_state, reward, False)

                state = next_state

                model.update()
            print(ep_ret)


            if step % 1000 == 0:
                test_rewards = model.test_agent()

            
            model.save_checkpoint(PATH)
