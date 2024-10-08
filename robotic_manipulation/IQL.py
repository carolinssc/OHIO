import numpy as np
import collections
import torch
import argparse
import pickle
from networks import Actor, Critic, Vf
from utils import parse_dataset_path, setup_controller, setup_controller_params
import robosuite_ohio as suite
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class ReplayData:
    """
    A simple FIFO experience replay buffer for IQL agents.
    """

    def __init__(self, device):
        self.device = device
        self.data_list = []
        self.rewards = []

    def create_dataset(self, data_path):
        data_path = f"data/{data_path}.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    
        states = np.array(data["state"])
        actions = np.array(data["action"])
    
        if 'scaled_z' in data.keys():
            zs = np.array(data["scaled_z"]) #for OHIO
        elif 'scaled_goal' in data.keys():
            zs = np.array(data["scaled_goal"]) #for HRL
        elif 'action' in data.keys():
            zs = np.array(data["action"][:, :-1]) # for IQL
        else: 
            raise ValueError("No goal state found in the data") 

        #assert that no absolute value of z is greater than 1
        assert np.all(np.abs(zs) <= 1)
        actions = np.concatenate([zs, actions[:, -1].reshape(-1, 1)], axis=1)
        next_states = np.array(data["next_state"])
        rewards = np.array(data["reward"])
        dones = np.array(data["done"])

        self.data = {}
        self.data["states"] = states
        self.data["next_states"] = next_states
        self.data["actions"] = actions
        self.data["rewards"] = rewards.reshape(-1)  
        self.data["dones"] = dones.reshape(-1)  

        self.size = len(self.data["rewards"])
        print(f"created dataset with {self.size} samples")
    
    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=self.data["states"][idxs],
            next_state=self.data["next_states"][idxs],
            action=self.data["actions"][idxs],
            reward=self.data["rewards"][idxs],
            done=self.data["dones"][idxs],
        )
        return batch

class IQL(nn.Module):
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
        lr=1e-3,
        device=torch.device("cuda"),
        obs_dim = 4, 
        act_dim=4, 
        quantile = 0.7, 
        temperature = 3,
        
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
        self.num_random = 1
        self.temperature = temperature
        self.quantile = quantile
        self.clip = 1
        
        # Networks
        self.actor = Actor(obs_dim, act_dim, hidden_sizes=(256, 256)).to(self.device)
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
        state_batch = torch.FloatTensor(batch["state"]).to(self.device)
        next_state_batch = torch.FloatTensor(batch["next_state"]).to(self.device)
        action_batch = torch.FloatTensor(batch["action"]).to(self.device)
        reward_batch = torch.FloatTensor(batch["reward"]).to(self.device)
        done_batch = torch.FloatTensor(batch["done"]).to(self.device)

        q1_pred = self.critic1(state_batch, action_batch)
        q2_pred = self.critic2(state_batch, action_batch)

        target_vf_pred = self.vf(next_state_batch).detach()

        with torch.no_grad():
            q_target = reward_batch + (1 - done_batch) * self.gamma * target_vf_pred
        
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
        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
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

        # one gradient descent step for policy network
        self.optimizers["a_optimizer"].zero_grad()
        loss_pi.backward(retain_graph=False)
        self.optimizers["a_optimizer"].step()
        
        # Unfreeze Q-networks
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True  
    
    def update_bc(self, batch): 
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
        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=1e-4)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=3e-4)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=3e-4)
        optimizers["v_optimizer"] = torch.optim.Adam(vf_params, lr=3e-4)
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

    def test_agent(self, env, kp, kd):
        observation = env.reset()
    
        state = np.hstack([observation['robot0_proprio-state'], observation['object-state']]) 
        i = 0
        ep_ret = 0 
        done = False
        while not done and i <500: 
            
            action = self.select_action(state, deterministic=True)

            if kp is not None:
                action = np.concatenate([kd, kp, action]) 

            observation, reward, done, info = env.step(action, verbose=False)
            
            next_state = np.hstack([observation['robot0_proprio-state'], observation['object-state']])
            ep_ret += reward

            state = next_state
            i+=1
        return ep_ret