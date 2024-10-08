import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import argparse
from dm_control import suite
from utils import LQR, get_matrices, get_A_B_learned

class MLP_AB(nn.Module):
    #Learned dynamics model
    def __init__(self, input_size, hidden_size):
        super(MLP_AB, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4 * 4) 
        self.fc4 = nn.Linear(hidden_size, 4 * 2)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        A = self.fc3(x).view(-1, 4, 4)  
        B = self.fc4(x).view(-1, 4, 2) 
        return A, B

def reg_A(model, K, s, s1, states, steps=1000):
    """Reg. Analytical inverse

    Args:
        model: mujoco model of environment
        K: Optimal control gain matrix from LQR
        s: Initial state
        s1: Next state
        states: states in the trajectory
        steps: Number of steps for optimization

    Returns:
        u_desired: Optimized high-level action
        loss: Loss value
    """
    K = torch.tensor(K, dtype=torch.float32, requires_grad=False)
    s = torch.tensor(s, dtype=torch.float32, requires_grad=False)
    s1 = torch.tensor(s1, dtype=torch.float32, requires_grad=False)

    x = torch.zeros(
        (6, 4), requires_grad=False
    )  
    u_desired = nn.Parameter(
        torch.tensor(s1), requires_grad=True
    )
    optimizer = optim.Adam([u_desired], lr=0.01)

    prev_loss = 1000
    c = 0
    for epoch in range(steps): 
        x = torch.zeros((6, 4), requires_grad=False)  
        x[0] = torch.tensor(s)

        optimizer.zero_grad()

        # uncomment the following lines if you want to use the finite-differencing dynamics
        # A,B = get_matrices(env.physics.model.ptr, x[i].detach().numpy(), 5)
        # A = torch.tensor(A, dtype=torch.float32, requires_grad=False)
        # B = torch.tensor(B, dtype=torch.float32, requires_grad=False)

        A, B = get_A_B_learned(model, s.numpy())
     
        term1 = B @ K[0]
        term2 = (A + B @ K[0]) @ s
        for l in range(1, 5):
            A, B = get_A_B_learned(model, states[l])
            term1 = (A + B @ K[l]) @ term1 + B @ K[l]
            term2 = (A + B @ K[l]) @ term2

        loss = -term1.t() @ (s1 - (term2 - term1 @ u_desired))
        loss = torch.norm(loss) ** 2

        # Backpropagation
        loss.backward()

        # Update u_desired
        optimizer.step()

        if epoch % 10 == 0:  
            if np.abs(prev_loss - loss.item()) < 0.000001:
                c += 1
                prev_loss = loss.item()
            else:
                c = 0
                prev_loss = loss.item()
            if c == 2:
                break

    # Final optimized u_desired
    return u_desired.detach().numpy(), loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--j",
        type=int,
        default=0,
    )

    """
    Load the learned dynamics model 
    """
    model_AB = MLP_AB(4, 128)
    model_AB.load_state_dict(
        torch.load("ckpt/A_B_model_e2e_reacher.pth",
            map_location=torch.device("cpu")
        )
    )
    
    """
    Load Dataset
    """
    path = "data/data_e2e_reacher_expert.pkl" 

    with open(path, "rb") as f:
        data = pickle.load(f)

    states = np.array(data["state"])
    actions = np.array(data["action"])
    rewards = np.array(data["reward"])

    states1 = states.reshape(-1, 1000, 6)
    rewards1 = rewards.reshape(-1, 1000)

    env = suite.load("reacher", "hard")
    timestep = env.reset()

    args = parser.parse_args()
    j_lim = args.j #number of trajectories for parallelization

    zs = []
    next_states = []
    states = []
    rews = []

    for j in range(states1.shape[0]):
        for i in range(0, len(states1[j]) - 5, 5):
            print(j, i)
            s = states1[j, i, [0, 1, 4, 5]]
            s1 = states1[j, i + 5, [0, 1, 4, 5]]
            
            K, A, B = LQR( 
                env.physics.model.ptr,
                s,  
                5,
            )

            s_5 = []
            for k in range(5):
                s_5.append(states1[j, i + k, [0, 1, 4, 5]])

            s_5 = np.array(s_5)

            z, loss = reg_A(model_AB, K, s, s1, states=s_5)
            
            if loss > 0.2: #if the loss is too high, don't include transition in dataset
                continue
           
            states.append(states1[j][i])
            next_states.append(states1[j][i + 5])
            zs.append(z)
            r = rewards1[j][i : i + 5].sum()
            rews.append(r)
    
        
        data = {}
        data["state"] = states
        data["z"] = zs #high-level actions
        data["reward"] = rews
        data["next_state"] = next_states

        with open(f"data/data_preprocessed/reacher_hard_reg_A_{j}.pkl", "wb") as f:
            pickle.dump(data, f)
