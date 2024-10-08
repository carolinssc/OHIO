import pickle
import numpy as np
import torch
import torch.nn as nn
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
        print(j)
        for i in range(0, len(states1[j]) - 5, 5):
            s = states1[j, i, [0, 1, 4, 5]]
            s1 = states1[j, i + 5, [0, 1, 4, 5]]
            
            K, A, B = LQR(
                env.physics.model.ptr,
                s,  # change
                5,
            )
            A, B = get_A_B_learned(model_AB, s.astype(np.float32))
            #A, B = get_matrices(env.physics.model.ptr, s) #uncomment this line to use the finite-differencing dynamics instead of the learned dynamics

            A = np.asarray(A)
            B = np.asarray(B)
            term1 = B @ K[0]
            term2 = (A + B @ K[0]) @ s
            for l in range(1, 5):
                A, B = get_A_B_learned(
                    model_AB, states1[j, i + l, [0, 1, 4, 5]].astype(np.float32)
                )
                #A, B = get_matrices(env.physics.model.ptr, states1[j, i + l, [0, 1, 4, 5]]) #uncomment this line to use the finite-differencing dynamics instead of the learned dynamics
                A = np.asarray(A)
                B = np.asarray(B)

                term1 = (A + B @ K[l]) @ term1 + B @ K[l]
                term2 = (A + B @ K[l]) @ term2

            term1_inv = np.linalg.pinv(term1)
            z = -term1_inv @ (s1 - term2)
            
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

        with open(f"data/data_preprocessed/reacher_hard_A_{j}.pkl", "wb") as f:
            pickle.dump(data, f)
