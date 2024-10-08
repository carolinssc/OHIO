import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 4 * 4)  # Output size: 4*4 = 16
        self.fc4 = nn.Linear(hidden_size, 4 * 2)  # Output size: 4*2 = 8

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        A = self.fc3(x).view(-1, 4, 4)  # Reshape to (batch_size, 4, 4)
        B = self.fc4(x).view(-1, 4, 2)  # Reshape to (batch_size, 4, 2)
        return A, B


if __name__ == "__main__":

    path = "data/e2e_reacher_expert.pkl" 

    with open(path, "rb") as f:
        data = pickle.load(f)

    states = data["state"]
    states = np.array(states)
    states = states[:, [0, 1, 4, 5]]

    action_lowlevel = data["action"]
    action_lowlevel = np.array(action_lowlevel)

    next_states = states[1:]
    states = states[:-1]
    action_lowlevel = action_lowlevel[:-1]

    states = np.concatenate([states, action_lowlevel], axis=1)

    size = states.shape[0]
    val_size = 0.1 * size
    val_size = int(val_size)
    # random sample indeces for training and validation
    idxs = np.random.randint(0, size, size=val_size)

    X_val = states[idxs]
    Y_val = next_states[idxs]

    states = np.delete(states, idxs, axis=0)
    next_states = np.delete(next_states, idxs, axis=0)

    print(states.shape)
    print(next_states.shape)
    print(X_val.shape)
    print(Y_val.shape)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert numpy arrays to torch tensors
    X_tensor_train = torch.tensor(states, dtype=torch.float32)
    Y_tensor_train = torch.tensor(next_states, dtype=torch.float32)

    X_tensor_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_tensor_val = torch.tensor(Y_val, dtype=torch.float32).to(device)

    # Create a dataset and data loader
    dataset_train = TensorDataset(X_tensor_train, Y_tensor_train)
    dataset_val = TensorDataset(X_tensor_val, Y_tensor_val)
    dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True)


    obs_dim = next_states.shape[1]
    act_dim = 2
    
    # Instantiate the model
    model = MLP(4, 128).to(device)

    # Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training Loop
    model.eval()
    with torch.no_grad():
        s = X_tensor_val[:, :4]
        u = X_tensor_val[:, 4:]
        # Forward pass
        A, B = model(s)

        next_s = torch.matmul(A, s.unsqueeze(-1)).squeeze(-1) + torch.matmul(
            B, u.unsqueeze(-1)
        ).squeeze(-1)
        print(Y_tensor_val[5], next_s[5])
        # Compute loss
        loss = criterion(next_s, Y_tensor_val)
        print(f"Epoch 0, Validation Loss: {loss.item():.4f}")
        
    for epoch in range(20):
        model.train()
        train_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            s = inputs[:, :4]
            u = inputs[:, 4:]
            # Forward pass
            A, B = model(s)

            next_s = torch.matmul(A, s.unsqueeze(-1)).squeeze(-1) + torch.matmul(
                B, u.unsqueeze(-1)
            ).squeeze(-1)

            # Compute loss
            loss = criterion(next_s, targets)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # print(loss.item())
        scheduler.step()
        train_loss = train_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{5}, Loss: {train_loss:.4f}")
        # validation loss
        best_val = np.inf
        model.eval()
        with torch.no_grad():
            s = X_tensor_val[:, :4]
            u = X_tensor_val[:, 4:]
            # Forward pass
            A, B = model(s)

            next_s = torch.matmul(A, s.unsqueeze(-1)).squeeze(-1) + torch.matmul(
                B, u.unsqueeze(-1)
            ).squeeze(-1)
            # Compute loss
            loss = criterion(next_s, Y_tensor_val)
            print(f"Epoch {epoch+1}/{5}, Validation Loss: {loss.item():.4f}")
            if loss.item() < best_val:
                # Optional: Save the trained model
                torch.save(
                    model.state_dict(),
                    "ckpt/A_B_model_e2e_reacher.pth",
                )
                best_val = loss.item()

    print("Training complete!", best_val)
