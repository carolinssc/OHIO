import pickle
import numpy as np
import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn 
from robosuite_ohio.utils.transform_utils import quat2axisangle

class ObservationConditionedRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(ObservationConditionedRNN, self).__init__()
        self.input_dim = input_dim 

        self.rnn = nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)  
        self.tanh = nn.Tanh()  

    def forward(self, inputs, h_0, c_0):
    
        out, hidden = self.rnn(inputs, (h_0, c_0))

        out = self.fc(out)
   
        out = self.tanh(out)*80
        return out, hidden


path = "data/data_door_osc_pose_replay.pkl"

#load pickle file
import pickle
import numpy as np
with open(path, "rb") as f:
    data = pickle.load(f)
print(data.keys())

qpos = data['q_poses']
qpos = np.array(qpos)   
qvel = data['q_vels']
qvel = np.array(qvel)

print(qpos.shape)
qpos = qpos.reshape(1500, 500, 6, -1)
qvel = qvel.reshape(1500, 500, 6, -1)

qpos = np.concatenate([qpos, qpos[:, -1, :, :].reshape(1500, 1,6,-1)], axis = 1)
qvel = np.concatenate([qvel, qvel[:, -1, :, :].reshape(1500, 1,6,-1)], axis = 1)

goal_pos = qpos[:, 1:, 0, :7] - qpos[:, :-1, 0, :7]
goal_vel = qvel[:, 1:, 0, :7] - qvel[:, :-1, 0, :7]
goal = np.concatenate([goal_pos,goal_vel], axis =-1) #goal is the difference between the current and previous joint position and velocity

goal_repeated = np.repeat(goal, repeats=5, axis=1) 


torques_data = data['torques']
torques_data = np.array(torques_data)


goal_repeated = goal_repeated.reshape(1500, 500, 5, 14)

qpos = np.concatenate([np.sin(qpos[:, :,:,:7]), np.cos(qpos[:, :,:,:7])], axis=-1)
rnn_inputs = np.concatenate([qpos[:, :-1, :-1, :],qvel[:, :-1, :-1, :7], goal_repeated], axis=-1) 

rnn_inputs = rnn_inputs.reshape(-1, 5, 35)


validation_split = 0.1
total_size = len(rnn_inputs)
val_size = int(total_size * validation_split)
train_size = total_size - val_size

dataset = TensorDataset(torch.tensor(rnn_inputs, dtype= torch.float32), torch.tensor(torques_data[:, :, :7], dtype= torch.float32))
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
num_epochs = 1
best_val = 1000

device = torch.device('cuda')
model = ObservationConditionedRNN(input_dim=35, hidden_dim=400, output_dim=7).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

train_losses = []
val_losses = []
for epoch in range(100):
    model.train()
    train_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        h_0 = torch.zeros(model.rnn.num_layers, inputs.size(0), model.rnn.hidden_size).to(device)
        c_0 = torch.zeros(model.rnn.num_layers, inputs.size(0), model.rnn.hidden_size).to(device)
        outputs, _ = model(inputs, h_0, c_0)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(dataloader)
    train_losses.append(avg_train_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_train_loss:.4f}')
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            h_0 = torch.zeros(model.rnn.num_layers, inputs.size(0), model.rnn.hidden_size).to(device)
            c_0 = torch.zeros(model.rnn.num_layers, inputs.size(0), model.rnn.hidden_size).to(device)
            outputs, _ = model(inputs, h_0, c_0)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')
    if avg_val_loss < best_val:
        torch.save(model.state_dict(), 'ckpt/RNN_door_joint_2.pth')
        best_val = avg_val_loss
    