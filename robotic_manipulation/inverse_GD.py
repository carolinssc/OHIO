import pickle
import numpy as np
import torch
import torch.optim as optim
import robosuite_ohio as suite
from robosuite_ohio import load_controller_config
import argparse

def GD(goal_torques, qpos, qvel, env, initial_joints, delta, kp, kd): 
    arm_action = torch.tensor(delta, requires_grad=True, dtype=torch.float64)

    optimizer = optim.Adam([arm_action], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.01)
    robot =  env.robots[0]
    controller = robot.controller
    controller.update_initial_joints(initial_joints)
    
    state = {}  
    state['qpos'] = qpos[0, :]
    state['qvel'] = qvel[0, :]

    controller.update_state_from_dataset(state)
    best_loss = np.inf
    c = 0 
    for iteration in range(10000):
        pred_torques = []
        controller.update_state_from_dataset({'qpos': qpos[0, :], 'qvel': qvel[0, :]})

        controller.set_goal_diff(torch.cat([kd, kp, arm_action]))
       
        losses = 0
        for j in range(5):
            controller.update_state_from_dataset({'qpos': qpos[j, :], 'qvel': qvel[j, :]})
            torques, _  = controller.run_controller_diff()
            low, high = robot.torque_limits
            torques = torch.clip(torques, torch.tensor(low), torch.tensor(high))
            pred_torques.append(torques)
            loss = torch.nn.functional.mse_loss(torques, torch.tensor(goal_torques[j]))
            losses += loss
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()
      
        if losses.item() < best_loss:
            if np.abs(losses.item() - best_loss) < 1e-3:
                c += 1
            else: 
                c = 0 
            best_loss = losses.item()
            best_action = arm_action.detach().numpy()
        else: 
            c += 1
        if c > 20:
            break
        if losses.item() < 1e-4:
            break
    return best_action, best_loss, iteration


parser = argparse.ArgumentParser()

parser.add_argument(
    "--j",
    type=int,
    default=0,
    help="starting index in the dataset (for parallel processing)",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="data_door_osc_pose_replay",
    help="path where original dataset is stored",
)

args = parser.parse_args()
j_lim = args.j
dataset_path = f"data/{args.dataset_path}.pkl"
with open(dataset_path, "rb") as f:
    data = pickle.load(f)

qpos = data['q_poses']
qpos = np.array(qpos)   
qvel = data['q_vels']
qvel = np.array(qvel)

next_states = data['next_state']
next_states = np.array(next_states)

torques_data = data['torques']
torques_data = np.array(torques_data)

dones = np.array(data['done'])
actions = np.array(data['action'])

print(qpos.shape)
print(qvel.shape)
print(next_states.shape)
print(torques_data.shape)
print(actions.shape)
print(dones.shape)

controller = load_controller_config(default_controller="OSC_POSE")

config = {
    "controller_configs": controller,
    "horizon": 500,
    "control_freq": 100,
    "reward_shaping": True,
    "reward_scale": 1.0,
    "use_camera_obs": False,
    "ignore_done": True,
    "hard_reset": False,
}

# this should be used during training to speed up training
# A renderer should be used if you're visualizing rollouts!
config["has_offscreen_renderer"] = False

task = dataset_path.split('_')[1]
env = suite.make(
    env_name=task.capitalize(),
    robots="Panda",
    **config,
)

zs = []
loss_items = []
first = True 
c = 0 

output_max = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
output_min = np.array([-0.05, -0.05, -0.05, -0.5, -0.5, -0.5])

input_max = 1
input_min = -1
action_input_transform = (input_max + input_min) / 2.0
action_scale = abs(output_max - output_min) / abs(input_max - input_min)
action_output_transform = (output_max + output_min) / 2.0


kd = torch.tensor([1,1,1,1,1,1]) #change damping
kp = torch.tensor([150, 150, 150, 150, 150, 150]) #change stiffness
        
for i in range(j_lim,j_lim+10000):
    goal_torques = torques_data[i, :, :7]
  
    if first:
        initial_joints = qpos[i, 0, :7]
        assert qvel[i, 0, 0] < 1e-6
    
    pos = qpos[i, :, :]
    vel = qvel[i, :, :]
    
    first = False
    c += 1
    if dones[i]: 
        first = True
        c = 0
    elif c ==500: 
        first = True
        c = 0
    
    ori_act = np.array([0,0,0,0.1,0.1,0.1])
    #uncomment the following lines if the original actions are assumed to be available (much better initailization)
    #ori_act = actions[i,:-1]
    #ori_act = (torch.tensor(ori_act) - torch.tensor(action_input_transform)) * torch.tensor(action_scale) + torch.tensor(action_output_transform)

    z, loss, iteration = GD(goal_torques, pos, vel, env, initial_joints, ori_act, kp, kd)
    
    print(i, loss)

    with open(f"data/data_preprocessed/data_robosuite_{i}_z_{task}.pkl", "wb") as f:
        pickle.dump(z, f)
    with open(f"data/data_preprocessed/data_robosuite_{i}_loss_{task}.pkl", "wb") as f:
        pickle.dump(loss, f)
    