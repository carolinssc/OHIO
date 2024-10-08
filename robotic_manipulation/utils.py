import numpy as np
from robosuite_ohio import load_controller_config

def parse_dataset_path(dataset_path):
    # Assuming the path is always well-formed and starts with 'data_'
    parts = dataset_path.split('_')
    parts[-1] = parts[-1].replace('.pkl', '')
    controller = f"{parts[2]}_{parts[3]}"
    task = parts[1]
   
    subtask = parts[4] if len(parts) > 4 else None
    return controller, task, subtask

def parse_checkpoint_path(checkpoint_path):

    if 'ohio' in checkpoint_path.lower(): 
        controller = 'osc_pose'
    elif 'rnn_joint' in checkpoint_path.lower(): 
        controller = 'rnn_joint'
    elif 'rnn_osc' in checkpoint_path.lower():
        controller = 'rnn_osc'
    else:
        controller = 'osc_pose'

    parts = checkpoint_path.split('_')
    task = parts[1]
    parts[-1] = parts[-1].replace('.pkl', '')
    if parts[-1] == 'kd' or parts[-1] == 'kp' or parts[-1] == 'ori' or parts[-1] == 'replay':
        subtask = parts[-1]
    else:
        subtask = None
    return controller, task, subtask
    
def setup_controller(controller, task, subtask): 
    if subtask is None or subtask == 'replay':
        config_name = f"{controller}_{task}"
    else: 
        config_name = f"{controller}_{task}_{subtask}"
    # load controller
    if controller == "osc_pose":
        controller = load_controller_config(default_controller="OSC_POSE",json_path=f"{config_name}")
    elif controller == "rnn_joint":
        controller = load_controller_config(default_controller="RNN_JOINT", json_path=f"{config_name}")
    elif controller == "rnn_osc":
        controller = load_controller_config(default_controller="RNN_OSC", json_path=f"{config_name}")
    else: 
        raise ValueError("Controller not found")
    return controller

def setup_controller_params(subtask):
    #set damping and stiffness 
    if subtask == 'ori':
        kp = np.array([150,150,150,150,150,150])
        kd = np.array([1,1,1,1,1,1])
    elif subtask == 'replay':
        kp = np.array([150,150,150,150,150,150])
        kd = np.array([1,1,1,1,1,1])

    elif subtask == 'kp':
        kp = np.array([150,150,150,50,50,50])
        kd = np.array([1,1,1,1,1,1])
    elif subtask == 'kd':
        kp = np.array([150,150,150,150,150,150])
        kd = np.array([3,3,3,1,1,1])
    elif subtask is None:
        kp = None
        kd = None
    else: 
        raise ValueError("Subtask not found")
    return kp, kd

