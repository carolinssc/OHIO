import math

import numpy as np

import robosuite_ohio.utils.transform_utils as T
from robosuite_ohio.controllers.base_controller import Controller
from robosuite_ohio.utils.control_utils import *
import torch.nn as nn 

class ObservationConditionedRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(ObservationConditionedRNN, self).__init__()
        self.input_dim = input_dim   # Concatenated input dimension

        self.rnn = nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True)

        #self.fc0 = nn.Linear(hidden_dim, hidden_dim)  # MLP for action prediction
        self.fc = nn.Linear(hidden_dim, output_dim)  # MLP for action prediction
        self.tanh = nn.Tanh()  # Tanh activation to constrain outputs between -1 and 1

    def forward(self, inputs, h_0, c_0):
    
        out, hidden = self.rnn(inputs, (h_0, c_0))

        #out = self.fc0(out)
        out = self.fc(out)
   
        out = self.tanh(out)*80
        return out, hidden


class RNN_joint_controller(Controller):

    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
        output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
        policy_freq=20,
        position_limits=None,
        orientation_limits=None,
        control_delta=True,
        model_path=None,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):
        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )
        # Determine whether we want to use delta or absolute values as inputs
        self.use_delta = control_delta

        self.position_limits = np.array(position_limits) if position_limits is not None else position_limits
        self.orientation_limits = np.array(orientation_limits) if orientation_limits is not None else orientation_limits

        # control frequency
        self.control_freq = policy_freq
        self.model = ObservationConditionedRNN(input_dim=35, hidden_dim=256, output_dim=7)
        
        #checkpoint = torch.load("/mnt/raid1/csasc_storage/RNN_door_jointvel_400.pth")
        #checkpoint = torch.load("/mnt/raid1/csasc_storage/RNN_door_jointvel_final_200_2.pth")
        #checkpoint = torch.load("/mnt/raid1/csasc_storage/RNN_lift_jointvel_final.pth")
        checkpoint = torch.load(model_path) 
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.control_dim = 14
        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

    
    def set_goal(self, action, set_pos=None, set_ori=None):
        # Update state
        self.update()

        scaled_delta = self.scale_action(action)
        self.goal_pos = scaled_delta

        self.h_0 = torch.zeros(self.model.rnn.num_layers, 1, self.model.rnn.hidden_size)
        self.c_0 = torch.zeros(self.model.rnn.num_layers, 1, self.model.rnn.hidden_size)

        # Initialize sequence buffer
        self.sequence = []

    def run_controller(self):
        
        self.update()

        qpos = np.concatenate([np.sin(self.joint_pos[:7]), np.cos(self.joint_pos[:7])], axis=-1)
        rnn_input = np.concatenate([qpos, self.joint_vel[:7], self.goal_pos], axis=-1)
   
        self.sequence.append(rnn_input)
        rnn_inputs = torch.tensor(self.sequence, dtype=torch.float32).unsqueeze(0)
     
        self.torques, (self.h_0, self.c_0) =  self.model(rnn_inputs, self.h_0, self.c_0)

        self.torques = self.torques[:, -1, :].cpu().detach().numpy().flatten()

        super().run_controller()
        
        return self.torques

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial confguration
        self.reset_goal()

    def reset_goal(self):
        """
        Resets the goal to the current state of the robot
        """
        self.goal_ori = np.array(self.ee_ori_mat)
        self.goal_pos = np.array(self.ee_pos)

    @property
    def control_limits(self):

        low, high = self.input_min, self.input_max
        return low, high

    @property
    def name(self):
        return "RNN_" + self.name_suffix
