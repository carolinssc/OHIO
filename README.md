# Official implementation of Offline Hierarchical Reinforcement Learning via Inverse Optimization (OHIO)

All experiments can be found in their corresponding folders: 
- `goal_directed`
- `robotic_manipulation`
- `supply_chain_management`
- `vehicle_routing`

#  Robotics
To install all required dependencies, run
```
pip install -r requirements.txt
```
Generally, these set of experiments require a working pytorch and mujoco installation. 
##  Goal-directed Control
In this folder, we provide code files for the reacher LQR experiments. 
## Contents

* `SAC_lqr.py`: Train hierarchical online SAC agent.
* `SAC_e2e.py`: Train e2e online SAC agent.
* `IQL_lqr.py`: Train hierarchical agent offline (IQL or BC).
* `IQL_e2e.py`: Train e2e agent offline (IQL or BC).
* `inverse_opt_A.py`: Generating high-level dataset (Analytical).
* `inverse_opt_reg_A.py`: Generating high-level dataset (Reg. Analytical).
* `inverse_opt_N.py`:  Generating high-level dataset (Numerical).
* `learn_A_B_dynamics.py`: PyTorch implementation for learning a MLP dynamics model.
* `networks.py`:  PyTorch implementation of shared actor and critic networks.
* `utils.py`: Helper functions.
* `data/`: Directory for saved data.
* `ckpt/`: Model checkpoints.

## Examples
We provide datasets (both original and created by the different inverse methods of OHIO) and pre-trained model checkpoints at https://drive.google.com/drive/folders/1pP421W_dQ8PFChNyGQbym3uzSK6CDunR?usp=sharing. 
After downloading, please move the datasets to '/goal_directed/data/' and the model checkpoints to '/goal_directed/ckpt/'.

To test the pre-trained E2E agent (on E2E data) run the following: 
```
python IQL_e2e.py --path reacher_e2e_2 --test True
```

To test the pre-trained OHIO agent (on E2E data) run the following for inv_method $\in$ {N, reg_A, A}:
```
python IQL_lqr.py --path reacher_{inv_method} --test True
```

To train an agent run: 
```
python IQL_lqr.py --path {agent_name} --data_path data_reacher_expert_{inv_method}
```

All training parameters are kept at their default values from the paper accross all implementations. 

##  Robotic manipulation
Our robotic manipulation experiments are built on the robosuite package (https://github.com/ARISE-Initiative/robosuite). We have extended this package and therefore provide our own version in this repository. Specifically, we have extended the controller implementation to be differentable in pytorch and have added lower-level learned controller (HRL baselines).

We provide all datasets (both original and created by f OHIO) and pre-trained model checkpoints at https://drive.google.com/drive/folders/1pP421W_dQ8PFChNyGQbym3uzSK6CDunR?usp=sharing. 
After downloading, please move the dataset to '/robotic_manipulation/data' and the model checkpoints to '/robotic_manipulation/ckpt/'.
###  Contents
The folder structure is as follows: 

* `IQL.py`: PyTorch implementation of an offline RL agent (IQL)
* `main_IQL.py`: Main file to train/test an offline RL agent.
* `inverse_GD.py`: Generating high-level dataset (Numerical with gradient descent).
* `inverse_cem.py`:  Generating high-level dataset (Numerical with cross-entropy method).
* `networks.py`:  PyTorch implementation of shared actor and critic networks.
* `robosuite_ohio/`: Modified robosuite package.
* `utils.py`: Helper functions.
* `data/`: Directory for saved data.
* `ckpt/`: Model checkpoints.

## Examples

### Training
When training, the appropriate controller and its parameters are automatically loaded according to the selected dataset. To train an agent, the script main_IQL.py accepts the following arguments:
```bash
model arguments:
    --data_path       data_{task}_{controller}_{parameter_setting}
    --path            where to save model checkpoints
```
Each dataset follows a consistent naming structure:

`data_{task}_{controller}_{parameter_setting}`
#### Task Selection
For `{task}`, choose from:
- `door`
- `lift`

#### Controller Options
For `{controller}`, choose from:
- `osc_pose` (OSC)
- `rnn_joint` (HRL with goal in joint space)
- `rnn_osc` (HRL with goal in operational space)

#### Parameter Settings
For `{parameter_setting}`, choose from:
- `replay` (original dataset)
- `ori` (original dataset recovered by OHIO)
- `kp` (changed stiffness, recovered by OHIO)
- `kd` (changed damping, recovered by OHIO)

To train an IQL agent on the original dataset for the door task run: 
```
python main_IQL.py --data_path data_door_osc_pose_replay --path {agent_name} --quantile 0.9 -temperature 0.1
```
To train an IQL agent on the dataset recovered by OHIO for the door task with changed stiffness, run: 
```
python main_IQL.py --data_path data_door_osc_pose_kp --path {agent_name} --quantile 0.9 -temperature 0.1
```
To train an HRL agent for the door task run: 
```
python main_IQL.py --data_path data_door_rnn_joint --path {agent_name} --quantile 0.9 -temperature 0.1
```
All training parameters are kept at their default values, except for "lift" we use quantile=0.7, temperature=3, whereas for "door", we use quantile=0.9, temperature=0.1 across all dataset. 
### Testing

Similarly, for testing the correct controller and controller settings are called depending on the agent name. To test a pre-trained agent, main_IQL.py takes the following arguments: 

```bash
model arguments:
    --test       True
    --path       IQL_{task}_{algorithm}_{parameter_setting}
```
All model checkpoint can be accessed at /ckpt. Each checkpoint follows a consistent naming structure:

`IQL_{task}_{algorithm}_{parameter_setting}`
#### Task Selection
For `{task}`, choose from:
- `door`
- `lift`

#### Controller Options
For `{algorithm}`, choose from:
- leave out for normal IQL agent
- `ohio` (OHIO agent)
- `rnn_joint` (HRL with goal in joint space)
- `rnn_osc` (HRL with goal in operational space)

#### Parameter Settings
For `{parameter_setting}`, choose from:
- `replay` (original IQL agent)
- `ori` (trained on original dataset recovered by OHIO)
- `kp` (trained on changed stiffness, recovered by OHIO)
- `kd` (trained on changed damping, recovered by OHIO)

To test a pre-trained IQL agent trained on the original dataset for the door task run: 
```
python main_IQL.py --path IQL_door_replay --test True
```
To test a pre-trained IQL agent on the dataset recovered by OHIO for the door task with changed stiffness, run: 
```
python main_IQL.py --path IQL_door_ohio_kp --test True
```
To test a pre-trained HRL agent for the door task run: 
```
python main_IQL.py --path IQL_door_rnn_joint --test True
```


# Network Optimization

You will need to have a working IBM CPLEX installation. If you are a student or academic, IBM is releasing CPLEX Optimization Studio for free. You can find more info [here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students)

If you want to create datasets, you will also need a gurobi license. 

**Important**: Take care of specifying the correct path for your local CPLEX installation. Typical default paths based on different operating systems could be the following
```bash
Windows: "C:/Program Files/ibm/ILOG/CPLEX_Studio128/opl/bin/x64_win64/"
OSX: "/Applications/CPLEX_Studio128/opl/bin/x86-64_osx/"
Linux: "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
```

The code for the two applications (vehicle_routing and supply_chain) can be found in the corresponding folders. 

#  Dynamic Vehicle Routing
To install all required dependencies, run
```
pip install -r requirements.txt
```

## Contents

* `src/algos/`: PyTorch implementation of Graph Neural Networks for SAC, CQL, IQL and BC
* `src/algos/reb_flow_solver.py`: thin wrapper around CPLEX formulation of the Minimum Rebalancing Cost problem.
* `src/envs/amod_env.py`: AMoD simulator.
* `src/cplex_mod/`: CPLEX formulation of Rebalancing and Matching problems.
* `src/misc/`: helper functions.
* `data/`: json files for the simulator of the cities.
* `saved_files/`: directory for saving results, logging, etc.
* `ckpt/`: model checkpoints.
* `Replaymemories/`: datasets for offline RL.
* `main_X.py`: main file to train algorithm X. 
* `create_dataset.py`: Collect datasets used in the paper (INF, DTV, PROP, DISP)

## Examples

To train an agent online, `main_SAC.py` for the bi-level agent and `main_SAC_e2e.py` for the end-to-end agent accept the following arguments:
```bash
cplex arguments:
    --cplexpath     defines directory of the CPLEX installation
    
model arguments:
    --test            activates agent evaluation mode (default: False)
    --max_episodes    number of episodes (default: 10000)
    --max_steps       number of steps per episode (default: T=20)
    --hidden_size     node embedding dimension (default: 256)
    --no-cuda         disables CUDA training (default: True, i.e. run on CPU)
    --directory       defines directory where to log files (default: saved_files)
    --batch_size      defines the batch size (default: 100)
    --alpha           entropy coefficient (default: 0.3)
    --p_lr            Actor learning reate (default 1e-3)
    --q_lr            Critic learning rate (default: 1e-3)
    --checkpoint_path path where to log or load model checkpoints
    --city            which city to train on 
    --rew_scale       reward scaling (default 0.01, for SF 0.1)
    --critic_version  defined critic version to use (default: 4)

simulator arguments: (unless necessary, we recommend using the provided ones)
    --seed          random seed (default: 10)
    --json_tsetp    (default: 3)
```
To collect a dataset `create_dataset.py` accepts the following arguments: 
```
    
model arguments:
    --Heuristic       choice is from INF, DTV, PROP, DISP
    --roh             roh parameter for INF model 
    --max_reb         max_reb parameter for INF model
    --city            for which city to collect the dataset
```

To train an agent offline, `main_CQL.py` and `main_CQL_e2e.py` accept the following arguments (additional to main_SAC):
```
    
model arguments:
    --memory_path     path, where the offline dataset is saved
    --min_q_weight    conservative coefficient (default: 1)
    --samples_buffer  number of samples to take from the dataset (max 10000)
    --lagrange_tresh  lagrange treshhold tau for autonamtic tuning of conservative coefficient
    --st              whether to standardize data (default: False)
    --sc              whether to scale (max-min) the data (default: Fasle)     
```

### Training and simulating an agent online
We tested our algorithms for following cities, i.e. city_name = {shenzhen_downtown_west, nyc_brooklyn} 

1. To train an agent online:
```
python main_SAC.py --city {city_name}
```
### Training an agent offline

1. To train an agent offline: 
```
python main_CQL.py --city city_name --memory_path dataset_name
python main_IQL.py --city city_name --memory_path dataset_name
```
for bevaviour cloning: 
```
python main_IQL.py --city city_name --memory_path dataset_name --max_episodes 0 --bc_steps 20000
```
e.g. to train an agent offline on the DTV dataset for new york: 
```
python main_X.py --city nyc_brooklyn --memory_path Replaymemory_nyc_brooklyn_DTV_distr
```

### Online fine-tuning 
For the online fine-tuning, we use the same hyperparameters as for online SAC with the exception that during training, we sample 25% of the batch from the offline dataset the Cal-CQL agent was trained on and 75% from the online replay buffer.

1. To train an agent with Cal-CQL 
```
python main_Cal_CQL.py --city city_name --memory_path dataset_name --enable_cql True
```
2. To fine-tune a pretrained agent online run the following: 
```
python main_Cal_CQL.py --city city_name --memory_path dataset_name --fine_tune True --load_ckpt pre_trained_checkpoint 
```


# Supply Chain Management
To install all required dependencies, run
```
pip install -r requirements.txt
```

## Contents

* `src/algos/`: PyTorch implementation of Graph Neural Networks for SAC, CQL, IQL and BC
* `src/envs/supply_chain_env.py`: Supply chain simulator.
* `src/cplex_mod/`: CPLEX formulation of lower-level optimizer and MPC baseline
* `src/misc/`: helper functions.
* `saved_files/`: directory for saving results, logging, etc.
* `ckpt/`: model checkpoints.
* `Replaymemories/`: directory for datasets
* `main_offline.py`: main file to train offline algorithms 
* `create_dataset.py`: To collect datasets used in the paper (MPC, HEUR)
* `main_fine_tune.py`: main file to fine-tune offline trained bi-level agent
* `main_fine_tune_e2e.py`: main file to fine-tune offline trained E2E agent
* `main_sac.py`: main file to train online SAC

## Examples
The environment takes a version parameter to pick the scenario: 

`version==1`: Toy example with two nodes (1W1S)

`version==2`: 1W3S from paper 

`version==3`: 1W10S from paper

To train an agent online, `main_sac.py` accept the following arguments:
```bash
cplex arguments:
    --cplexpath     defines directory of the CPLEX installation
    
model arguments:
    --test            activates agent evaluation mode (default: False)
    --max_episodes    number of episodes (default: 10000)
    --hidden_size     node embedding dimension (default: 256)
    --no-cuda         disables CUDA training (default: True, i.e. run on CPU)
    --directory       defines directory where to log files (default: saved_files)
    --batch_size      defines the batch size (default: 100)
    --alpha           entropy coefficient (default: 0.3)
    --p_lr            Actor learning reate (default 1e-3)
    --q_lr            Critic learning rate (default: 1e-3)
    --checkpoint_path path where to log or load model checkpoints
    --version         which scenario to train on (default: 2) 
    --rew_scale       reward scaling 

simulator arguments: (unless necessary, we recommend using the provided ones)
    --seed          random seed (default: 10)
    --json_tsetp    (default: 3)
```

To train an agent offline, `main_offline.py` accept the following arguments (additional to main_SAC):
```
    
model arguments:
    --algo            Choose from 'CQL', 'CQL_e2e', 'IQL', 'IQL_e2e', 'BC', 'BC_e2e'
    --memory_path     path, where the offline dataset is saved
    --min_q_weight    conservative coefficient for CQL (default: 1)
    --samples_buffer  number of samples to take from the dataset (max 20000) 
```

To collect a dataset `create_dataset.py` accepts the following arguments: 
```
    
model arguments:
    --Heuristic       Choice is from MPC and SPolicy
    --version         Scenario version 
    --checkpoint_path Dataset name
```

### Training and simulating an agent online

1. To train an agent online:
```
python main_SAC.py --version {scenario_version}
```
### Training an agent offline

1. To train an agent offline, e.g. IQL: 
```
python main_offline.py --version {scenario_version} --memory_path dataset_name --algo IQL
```

### Online fine-tuning 

1. To fine-tune a pretrained bi-level agent online run the following: 
```
python main_fine_tune.py --version {scenario_version} --memory_path dataset_name --checkpoint_path pre_trained_checkpoint 
```

