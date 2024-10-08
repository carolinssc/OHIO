import numpy as np
import collections
import torch
import argparse
import pickle
from networks import Actor, Critic, Vf
from utils import parse_dataset_path, setup_controller, setup_controller_params, parse_checkpoint_path
import robosuite_ohio as suite
from IQL import ReplayData, IQL

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
        default="IQL.pth",
        help="path where to save the model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data_door_osc_pose_replay",
        help="path where the data is stored",
    )

    parser.add_argument(
        "--quantile",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=3,
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  
    PATH = f"ckpt/{args.path}.pth"
    
    if not args.test:
        #load controller config file
        controller, task, subtask = parse_dataset_path(dataset_path=args.data_path)
        print(f"loading {controller} controller for task {task}_{subtask}")

        controller_config = setup_controller(controller, task, subtask)
        
        #setup damping and stiffness coefficents
        kp, kd = setup_controller_params(subtask)
        print(f'testing with kp: {kp} and kd: {kd}')

        config = {
            "controller_configs": controller_config,
            "horizon": 500,
            "control_freq": 100,
            "reward_shaping": True,
            "reward_scale": 1.0,
            "use_camera_obs": False,
            "ignore_done": True,
            "hard_reset": False,
        }
        # this should be used during training to speed up training
        config["has_offscreen_renderer"] = False

        env = suite.make(
            env_name=task.capitalize(),
            robots="Panda",
            **config,
        )

        if task =='door':
            obs_dim = 46
        elif task =='lift': 
            obs_dim = 42
        else: 
            raise ValueError("Task not found")
    
        
        if controller == "osc_pose":
            act_dim = 7
        elif controller == "rnn_joint":
            act_dim = 15
        elif controller == "rnn_osc":
            act_dim = 7

        replay_buffer = ReplayData(device)
        replay_buffer.create_dataset(data_path=args.data_path)
        
        print(obs_dim)
        print(act_dim)

        rl_agent = IQL(
            env,
            batch_size=256,
            device=device,
            obs_dim=obs_dim,
            act_dim=act_dim,
            quantile=args.quantile,
            temperature=args.temperature
        )

        total_steps = 1000*args.max_episodes
      
        for step in range(total_steps):
            
            data = replay_buffer.sample_batch(100)
            rl_agent.update(data)

            if step % 1000 == 0:
                test_rew = rl_agent.test_agent(env=env, kp=kp, kd=kd)
                print(step, test_rew)
    
            rl_agent.save_checkpoint(PATH)
        
    else:
        
        controller, task, subtask = parse_checkpoint_path(args.path)
   
        print(f"loading {controller} controller for task {task}_{subtask}")
        controller_config = setup_controller(controller, task, subtask)
        #setup damping and stiffness coefficents
        kp, kd = setup_controller_params(subtask)
        #kp, and kd can be changed here to test the agent with different damping and stiffness values
        # e.g., kp = np.array([150,150,150,50,50,50])
        # e.g., kd = np.array([3,3,3,1,1,1])
        print(f'testing with kp: {kp} and kd: {kd}')
    
        config = {
            "controller_configs": controller_config,
            "horizon": 500,
            "control_freq": 100,
            "reward_shaping": True,
            "reward_scale": 1.0,
            "use_camera_obs": False,
            "ignore_done": True,
            "hard_reset": False,
        }
        # this should be used during training to speed up training
        config["has_offscreen_renderer"] = False
 
        env = suite.make(
            env_name=task.capitalize(),
            robots="Panda",
            **config,
        )

        if task =='door':
            obs_dim = 46
        elif task =='lift': 
            obs_dim = 42
        else: 
            raise ValueError("Task not found")
        
   
        if controller == "osc_pose":
            act_dim = 7
        elif controller == "rnn_joint":
            act_dim = 15
        elif controller == "rnn_osc":
            act_dim = 7
        
        rl_agent = IQL(
            env,
            batch_size=256,
            device=device,
            obs_dim=obs_dim,
            act_dim=act_dim,
            quantile=args.quantile,
            temperature=args.temperature
        )

        num_episodes = 30
    
        rl_agent.load_checkpoint(PATH)
        rewards_test = []
        
        for _ in range(num_episodes):
            ep_ret = rl_agent.test_agent(env=env, kp=kp, kd=kd)
            print(ep_ret)
            rewards_test.append(ep_ret)

        print(rewards_test)
        print(np.mean(rewards_test), np.std(rewards_test))
