import threading
import gym
import highway_env
from matplotlib import pyplot as plt
import pprint
from Keyboard import KeyboardEventHandler
import threading
import numpy as np
from datetime import datetime
from highway_env.vehicle.behavior import IDMVehicle
import string
import model
import torch
import utils
from numpy import random
from policy_gradient import Policy
from TD3 import TD3
import torch.nn.functional as F

f = None

time_steps = 2500
max_action = 0.3
expl_noise = 0.1
batch_size = 256
teraminal_penalty = -100

max_timesteps = int(1e6)
def main():
    global f

    generate_data = False
    num_of_other_vehicles = 10
    env = gym.make('highway-v0')
    # env.config["show_trajectories"] = True
    env.config["vehicles_count"] = num_of_other_vehicles
    env.configure({
        "lanes_count": 5,
        "action": {
            "type" : "ContinuousAction"
        },
        "collision_reward": -100,
        "duration": 600,
        "on_road_reward" : 1,
        "off_road_reward" : -5,
        'offroad_terminal': True,
        'high_speed_reward': 0.1,
        'screen_height': 600,
        'screen_width': 2400,
        # "observation": {
        #     "type": "Kinematics",
        #     "vehicles_count": num_of_other_vehicles,
        #     "features": ["presence", "x", "y", "vx", "vy", "heading","lat_off"],
        #     "absolute": True,
        #     "normalize": False,
        #     "order": "sorted"
        # }

        "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": num_of_other_vehicles,
        # "vehicles_count": 15,
        # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features": ["presence", "x", "y", "lat_off", "heading","ang_off","vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "grid_size": [[-25, 25], [-25, 25]],
        "grid_step": [2.5, 2.5],
        "absolute": False
        }
    })
    if generate_data:
        dir = '/Users/hwpark/Desktop/highway_env/data/'
        date_time = datetime.now()
        dir += date_time.strftime("%Y %m %d %H %M")
        f = open(dir,'w')

    idm_vehicle = None
    obs = None
    

    policy = TD3(401, 2, 1)
    replay_buffer = utils.ReplayBuffer(state_dim = 401, action_dim = 2)

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    state = env.reset()
    state = state[6].reshape(-1)
    ego = env.road.vehicles[0].position
    ego_lane_idx = np.array(env.road.network.get_closest_lane_index(np.array(ego))[2],np.float32)
    state = np.append(state,[ego_lane_idx])
    for t in range(max_timesteps):
        episode_timesteps += 1

        # acceleration and steering angle

        done = False


        if keyboard_listener.is_space_pressed:
            evt.wait()
        if t < episode_timesteps:
            action = [np.random.normal(0, 0.05, 1),np.random.normal(0, 0.05, 1)]
        else:
            action = (policy.select_action(np.array(state))
            + np.random.normal(0, max_action * expl_noise, size=2)
        ).clip(-max_action, max_action)
        
        # action[0] = action[0]
        # action[1] = action[1]
        print(action)
        state_prime, reward, done, info = env.step(action)
        # done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0         
        done_bool = float(done)
        if(done_bool): reward += teraminal_penalty 
        state_prime = state_prime[6].reshape(-1)

        ego = env.road.vehicles[0].position
        ego_lane_idx = np.array(env.road.network.get_closest_lane_index(np.array(ego))[2],np.float32)
        state_prime = np.append(state_prime,[ego_lane_idx])

        state = state_prime
        replay_buffer.add(state,action,state_prime, reward, done_bool)
        
        episode_reward += reward

        if t >= time_steps:
            policy.train(replay_buffer, batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment 
            state, done = env.reset(), False

            state = state[6].reshape(-1)
            ego = env.road.vehicles[0].position
            ego_lane_idx = np.array(env.road.network.get_closest_lane_index(np.array(ego))[2],np.float32)
            state = np.append(state,[ego_lane_idx])

            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        env.render()



        if done or keyboard_listener.reset_flag:
            obs = env.reset()
            keyboard_listener.reset_flag = False

        if generate_data:
            f.write(np.array_str(obs))

    evt.clear()
if __name__ == '__main__':
    evt = threading.Event()
    keyboard_listener = KeyboardEventHandler(evt)
    main()
    if f:
        f.close()
