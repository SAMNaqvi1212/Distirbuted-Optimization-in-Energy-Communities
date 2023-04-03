#!/usr/bin/env python
# coding: utf-8

# # BASIC MODEL FOR REINFORCEMENT LEARNING

# This file is a basic model for Reinforcement Learning model. 
# This code contains the basic structure on which further improvements 
# were made to build the Reinforcement Learning model for grid optimization.
# Initially in this file there are two sources of electricity
# that are used to build the environment. 
# One is the main grid source of electricity and other one is from PV source of electricity. 

# In[1]:


#Import GYM stuff, basic libraries import
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

#Import helpers
import numpy as np
import random
import os
import pandas as pd

# Import stable baselines stuff
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# In[2]:


def get_data(start = '2017-01-01 00:00:00', end = '2017-03-01 23:55:00'):

    # import standard load profiles

    slp = pd.read_csv('df_p.csv', index_col=0, parse_dates=True)['0'] / 1000
    slp = slp.resample('15min').mean() * 3

    pv = pd.read_csv('Solar_Data-2011.csv', delimiter=';',
                    index_col=0, parse_dates=False)["Generation"] * 3
    pv.index = slp.index
    print("Load values:")
    print(slp.values)
    print("PV values:")
    print(pv.values)
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    return slp[start:end], pv[start:end]


# In[5]:


class CostEnv(Env):
    def __init__(self):
        # Actions we can take increase in cost, lowering of cost
        self.action_space = Discrete(2)
        #import data, we take first 3 months
        self.load, self.pv = get_data(start = '2017-01-01 00:00:00', end = '2017-03-01 23:55:00')
        self.pv_price = 0.10
        self.grid_price = 0.40     # setting the main basic price for different sources 
        self.line_max = 15
        self.grid_penalty = 100
        self.battery_max = 18
        self.battery_state = 10
        self.pv_values = []
        self.grid_values = []
        self.episode_length = len(self.load)
        self.observation_space = Dict(
            {
                "load": Box(0, self.episode_length - 1, shape=(2,), dtype=int),
                "pv": Box(0, self.episode_length - 1, shape=(2,), dtype=int),
            }
        )

    def step(self, action):
        ### Apply action
        ### We calculate the reward based on the price for the electricity,
        ###   lower price, "higher" reward
        print(f"Action taken: {action}")
        last_load = self.load[len(self.load) - self.episode_length]
        last_pv = self.pv[len(self.pv) - self.episode_length]
        print(f"last_load: {last_load}")
        print(f"last_pv: {last_pv}")

        if action == 0:
            #Take all electricity from the grid
            if last_load > self.line_max:
                reward = -1 * (last_load * self.grid_price + (last_load - self.line_max) * self.grid_penalty)
            else:
                reward = -1 * last_load * self.grid_price

            self.grid_values.append(last_load)

        elif action == 1:
            #Take electricity from PV
            excess_pv = max(0, last_pv - last_load)
            shortfall_load = max(0, last_load - last_pv)

            if shortfall_load > self.line_max:
                reward = -1 * (excess_pv * self.pv_price + shortfall_load * (self.grid_price + self.grid_penalty))
            else:
                reward = -1 * (excess_pv * self.pv_price + shortfall_load * self.grid_price)

            self.pv_values.append(last_pv)

        else:
            ## If neither of the action is chosen the agent will be penalized with -infinity reward
            reward = -float('inf')

        info = {}
        observation = {
            "load": (0,self.load[len(self.load)-self.episode_length]),
            "pv": (0,self.pv[len(self.pv)-self.episode_length]),
        }
        ### Either here or before checking self.episode_length
        self.episode_length -= 1
        ### Check if the timeseries is over
        if self.episode_length <= 0: 
            done = True
        else:
            done = False
            
        print(f"Reward: {reward}")
        # Return step information
        # return self.state, reward, done, info
        return observation, reward, done, info

    def render(self):
        pass
    
    def reset(self):
        self.done=False
        #Set episode length
        self.episode_length = len(self.load)
        observation = {
            "load": (0, self.load[len(self.load)-self.episode_length]),
            "pv": (0, self.pv[len(self.pv)-self.episode_length]),
        }
        
        return observation


# In[7]:


### Initialize the initial environment 
env = CostEnv()


# In[25]:





# In[8]:


### Testing the model on how it performs by training it initially on twenty episodes
episodes = 20
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        # n_state, reward, done, info = env.step(action)
        observation, reward, done, info = env.step(action)
        score+=reward
        #print('Action taken:', action)
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


# In[ ]:





# In[15]:


### Training the model on PPO (Proximal Policy Optimization) algorithm 
log_path = os.path.join('Training', 'Logs')

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=300000)


# In[29]:


### Through this we evaluate our policy and observe we evaluation shall be proceeded with what level.
from stable_baselines3.common.evaluation import evaluate_policy
evaluate_policy(model, env, n_eval_episodes=20)


# 

# In[ ]:





# In[ ]:





# In[32]:


print(env.action_space.sample())


# In[16]:


print(observation)


# In[17]:


print(action)


# In[18]:


print(reward)


# In[13]:


env.grid_values


# In[14]:


env.pv_values


# In[19]:


### Saving the model PPO
model.save("PPO_Policy_Improved")


# In[20]:


### Plotting the graphs for PV and Grid values 
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.plot(env.pv_values, label="PV")
plt.plot(env.grid_values, label="Grid")
plt.legend()
plt.show()


# In[ ]:




