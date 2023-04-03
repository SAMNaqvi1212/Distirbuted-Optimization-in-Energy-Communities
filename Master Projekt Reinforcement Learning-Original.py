#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning Main Model 

# This is the main model for the Reinforcement Learning. The third storage of battery is also added in this model and the date range is taken from 7th of July till 15th of July 2021. The third storage takes its values from the PV source of electricity. The battery then discharges its stored electricity for the main grid when there is no electricity available either from the PV or from the grid. The model is trained on both of the algorithms A2C (Actor critic Method) and also PPO (Proximal Policy Optimization). The model is trained on 300,000 timesteps. 

# In[1]:


#Import GYM stuff, importing the libraries
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


def get_data(start = '2017-01-01 00:00:00', end = '2017-12-01 23:55:00'):

    # import standard load profiles

    slp = pd.read_csv('df_p.csv', index_col=0, parse_dates=True)['0'] / 1000
    slp = slp.resample('15min').mean() 

    pv = pd.read_csv('Solar_Data-2011.csv', delimiter=';',
                    index_col=0, parse_dates=False)["Generation"] *10
    pv.index = slp.index
    print("Load values:")
    print(slp.values)
    print("PV values:")
    print(pv.values)
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    
    return slp[start:end], pv[start:end]


# In[ ]:





# In[3]:


class CostEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)    
        self.load, self.pv = get_data(start='2017-07-15 00:00:00', end='2017-07-18 23:55:00')
        self.pv_price = 0.10
        self.grid_price = 0.40
        self.line_max = 15
        self.grid_penalty = 100   # Specify the value prices for all of the sources and grid penalty.
        self.battery_max = 10
        self.battery_state = 0
        self.pv_values = []
        self.grid_values = []
        self.battery_state_values = []
        self.episode_length = len(self.load)
        self.observation_space = Dict(
            {
                "load": Box(0, self.episode_length - 1, shape=(2,), dtype=int),
                "pv": Box(0, self.episode_length - 1, shape=(2,), dtype=int),
                "battery_state": Box(0, self.battery_max, shape=(2,), dtype=int),
            }
        )

    def step(self, action):
        #print(f"Action taken: {action}")
        last_load = self.load[len(self.load) - self.episode_length]
        last_pv = self.pv[len(self.pv) - self.episode_length]
        if len(self.battery_state_values) > 0:  # check if the battery_state_values list is not empty
            last_battery_state = self.battery_state_values[-1]  # get the previous battery storage value
        else:
            last_battery_state = 0.0 
       # print(f"last_load: {last_load}")
       # print(f"last_pv: {last_pv}")

        if action == 0:
            #Take all the electricity from the main grid. 
            if last_load > self.line_max:
                reward = -1 * (last_load * self.grid_price + (last_load - self.line_max) * self.grid_penalty)
            else:
                reward = -1 * last_load * self.grid_price

            self.grid_values.append(last_load)
            self.pv_values.append(0.0)
            self.battery_state_values.append(last_battery_state)

        elif action == 1:
            #Take as much electricity from PV as possible
            excess_pv = max(0, last_pv - last_load)
            shortfall_load = max(0, last_load - last_pv)

            if shortfall_load > self.line_max:
                reward = -1 * (excess_pv * self.pv_price + shortfall_load * (self.grid_price + self.grid_penalty))
            else:
                reward = -1 * (excess_pv * self.pv_price + shortfall_load * self.grid_price)

            self.pv_values.append(last_pv)
            self.grid_values.append(0.0)

        elif action == 2:
            # Take electricity from the Battery storage when there is no PV or main electricity available taken from the grid. 
            excess_pv = max(0, last_pv - last_load)
            if excess_pv > 0:
                stored_pv = min(self.battery_max - self.battery_state, excess_pv)
                self.battery_state += stored_pv
                excess_pv -= stored_pv
        
            discharge_pv = min(last_load - last_pv + excess_pv, self.battery_state)
            self.battery_state -= discharge_pv

            self.pv_values.append(last_pv + discharge_pv)
            self.grid_values.append(last_load - discharge_pv)
            self.battery_state_values.append(self.battery_state)

            shortfall_load = max(0, last_load - last_pv - discharge_pv)
            if shortfall_load > self.line_max:
                reward = -1 * (shortfall_load * (0.15 + self.grid_penalty))
            else:
                reward = -1 * (shortfall_load * 0.15)

        else:
            #Penalize the agent if it takes neither of the above actions.
            reward = -float('inf')

        info = {}
        observation = {
            "load": (0, self.load[len(self.load) - self.episode_length]),
            "pv": (0, self.pv[len(self.pv) - self.episode_length]),
            "battery_state": (0, self.battery_state),
        }
        self.episode_length -= 1
        if self.episode_length <= 0: 
            done = True
        else:
            done = False

        #print(f"Reward: {reward}")
        return observation, reward, done, info
    
    def render(self):
        pass
    
    
    def reset(self):
        self.episode_length = len(self.load)
        #self.pv_values = []
        #self.grid_values = []
        #self.battery_state_values = []
        self.battery_state = 0
        observation = {
            "load": (0, self.load[len(self.load) - self.episode_length]),
            "pv": (0, self.pv[len(self.pv) - self.episode_length]),
            "battery_state": (0, self.battery_state),
        }
        
        return observation


# In[4]:


#Initialize the environment 
env = CostEnv()


# In[ ]:





# In[5]:


### Train the environment for inital first twenty episodes.
episodes = 20
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    reward = 0
    while not done:
        #env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        battery_reward = 0  # Add the battery reward calculation here
        score += reward + battery_reward
        #print('Action taken:', action)
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


# In[ ]:





# In[6]:


### Train the model on PPO algorithm 
log_path = os.path.join('Training', 'Logs')

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=300000)


# In[8]:


### Through this we evaluate our policy and observe we evaluation shall be proceeded with what level.
from stable_baselines3.common.evaluation import evaluate_policy
evaluate_policy(model, env, n_eval_episodes=20)


# 
# 

# In[8]:


### Train the algorithm on A2C algorithm 
log_path = os.path.join('Training', 'Logs')

model_1 = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=log_path)

model_1.learn(total_timesteps=300000)


# In[19]:


### Through this we evaluate our policy and observe we evaluation shall be proceeded with what level.
from stable_baselines3.common.evaluation import evaluate_policy
evaluate_policy(model_1, env, n_eval_episodes=20)


# In[10]:


model_1.save("A2C_Policy_Battery_new_22")


# In[11]:


#print(env.action_space.sample())


# In[12]:


#print(observation)


# In[13]:


#print(action)


# In[14]:


#print(reward)


# In[15]:


#env.grid_values


# In[16]:


##env.pv_values


# In[10]:


model.save("PPO_Policy_Battery_new_2")


# In[9]:


### Plotting the data for PPO algorithm.
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.plot(env.pv_values, label="PV")
plt.plot(env.grid_values, label="Grid")
plt.plot(env.battery_state_values,label="Battery")
plt.legend()
plt.show()


# In[ ]:




