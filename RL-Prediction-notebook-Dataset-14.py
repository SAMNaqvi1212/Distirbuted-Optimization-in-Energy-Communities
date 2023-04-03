#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning Model For Dataset-14

# This notebook takes the saved Reinforcement Learning model that is trained and then implemented on the dataset-14. The agent is loaded in this notebook and then predicts while using the dataset-14. 

# In[1]:


#Import GYM stuff, import the basic libraries.
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

    slp = pd.read_csv('df_p.csv', index_col=0, parse_dates=True)['14'] / 1000
    slp = slp.resample('15min').mean() 

    pv = pd.read_csv('Solar_Data-2011.csv', delimiter=';',
                    index_col=0, parse_dates=False)["Generation"] * 10  
    pv.index = slp.index
    print("Load values:")
    print(slp.values)
    print("PV values:")
    print(pv.values)
    start = pd.to_datetime(start)
    end = pd.to_datetime(end) 
    return slp[start:end], pv[start:end]


# In[3]:


class CostEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)    
        self.load, self.pv = get_data(start='2017-07-15 00:00:00', end='2017-07-18 23:55:00')
        self.pv_price = 0.10
        self.grid_price = 0.40
        self.line_max = 15
        self.grid_penalty = 100
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
            ##Take all the electricity from the grid
            if last_load > self.line_max:
                reward = -1 * (last_load * self.grid_price + (last_load - self.line_max) * self.grid_penalty)
            else:
                reward = -1 * last_load * self.grid_price

            self.grid_values.append(last_load)
            self.pv_values.append(0.0)
            self.battery_state_values.append(last_battery_state)

        elif action == 1:
            ### Take the electricity as much as possible from PV
            excess_pv = max(0, last_pv - last_load)
            shortfall_load = max(0, last_load - last_pv)

            if shortfall_load > self.line_max:
                reward = -1 * (excess_pv * self.pv_price + shortfall_load * (self.grid_price + self.grid_penalty))
            else:
                reward = -1 * (excess_pv * self.pv_price + shortfall_load * self.grid_price)

            self.pv_values.append(last_pv)
            self.grid_values.append(0.0)

        elif action == 2:
            #Take the electricity from the battery storage when there is no excess pv available and no electricity
            # from the grid. It first stores the electricity from the excess PV
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
                reward = -1 * (shortfall_load * (self.grid_price + self.grid_penalty))
            else:
                reward = -1 * (shortfall_load * self.grid_price)

        else:
            # Penalize the agent when it takes neither of the actions above. 
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


### Initialize the environment
env = CostEnv()


# In[5]:


#Loading the model. 
model = PPO("MultiInputPolicy",env=env)
model = model.load("PPO_Policy_Battery_new_2",env=env)


# In[6]:


### Testing the model on dataset-14 and saving the in-between values for PV,battery storage and grid from the environment. 
env = CostEnv()
obs = env.reset()
done = False
while not done:
    action,state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.pv_values.append(obs['pv'][1])
    env.grid_values.append(obs['load'][1] - obs['pv'][1])
    env.battery_state_values.append(obs['battery_state'][1])


# In[7]:


### Plotting the graphs for the given values. 
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
plt.plot(env.pv_values, label="PV")
plt.plot(env.grid_values, label="Grid")
plt.plot(env.battery_state_values,label="Battery")
plt.title("Grid Optimization for Dataset-14")
plt.legend(loc="upper left")
plt.show()


# In[8]:


#grid_values = []
#grid_values.append(env.grid_values)


# In[ ]:





# In[9]:


print(action)


# In[10]:


print(obs)


# In[11]:


print(info)


# In[12]:


print(reward)


# In[ ]:


# Saving the values in a separate dataframe to plot it for the paper.


# In[13]:


start_date = '2017-07-15 00:00:00'
end_date = '2017-07-18 23:55:00'
time_interval = 15 # minutes
xticks = pd.date_range(start=start_date, end=end_date, freq=f"{time_interval}T")


# In[8]:


import pandas as pd

# Calculate the length of the values
n_values = len(env.grid_values)

# Calculate the frequency based on the length of the values
freq = '{}min'.format((len(env.grid_values) * 5) // (24 * 60))

# Generate the date range
date_range = pd.date_range(start='2017-07-15 00:00:00',end ='2017-07-18 23:55:00', periods=n_values)

# Create the dataframe
df_14 = pd.DataFrame({
    'grid_values_14': env.grid_values,
    'pv_values_14': env.pv_values,
    'battery_state_values_14': env.battery_state_values
}, index=date_range)


# In[9]:


import os

# create the directory if it does not exist
directory = 'Desktop/tootoo/Datasets/Reinforcement-Learning/'
if not os.path.exists(directory):
    os.makedirs(directory)

# save the DataFrame to a CSV file
df_14.to_csv(directory + '14_dataframe.csv', index=True)
df_14.to_csv(r'Desktop/tootoo/Datasets/Reinforcement-Learning/14_dataframe.csv',index=True)


# In[24]:


### Plotting the graphs 
import matplotlib.pyplot as plt
# Plot the data
plt.figure(figsize=(10,10))
plt.plot(df_14.index, env.pv_values, label='PV')
plt.plot(df_14.index, env.grid_values, label='Grid')
plt.plot(df_14.index, env.battery_state_values, label='Battery')
plt.xlabel('Time')
plt.ylabel('Power (kW)')
#plt.title('Power Data from {} to {}'.format(df.index[0], df.index[-1]))
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




