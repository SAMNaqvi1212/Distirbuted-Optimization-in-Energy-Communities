# Distirbuted-Optimization-in-Energy-Communities
This repository contains code and resources for my master's project on Distributed Optimization in Energy Communities, which was conducted under the supervision of the Cologne Institute of Renewable Energy. The project explores strategies for optimizing energy usage in communities using distributed computing techniques.

## Table of Contents 
- Introduction
- Methodology
- Conclusion
- Acknowledgements

# Introduction 
The Master Project comprises of two methodologies to optimize the distribution of electricity across the main grid. One approach is to use conventional mathematical model and other approach is to use custom Reinforcement Learning environment. Both models were built very similarlyto enable a later compari-son.However,certain parameters that were used to describe the environments hadto differ.The focus of the models was set on houses with PV-generation and batterystorage units. The maxi-mum power of the PV-production was taken from  the data pro-vided. The storagecapacity as well as thecharging powerweresized adequately. 
The prices of the sources of electricity are as follows: 

Parameter 

Reinforcement learning 

Mathematical  
optimization 

Grid capacity 

15 kWh 

infinite 

Storage capacity 

10 kWh 

10 kWh 

Charging power 

 / 

5 kW 

Initial storage charge 

0 kWh 

0 kWh 

Energy cost (grid) 

40 cents 

40 cents 

Energy cost (storage) 

15 cents 

15 cents 

Energy cost (PV) 

10 cents 

10 cents 


