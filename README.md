### Distirbuted-Optimization-in-Energy-Communities
This repository contains code and resources for my master's project on Distributed Optimization in Energy Communities, which was conducted under the supervision of the Cologne Institute of Renewable Energy. The project explores strategies for optimizing energy usage in communities using distributed computing techniques.

![image](https://user-images.githubusercontent.com/76792427/229624080-a791891b-2161-4932-bee8-3ec09f4bda42.png)


### Table of Contents 
- Introduction
- Methodology
- Conclusion
- Acknowledgements

### Introduction 
The Master Project comprises of two methodologies to optimize the distribution of electricity across the main grid. One approach is to use conventional mathematical model and other approach is to use custom Reinforcement Learning environment. Both models were built very similarlyto enable a later comparison.However,certain parameters that were used to describe the environments had to differ.The focus of the models was set on houses with PV-generation and battery-storage units. The maximum power of the PV-production was taken from  the data provided. The storage capacity as well as the charging power were sized adequately. The primary price of the grid was approximated were to be 40 cents. The cost of energy storage is 15 cents and the cost of PV source is 10 cents.

### Methodology
We designed a custom environment in which agent was provided with three actions. These actions were to take electricity from the three different sources named as PV, battery storage and main national grid. The agent's observation space is a dictionary which is apparently the load that is taken from the datasets. The agent was trained on two different algorithms namely Actor-Critic Method and Proximal Policy Optimization algorithm. Proximal Policy Optimization algorithm was the most efficeint of them. 

### Conclusion
The RL method implemented to optimize the distributed optimization of algorithm gave some interesting results that are described in detail in the paper in the repository. In general the battery storage amounted to rapid charge and discharge. While, the PV was extracted by the agent when there was excess sunlight and otherwise the electricity was taken from the grid. The agent extracted the most of the electricity from the battery storage and when there was more PV available it extracted it from the PV

### Acknowledgement
In this project, I would acknowledge the efforts of Mr.Sascha Birk who was my project supervisor. He was very kind and helpful. Throughout the project he was very patient and helped me with the project throughout the course. 



