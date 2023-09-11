# MARL_for_MODN
Repository of code for reproducing the results of paper "Scaling up Energy-Aware Multi-Agent Reinforcement Learning for Mission-Oriented Drone Networks with Individual Reward"

## Installing Dependencies
  python version >=3.6
   ```bash
   pip3 install -r requirements.txt
   ```

## Run Experiments
### for individual reward
   ```bash
   python3 individual_reward/simulation.py --random_location=True --random_length=True
   ```
### for shared reward 
   ```bash
   python3 shared_reward/simulation.py --random_location=True --random_length=True
   ```
