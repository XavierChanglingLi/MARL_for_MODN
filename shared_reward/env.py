import random
import numpy as np
import operator

K = 20 #2x2 tasks #9 #3x3 tasks # number of drones
NUMBER_OF_TASKS = 20 # 2x2 tasks #9 # 3x3 tasks # number of tasks
GRID_SIZE = 7# 2x2 tasks #7 #3x3 tasks # size of the grid/environment
ALPHA = 0.5  # constant coefficient for execution in reward function
C_H = 2  # energy consumption rate for hovering
C_T = 3  # energy consumption rate for task execution
C_F = 2.5  # energy consumption rate for forwarding
EFFICIENCY_THRESHOLD = 0.3  # energy efficiency threshold for calculating reward
B = 1200  # battery capacity for drone
T = 15  # energy required for one single task # task length is 5 as C_T = 3
Q = -50  # drone not coming back penalty
TRAVEL_ENERGY_THRESHOLD = 8000

# actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
UPPER_LEFT = 4
UPPER_RIGHT = 5
BOTTOM_LEFT = 6
BOTTOM_RIGHT = 7
STATIONARY = 8
EXECUTE = 9

# base station
BASE_X = 6 #2x2 tasks #6 # 3 x 3 tasks
BASE_Y = 6 #6


class Environment:
    def __init__(self, random_loc = True, random_leng = True):
        self.num_agents = K
        self.num_tasks = NUMBER_OF_TASKS
        self.grid_size = GRID_SIZE
        self.state_size = self.num_agents * 4 + self.num_tasks * 3  # not used
        self.agents_positions = []
        self.tasks_positions = []
        self.cells = []
        self.random_loc = random_loc
        self.random_leng = random_leng

        self.y_ik = np.zeros((self.num_tasks, self.num_agents))
        self.B_k = np.array([B for i in range(self.num_agents)])
        self.T_i = np.array([T for i in range(self.num_tasks)])
        self.tasks_positions_idx = []

        self.action_space = [UP, DOWN, LEFT, RIGHT, UPPER_LEFT, UPPER_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, STATIONARY, EXECUTE]
        self.action_diff = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0), (0, 0)]
        self.num_episodes = 0
        self.terminal = False

    def set_positions_idx(self):

        cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)]
        #tasks_positions_idx = [8, 10, 12, 22, 24, 26, 36, 38, 40] # 3x3 tasks
        #tasks_positions_idx = [6, 8, 16, 18] # 2x2 tasks
        if self.random_loc:
            tasks_positions_idx = np.random.choice(len(cells) - 1, size=self.num_tasks, replace=False)
        else:
            tasks_positions_idx = [6, 8, 16, 18] # 2x2 tasks
        return [cells, tasks_positions_idx]

    def reset(self):  # initialize the world

        self.terminal = False
        [self.cells, self.tasks_positions_idx] = self.set_positions_idx()
        self.y_ik = np.zeros((self.num_tasks, self.num_agents))
        self.B_k = np.array([B for i in range(self.num_agents)])
        # random task length per task,  the max length is 5
        if self.random_leng:
            self.T_i = np.array([random.randint(1, 5) * C_T for i in range(self.num_tasks)])
        else:
            self.T_i = np.array([T for i in range(self.num_tasks)])

        # map generated position indices to positions
        self.tasks_positions = [self.cells[pos] for pos in self.tasks_positions_idx]
        self.agents_positions = [(BASE_X, BASE_Y) for i in range(K)]
        initial_actions = [8 for i in range(self.num_agents)]
        initial_pos_state = list(sum(self.tasks_positions + self.agents_positions, ()))
        initial_state = initial_pos_state + initial_actions + list(self.T_i)
        initial_state = initial_state + list(self.B_k)
        return np.array(initial_state)

    def step(self, agents_actions):
        # task finisihed & drones all go back to base station
        # update the position of agents
        self.agents_positions = self.update_positions(self.agents_positions, agents_actions)
        #
        # update drones energy state
        self.B_k = self.update_agents_energy(agents_actions)
        # update task energy required
        sum_before = np.sum(self.T_i)
        self.T_i = self.update_tasks_energy(self.agents_positions, agents_actions)
        # update 2D array (the portion of task at location i executed by drone k)
        self.y_ik = self.update_2D_array(self.agents_positions, agents_actions)
        sum_after = np.sum(self.T_i)
        
        if np.all(self.T_i <= 0): #and np.all(np.asarray(self.agents_positions)==(BASE_X,BASE_Y)):
            if np.all(self.B_k >= 0):
                reward = (sum_before - sum_after) / C_T + np.sum(self.B_k)/(self.num_agents*B) * 100
            else:
                reward = (sum_before - sum_after) / C_T - (self.B_k<0).sum()
            self.terminal = True

        #task unfinished
        else:
            if sum_after < sum_before:
                reward = (sum_before - sum_after) / C_T
            else:
                reward = 0

        new_pos_state = list(sum(self.tasks_positions + self.agents_positions, ()))
        new_state = new_pos_state + agents_actions + list(self.T_i) + list(self.B_k)
        return np.array(new_state), reward, self.terminal, {}


    def update_agents_energy(self, act_list):
        B_k = self.B_k
        for i in range(len(act_list)):
            if self.agents_positions[i] == (BASE_X,BASE_Y) and act_list[i]==8:
                B_k[i] = B_k[i]
            if 4 <= act_list[i] <= 7:
                B_k[i] = B_k[i] - C_F
            elif act_list[i] <= 3:
                B_k[i] = B_k[i] - C_F * (2 ** (1 / 2) / 2) - C_H * (1 - (2 ** (1 / 2) / 2))
            elif act_list[i] == 9:
                B_k[i] = B_k[i] - C_T - C_H
            else:
                B_k[i] = B_k[i] - C_H
        return B_k

    def update_tasks_energy(self, pos_list, act_list):
        T_i = self.T_i
        execute_idx = [i for i in range(len(act_list)) if act_list[i] == 9]
        agents_execute_pos = list(np.array(pos_list)[execute_idx])
        task_idx = []
        for i in range(len(agents_execute_pos)):
            for j in range(len(self.tasks_positions)):
                if (self.tasks_positions[j] == tuple(agents_execute_pos[i]) and self.B_k[execute_idx[i]]>C_T+C_H and self.T_i[j] >= C_T):
                    task_idx.append(j)
        for idx in task_idx:
            T_i[idx] = T_i[idx] - C_T

        return T_i

    def update_2D_array(self, pos_list, act_list):
        y_ik = self.y_ik
        execute_idx = [i for i in range(len(act_list)) if act_list[i] == 9]
        agents_execute_pos = list(np.array(pos_list)[execute_idx])
        for p in range(len(agents_execute_pos)):
            for q in range(len(self.tasks_positions)):
                if (self.tasks_positions[q] == tuple(agents_execute_pos[p]) and self.B_k[execute_idx[p]]>C_T+C_H and self.T_i[q] >= C_T):
                    y_ik[q][p] = y_ik[q][p] + C_T
        return y_ik

    def update_positions(self, pos_list, act_list):
        positions_action_applied = []
        for idx in range(len(pos_list)):
            if act_list[idx] != 8 and act_list[idx] != 9:
                # pos_act_applied = map(operator.add, pos_list[idx], self.action_diff[act_list[idx]])
                pos_act_applied = list(np.asarray(pos_list[idx]) + np.asarray(self.action_diff[act_list[idx]]))

                for i in range(0, 2):
                    if pos_act_applied[i] < 0:
                        pos_act_applied[i] = 0
                    if pos_act_applied[i] >= self.grid_size:
                        pos_act_applied[i] = self.grid_size - 1
                positions_action_applied.append(tuple(pos_act_applied))
            else:
                positions_action_applied.append(pos_list[idx])


        return positions_action_applied

    def get_action_space_size(self):
        return len(self.action_space)

  
    def find_frequency(self, a, items):
        freq = 0
        for item in items:
            if item == a:
                freq += 1

        return freq
