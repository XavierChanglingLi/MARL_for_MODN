import random
import numpy as np

K = 26  # number of drones
NUMBER_OF_TASKS = 26  # number of tasks
GRID_SIZE = 8  # size of the grid/environment
ALPHA = 0.5  # constant coefficient for execution in reward function
C_H = 2  # energy consumption rate for hovering
C_T = 3  # energy consumption rate for task execution
C_F = 2.5  # energy consumption rate for forwarding
EFFICIENCY_THRESHOLD = 0.3  # energy efficiency threshold for calculating reward
B = 1200  # battery capacity for drone
T = 15  # energy required for one single task
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
BASE_X = 7
BASE_Y = 7


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
        # energy left of each agent
        self.B_k = np.array([B for i in range(self.num_agents)])
        # energy left for each task
        self.T_i = np.array([T for i in range(self.num_tasks)])
        # self.tasks_positions_idx = np.random.choice(len(self.cells) - 1, size=self.num_tasks,
        #                                     replace=False)
        self.tasks_positions_idx = []

        self.action_space = [UP, DOWN, LEFT, RIGHT, UPPER_LEFT, UPPER_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, STATIONARY, EXECUTE]
        self.action_diff = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0), (0, 0)]
        self.num_episodes = 0
        self.terminal = False

    def set_positions_idx(self):

        cells = [(i, j) for i in range(0, self.grid_size) for j in range(0, self.grid_size)]
        if self.random_loc:
            tasks_positions_idx = np.random.choice(len(cells) - 1, size=self.num_tasks, replace=False)
        else:
            tasks_positions_idx = [6, 8, 16, 18]
        return [cells, tasks_positions_idx]

    def reset(self):  # initialize the world

        self.terminal = False
        [self.cells, self.tasks_positions_idx] = self.set_positions_idx()
        self.y_ik = np.zeros((self.num_tasks, self.num_agents))
        self.B_k = np.array([B for i in range(self.num_agents)])
        if self.random_leng:
            self.T_i = np.array([random.randint(1, 5) * C_T for i in range(self.num_tasks)])
        else:
            self.T_i = np.array([T for i in range(self.num_tasks)])
        
        # map generated position indices to positions
        self.tasks_positions = [self.cells[pos] for pos in self.tasks_positions_idx]
        # print(self.tasks_positions)
        self.agents_positions = [(BASE_X, BASE_Y) for i in range(K)]
        initial_actions = [8 for i in range(self.num_agents)]
        initial_pos_state = list(sum(self.tasks_positions + self.agents_positions, ()))
        initial_state = initial_pos_state + initial_actions + list(self.T_i)
        initial_state = initial_state + list(self.B_k)
        return np.array(initial_state)

    def step(self, agents_actions):
                # instead, update on each agent use a for loop, and then decide whether it is terminated or not
        self.agents_positions = self.update_positions(self.agents_positions, agents_actions)
        rewards = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            
            sum_before = np.sum(self.T_i)
            #update tasks energy
            if agents_actions[i] == 9 and self.B_k[i] > C_T+C_H :
                self.T_i = self.update_task_energy(self.agents_positions[i])
            sum_after = np.sum(self.T_i)
            # if the task energy is reduced, give reward
            reward = (sum_before - sum_after) / C_T
            rewards[i] = reward

            #update each agent energy
            if self.agents_positions[i] == (BASE_X,BASE_Y) and agents_actions[i] ==8:
                self.B_k[i] = self.B_k[i]
            else:
                self.B_k[i]= self.update_agent_energy(self.B_k[i], agents_actions[i])

            # check each agent energy and if the agent energy is negative, give the agent a negative reward 
            if self.B_k[i] <0:
                rewards[i] += - 1

        #check to see if the tasks are finished 
        if np.all(self.T_i<=0): #and np.all(np.asarray(self.agents_positions)==(BASE_X,BASE_Y)):
            if np.all(self.B_k >= 0):
                #give reward to each agent based on their left energy
                rewards = np.array(rewards, dtype='float64')
                rewards += self.B_k/B * 3
            self.terminal = True


        self.y_ik = self.update_2D_array(self.agents_positions, agents_actions)
        new_pos_state = list(sum(self.tasks_positions + self.agents_positions, ()))
        new_state = new_pos_state + agents_actions + list(self.T_i) + list(self.B_k)
        # print(rewards)
        return np.array(new_state), rewards, self.terminal, {}


    def update_agent_energy(self, energy, action):
        B_k = energy
        if 4 <= action <= 7:
            B_k = B_k - C_F
        elif action <= 3:
            B_k = B_k - C_F * (2 ** (1 / 2) / 2) - C_H * (1 - (2 ** (1 / 2) / 2))
        elif action == 9:
            B_k = B_k - C_T - C_H
        else:
            B_k = B_k - C_H
        return B_k

    def update_task_energy(self, position):
        T_i = self.T_i
        for i in range(len(self.tasks_positions)):
            if (self.tasks_positions[i]==tuple(position) and self.T_i[i]>=C_T):
                T_i[i] = T_i[i] - C_T

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
                pos_act_applied = list(np.asarray(pos_list[idx]) + np.asarray(self.action_diff[act_list[idx]]))

                # checks to make sure the new pos in inside the grid
                # for key in pos_act_applied:
                #     print(key)
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
