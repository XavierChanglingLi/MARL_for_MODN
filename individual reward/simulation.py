import numpy as np
import tensorflow as tf
import argparse
import itertools
import time
import os
import pickle
import code
import random

from dqn import DQN
from memory import Memory
from env import Environment
import general_utilities

def play(episodes, is_render, is_testing, checkpoint_interval,
         weights_filename_prefix, csv_filename_prefix, batch_size):
    # init statistics. NOTE: simple tag specific!
    statistics_header = ["episode"]
    statistics_header.append("steps")
    statistics_header.append("done")
    statistics_header.extend(["reward_{}".format(i) for i in range(env.num_agents)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.num_agents)])
    statistics_header.extend(["eps_greedy_{}".format(i) for i in range(env.num_agents)])
    statistics_header.extend(["Task location_{}".format(i) for i in range(env.num_tasks)])
    statistics_header.extend(["Agent Energy Left_{}".format(i) for i in range(env.num_agents)])
    statistics_header.extend(["Task Energy Left_{}".format(i) for i in range(env.num_agents)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)

    for episode in range(args.episodes):
        states = env.reset()
        episode_losses = np.zeros(env.num_agents)
        episode_rewards = np.zeros(env.num_agents)

        steps = 0

        all_states = [states]
        while steps <= 600:
            steps += 1
            actions = []
            # n represents agents' number
            for i in range(env.num_agents):
                action = dqns[i].choose_action(states)
                actions.append(action)

            # step
            states_next, rewards, done, info = env.step(actions)
            all_states.append(states_next)
            # learn
            if not args.testing:
                size = memories[0].pointer
                batch = random.sample(range(size), size) if size < batch_size else random.sample(
                    range(size), batch_size)

                for i in range(env.num_agents):
                    memories[i].remember(states, actions[i], rewards[i], states_next, done)

                    if memories[i].pointer > batch_size * args.phi:
                        history = dqns[i].learn(*memories[i].sample(batch))
                        episode_losses[i] += history.history["loss"][0]
                    else:
                        for i in range(env.num_agents):
                            episode_losses[i] = -1

            states = states_next
            episode_rewards += rewards

            # reset states if done
            if done or steps >= 600:
                episode_losses = episode_losses / steps

                statistic = [episode]
                statistic.append(steps)
                statistic.append(done)
                statistic.extend([episode_rewards[i] for i in range(env.num_agents)])
                statistic.extend([episode_losses[i] for i in range(env.num_agents)])
                statistic.extend([dqns[i].eps_greedy for i in range(env.num_agents)])
                statistic.extend([env.tasks_positions[i] for i in range(env.num_tasks)])
                statistic.extend([env.B_k[i] for i in range(env.num_agents)])
                statistic.extend([env.T_i[i] for i in range(env.num_agents)])
                statistics.add_statistics(statistic)
                if episode % 1 == 0:
                    print(statistics.summarize_last())

                if done:
                    myfile = open('./save/states/episode{}_states.txt'.format(episode), 'w')
                    for each in all_states:
                        myfile.write(str(each))
                        myfile.write('\n')
                    myfile.close()
                break

        if episode % checkpoint_interval == 0:
            statistics.dump("{}_{}.csv".format(csv_filename_prefix,
                                               episode))
            general_utilities.save_dqn_weights(dqns,
                                               "{}_{}_".format(weights_filename_prefix, episode))
            if episode >= checkpoint_interval:
                os.remove("{}_{}.csv".format(csv_filename_prefix,
                                             episode - checkpoint_interval))

    return statistics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--episodes', default=20000, type=int)
    parser.add_argument('--render', default=False, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--experiment_prefix', default=".",
                        help="directory to store all experiment data")
    parser.add_argument('--weights_filename_prefix', default='/save/tag-dqn',
                        help="where to store/load network weights")
    parser.add_argument('--csv_filename_prefix', default='/save/statistics-dqn',
                        help="where to store statistics")
    parser.add_argument('--checkpoint_frequency', default=50,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--phi', default=5, type=int,
                        help="adjust batch size")
    parser.add_argument('--replace_target_freq', default=2000, type=int,
                        help="adjust the frequency of updating target network")
    parser.add_argument('--random_location', default=True,
                        help="set task location as random")
    parser.add_argument('--random_length', default=True,
                        help="set task length as random")
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--memory_size', default=10000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epsilon_greedy', nargs='+', type=float,
                        help="Epsilon greedy parameter for each agent")
    args = parser.parse_args()

    general_utilities.dump_dict_as_json(vars(args),
                                        args.experiment_prefix + "/save/run_parameters.json")
    # init env
    env = Environment(random_loc=args.random_location, random_leng=args.random_length)

    if args.epsilon_greedy is not None:
        if len(args.epsilon_greedy) == env.num_agents:
            epsilon_greedy = args.epsilon_greedy
        else:
            raise ValueError("Must have enough epsilon_greedy for all agents")
    else:
        # change the initial exploitation rate
        epsilon_greedy = [0.5 for i in range(env.num_agents)]


    # set random seed
    # env.seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    # tf.random.set_seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # init DQNs
    n_actions = len(env.action_space)
    state_sizes = env.state_size
    memories = [Memory(args.memory_size) for i in range(env.num_agents)]
    dqns = [DQN(n_actions, state_sizes, eps_greedy=epsilon_greedy[i], eps_increment=0.0000003, replace_target_freq=args.replace_target_freq)
            for i in range(env.num_agents)]

    general_utilities.load_dqn_weights_if_exist(
        dqns, args.experiment_prefix + args.weights_filename_prefix)

    start_time = time.time()

    # play
    statistics = play(args.episodes, args.render, args.testing,
                      args.checkpoint_frequency,
                      args.experiment_prefix + args.weights_filename_prefix,
                      args.experiment_prefix + args.csv_filename_prefix,
                      args.batch_size)

    print("Finished {} episodes in {} seconds".format(args.episodes,
                                                      time.time() - start_time))
    general_utilities.save_dqn_weights(
        dqns, args.experiment_prefix + args.weights_filename_prefix)
    statistics.dump(args.experiment_prefix + args.csv_filename_prefix + ".csv")
