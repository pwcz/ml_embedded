#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
import matplotlib.pyplot as plt
import logging
from symulator import System
from test_plan import TestPlan
from timeit import default_timer as timer
from copy import deepcopy
import numpy as np
import concurrent.futures


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


class SimulationProcess:
    def __init__(self, _test_plan):
        self.test_plan = _test_plan
        self.start_time = _test_plan.start_time
        self.end_time = _test_plan.end_time
        self.time_delta = _test_plan.time_delta
        self.time_delta_sym = _test_plan.time_delta_sym
        self.training_data_generator = _test_plan.data_generator
        self.epoch_start = _test_plan.epoch_start
        self.epoch_stop = _test_plan.epoch_stop
        self.epoch_step = _test_plan.epoch_step
        self.start_power = _test_plan.start_power
        self.get_new_test_data_for_each_epoch = _test_plan.get_new_test_data_for_each_epoch
        self.legend = _test_plan.get_legend()
        self.logger_info = _test_plan.get_info()
        self.epoch_data = None

    def start_simulation(self):
        """
        Function for starting simulation
        """
        # prepare data
        self.epoch_data = {
            'average_delay': [],
            'power_left': [],
            'quality_data': [],
            'epoch': []
        }

        # get training data
        sim_start = timer()
        training_data_gold = self.training_data_generator.get_train_data()

        # prepare simulating system
        embedded = System(self.start_time, start_power=self.start_power, test_plan=self.test_plan)

        for epoch in range(self.epoch_start, self.epoch_stop + 1, self.epoch_step):
            # time measure
            start = timer()

            # initial data set
            embedded.reset_epoch()
            data_record = {
                'delays_data': []
            }

            # training data
            if self.get_new_test_data_for_each_epoch:
                training_data = self.training_data_generator.get_train_data()
                # print(training_data)
            else:
                training_data = deepcopy(training_data_gold)
            next_action = training_data.pop()

            for current_time in self.training_data_generator.time_range(self.start_time, self.end_time,
                                                                        self.time_delta_sym):
                # external clock
                embedded.simulation_step(current_time)

                # event actions
                if next_action == current_time:
                    delay = embedded.event_action(current_time)
                    data_record['delays_data'].append(delay)
                    if len(training_data) > 0:
                        next_action = training_data.pop()

            # collecting data
            end = timer()
            avg_delay = sum(data_record['delays_data'])/len(data_record['delays_data'])
            norm_power_left = float(embedded.power_left)  # /float(self.start_power)
            self.epoch_data['average_delay'] .append(avg_delay)
            self.epoch_data['power_left'].append(norm_power_left)
            self.epoch_data['epoch'].append(epoch)
            logging.debug("Epoch " + '{:3}'.format(epoch) + " | Time: " + '{:05.2f}'.format(end - start) + " s" +
                          " | Avg_delay = " + '{:5.2f}'.format(avg_delay) + " s | Norm. power consumption = " +
                          '{:03.3f}'.format(1 - norm_power_left))
        # logging.info("LEARNING FINISHED! Elapsed time: " + '{:3.3f}'.format(timer()-sim_start))
        logging.info("| Epoch number: " + str(len(self.epoch_data['epoch'])) + " | " + self.logger_info + " | Time: " +
                     '{:05.2f} s'.format(end - start) + " | avg. delay: " +
                     '{:05.2f}'.format(np.mean(self.epoch_data['average_delay'][-20:])) + " | avg. power consum.: " +
                     '{:05.2f}'.format(np.mean(self.epoch_data['power_left'][-20:])) + " |"
                     )

    def display_epoch_data(self, plots):
        plt.figure(1)
        plot_layout = {'average_delay': ['epoch', 'average_delay', 'r--', 'epoka', 'średnie opóźnienia [s]'],
                       'power_left': ['epoch', 'power_left', 'bs', 'epoch', 'epoka. zużycie energii']
                       }
        for i, p in enumerate(plots):
            plt.subplot(len(plots)*100 + 11 + i)
            plt.plot(self.epoch_data[plot_layout[p][0]], self.epoch_data[plot_layout[p][1]], plot_layout[p][2])
            plt.xlabel(plot_layout[p][3])
            plt.ylabel(plot_layout[p][4])
            plt.grid(True)
        plt.savefig("reward_function.png")
        plt.show()


class MultipleTests:
    def __init__(self, simulation_systems, test_count=1):
        self.sim_sys = simulation_systems
        self.test_count = test_count
        self.plot_layout = ['average_delay', 'power_left']
        self.sim_numer = len(simulation_systems)
        self.result = [[{} for _ in range(self.test_count)] for _ in range(self.sim_numer)]
        self.plot_data = None

    def start(self):
        for k, sim in enumerate(self.sim_sys):
            for m in range(self.test_count):
                sim.start_simulation()
                # print(sim.epoch_data['average_delay'])
                for j in self.plot_layout:
                    self.result[k][m][j] = deepcopy(sim.epoch_data[j])
        self.plot_data = self.prepare_data()

    def parallel_start(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(self.get_result_data, self.sim_sys)
        for i, d in enumerate(result):
            self.result[i] = d

    def get_result_data(self, _sim):
        data = []
        for m in range(self.test_count):
            _sim.start_simulation()
            for j in self.plot_layout:
                data[m][j] = deepcopy(_sim.epoch_data[j])
        return data

    def prepare_data(self):
        # print(self.result)
        epoch_number = len(self.result[0][0][self.plot_layout[0]])
        summary_result = {x: np.empty([self.sim_numer, 3, epoch_number]) for x in self.plot_layout}
        for m, test in np.ndenumerate(self.result):
            for epoch in range(epoch_number):
                dd = {x: [] for x in self.plot_layout}
                for repeat_num in range(self.test_count):
                    for data_taken in self.plot_layout:
                        dd[data_taken].append(self.result[m[0]][repeat_num][data_taken][epoch])
                for b in self.plot_layout:
                    standard_deviation = np.std(dd[b])
                    mean = np.mean(dd[b])
                    summary_result[b][m[0]][0][epoch] = mean - standard_deviation
                    summary_result[b][m[0]][1][epoch] = mean
                    summary_result[b][m[0]][2][epoch] = mean + standard_deviation

        for b in self.plot_layout:
            # print(b)
            # print(summary_result[b])

            np.save(b+'.txt', summary_result[b])

        return summary_result

    def display_results(self, plots):
        plt.figure(1)
        plt.rc('font', family='serif', size=12)
        line_style = '.-'
        plot_line_colors = ['r', 'b', 'g', 'k', 'm', 'c']
        fill_colors = ['red', 'blue', 'green', 'black', 'magenta', 'cyan']
        labels = {'average_delay': ['epoka', 'średnie opóźnienia [s]'],
                  'power_left': ['epoka', 'unorm. zużycie energii']}
        for i, p in enumerate(plots):
            plt.subplot(len(plots)*100 + 11 + i)
            legend = []
            for j in range(self.sim_numer):
                plt.plot(self.sim_sys[0].epoch_data['epoch'], self.plot_data[p][j][1], plot_line_colors[j] + line_style)
                plt.fill_between(self.sim_sys[0].epoch_data['epoch'], self.plot_data[p][j][0], self.plot_data[p][j][2],
                                 alpha=0.25, facecolor=fill_colors[j])
                legend.append(self.sim_sys[j].legend)

            plt.legend(legend, loc='best')
            plt.xlabel(labels[p][0])
            plt.ylabel(labels[p][1])
            plt.grid(True)
        plt.savefig("reward_function.png")
        plt.show()
        # input()


if __name__ == "__main__":
    logging.info("Program start")

    # Standard reinforcement learning with greedy policy
    # env = MultipleTests([SimulationProcess(TestPlan.StandardTest(e_greedy=0.1)),
    #                      SimulationProcess(TestPlan.StandardTest(e_greedy=0.5)),
    #                      SimulationProcess(TestPlan.StandardTest(e_greedy=0.9))])

    # Q-learning e_greedy
    # env = MultipleTests([SimulationProcess(TestPlan.QLearning(e_greedy=0.1)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.5)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.9))])

    # Q-learning e_greedy=0.9 learning_rate
    # env = MultipleTests([SimulationProcess(TestPlan.QLearning(e_greedy=0.9, learning_rate=.1)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.9, learning_rate=.5)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.9, learning_rate=.9)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.9, learning_rate=1))])

    # Q-learning e_greedy=0.9 learning_rate=1
    # env = MultipleTests([SimulationProcess(TestPlan.QLearning(e_greedy=0.9, learning_rate=1, discount_factor=0)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.9, learning_rate=1, discount_factor=.1)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.9, learning_rate=1, discount_factor=0.5)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.9, learning_rate=1, discount_factor=0.9)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.9, learning_rate=1, discount_factor=1))
    #                      ])

    # Best
    env = MultipleTests([SimulationProcess(TestPlan.QLearning(e_greedy=.99, learning_rate=1, discount_factor=0.5)),
                         # SimulationProcess(TestPlan.QLearning(e_greedy=.9, learning_rate=1, discount_factor=0.5)),
                         SimulationProcess(TestPlan.FixedDelay(delay=7))]
                        ,test_count=1)
    # env = MultipleTests([SimulationProcess(TestPlan.StandardTest(e_greedy=1))], test_count=3)

    env.start()

    env.display_results(['average_delay', 'power_left'])
