#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import logging
from symulator import System
import matplotlib.pyplot as plt
from test_plan import TestPlan
from timeit import default_timer as timer
from copy import deepcopy

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
        self.epoch_data = {
            'average_delay': [],
            'power_left': [],
            'quality_data': [],
            'epoch': []
        }

    def start_simulation(self):
        """
        Function for starting simulation
        """
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
            norm_power_left = float(embedded.power_left)/float(self.start_power)
            self.epoch_data['average_delay'] .append(avg_delay)
            self.epoch_data['power_left'].append(norm_power_left)
            self.epoch_data['epoch'].append(epoch)
            logging.info("Epoch " + '{:3}'.format(epoch) + " | Time: " + '{:05.2f}'.format(end - start) + " s" +
                         " | Avg_delay = " + '{:5.2f}'.format(avg_delay) + " s | Norm. power left = " +
                         '{:03.3f}'.format(norm_power_left))
        logging.info("LEARNING FINISHED! Elapsed time: " + '{:3.3f}'.format(timer()-sim_start))

    def display_epoch_data(self, plots):
        plt.figure(1)
        plot_layout = {'delays': ['epoch', 'average_delay', 'r--', 'epoch', 'average delays [s]'],
                       'power_left': ['epoch', 'power_left', 'bs', 'epoch', 'normalized power left']
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
    def __init__(self, simulation_systems):
        self.sim_sys = simulation_systems

    def start(self):
        for sim in self.sim_sys:
            # print(sim)
            sim.start_simulation()

    def display_results(self, plots):
        plt.figure(1)
        line_style = '.-'
        plot_line_colors = ['r', 'b', 'g', 'k', 'm', 'c']
        plot_layout = {'delays': ['epoch', 'average_delay', None, 'epoch', 'average delays [s]'],
                       'power_left': ['epoch', 'power_left', None, 'epoch', 'normalized power left']
                       }
        for i, p in enumerate(plots):
            plt.subplot(len(plots)*100 + 11 + i)
            legend = []
            for j, data in enumerate(self.sim_sys):
                plt.plot(data.epoch_data[plot_layout[p][0]], data.epoch_data[plot_layout[p][1]], plot_line_colors[j] +
                         line_style)
                legend.append(data.legend)

            plt.legend(legend, loc='best')
            plt.xlabel(plot_layout[p][3])
            plt.ylabel(plot_layout[p][4])
            plt.grid(True)
        plt.savefig("reward_function.png")
        plt.show()


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
    env = MultipleTests([SimulationProcess(TestPlan.StandardTest(e_greedy=0.9)),
                         SimulationProcess(TestPlan.QLearning(e_greedy=0.9, learning_rate=1, discount_factor=0.5)),
                         SimulationProcess(TestPlan.QLearning())
                         ])

    env.start()
    env.display_results(['delays', 'power_left'])
