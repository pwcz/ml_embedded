#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import datetime
import csv
import logging
import socket
from symulator import System
from test_plan import TestPlan
from timeit import default_timer as timer
from copy import deepcopy
import numpy as np
import sys
import os

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


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
        self.sim_type = _test_plan.get_type()

    def start_simulation(self):
        """
        Function for starting simulation
        """
        # prepare data
        self.epoch_data = {
            'average_delay': [],
            'power_left': [],
            'quality_data': [],
            'epoch': [],
            'button_push_counter': [],
            'delay_median': []
        }

        # get training data
        sim_start = timer()
        training_data_gold = self.training_data_generator.get_train_data()
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
                # print(current_time)
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
            self.epoch_data['average_delay'] .append(avg_delay)
            self.epoch_data['power_left'].append(embedded.power_left)
            self.epoch_data['epoch'].append(epoch)
            self.epoch_data['button_push_counter'].append(embedded.button_push_counter)
            if self.sim_type == "transmission":
                self.epoch_data['delay_median'].append(0)
            else:
                self.epoch_data['delay_median'].append(np.median(data_record['delays_data']))
            logging.debug("Epoch " + '{:3}'.format(epoch) + " | Time: " + '{:05.2f}'.format(end - start) + " s" +
                          " | Avg_delay = " + '{:5.2f}'.format(avg_delay) + " s | Norm. power consumption = " +
                          '{:03.3f}'.format(embedded.power_left))
        logging.info("LEARNING FINISHED! Elapsed time: " + '{:3.3f}'.format(timer() - sim_start))
        logging.info("| Epoch number: " + str(len(self.epoch_data['epoch'])) + " | " + self.logger_info + " | Time: " +
                     '{:05.2f} s'.format(end - start) + " | avg. delay: " +
                     '{:05.2f}'.format(np.mean(self.epoch_data['average_delay'][-10:])) + " | avg. power consum.: " +
                     '{:05.2f}'.format(np.mean(self.epoch_data['power_left'][-10:])) + " |"
                     )
        try:
            embedded.learning_a.show_q_data()
        except AttributeError:
            pass


class MultipleTests:
    def __init__(self, simulation_systems, test_count=1):
        self.sim_sys = simulation_systems
        self.test_count = test_count
        self.plot_layout = ['average_delay', 'power_left', 'button_push_counter','delay_median']
        self.sim_numer = len(simulation_systems)
        self.result = [[{} for _ in range(self.test_count)] for _ in range(self.sim_numer)]
        self.plot_data = None

    def start(self):
        for k, sim in enumerate(self.sim_sys):
            for m in range(self.test_count):
                sim.start_simulation()
                for j in self.plot_layout:
                    self.result[k][m][j] = deepcopy(sim.epoch_data[j])
        self.plot_data = self.prepare_data()

    def get_result_data(self):
        self.start()
        data = []
        for j in self.plot_layout:
            data.append(deepcopy(np.mean(self.plot_data[j][0][1][-10:])))
        return data

    def prepare_data(self):
        epoch_number = len(self.result[0][0][self.plot_layout[0]])
        summary_result = {x: np.empty([self.sim_numer, 3, epoch_number]) for x in self.plot_layout}
        last_epoch_data = {x: np.empty([self.sim_numer, self.test_count]) for x in self.plot_layout}
        for m, test in np.ndenumerate(self.result):                     # loop through tests
            for epoch in range(epoch_number):                           # loop through epoch number
                dd = {x: [] for x in self.plot_layout}
                for repeat_num in range(self.test_count):               # loop through test count
                    for data_taken in self.plot_layout:
                        dd[data_taken].append(self.result[m[0]][repeat_num][data_taken][epoch])
                for b in self.plot_layout:
                    standard_deviation = np.std(dd[b])
                    mean = np.mean(dd[b])
                    summary_result[b][m[0]][0][epoch] = mean - standard_deviation
                    summary_result[b][m[0]][1][epoch] = mean
                    summary_result[b][m[0]][2][epoch] = mean + standard_deviation

        for m, test in np.ndenumerate(self.result):
            for data_taken in self.plot_layout:
                for repeat_num in range(self.test_count):
                    last_epoch_data[data_taken][m[0]][repeat_num] = \
                        self.result[m[0]][repeat_num][data_taken][epoch_number - 1]

        for m, test in np.ndenumerate(self.result):
            csv_file = []
            for repeat_num in range(self.test_count):
                tt_table = []
                for data_taken in self.plot_layout:
                    tt_table.append(last_epoch_data[data_taken][m[0]][repeat_num])
                csv_file.append(tt_table)
            with open('prepare_data_' + str(m[0]) + '.csv', 'w+', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_file)

        print(last_epoch_data)

        return summary_result

    def display_results(self, plots):
        import matplotlib
        matplotlib.use('TkAgg')
        matplotlib.rcParams.update({'font.size': 22})
        matplotlib.rc('xtick', labelsize=12)
        matplotlib.rc('ytick', labelsize=12)
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.rc('font', family='serif', size=12)
        line_style = '.-'
        plot_line_colors = ['r', 'b', 'g', 'k', 'm', 'c']
        fill_colors = ['red', 'blue', 'green', 'black', 'magenta', 'cyan']
        labels = {'average_delay': ['epoka', 'średnie opóźnienia [s]'],
                  'power_left': ['epoka', 'zużycie energii [J]'],
                  'delay_median': ['epoka', 'mediana opóźnienia [s]'],
                  'button_push_counter': ['epoka', 'liczba kar']}
        for i, p in enumerate(plots):
            ax = plt.subplot(len(plots)*100 + 11 + i)
            legend = []
            for j in range(self.sim_numer):
                plt.plot(self.sim_sys[0].epoch_data['epoch'], self.plot_data[p][j][1], plot_line_colors[j] + line_style)
                plt.fill_between(self.sim_sys[0].epoch_data['epoch'], self.plot_data[p][j][0], self.plot_data[p][j][2],
                                 alpha=0.15, facecolor=fill_colors[j])
                legend.append(self.sim_sys[j].legend)

            plt.legend(legend, loc='best')
            for j in range(self.sim_numer):
                plt.plot(self.sim_sys[0].epoch_data['epoch'], self.plot_data[p][j][0], plot_line_colors[j],
                         linewidth=0.5)
                plt.plot(self.sim_sys[0].epoch_data['epoch'], self.plot_data[p][j][2], plot_line_colors[j],
                         linewidth=0.5)
            if i == len(plots) - 1:
                plt.xlabel(labels[p][0])
            plt.ylabel(labels[p][1])
            ax.get_yaxis().set_label_coords(-0.1, 0.5)
            plt.grid(True)
        plt.savefig("reward_function.png")
        plt.show()


def primitive_hill_climb(split_number=10, current_number=0, logs_name="test"):
    single_epoch_time = 3
    test_count = 3
    _epoch_number = 80
    result_tab = []
    if socket.gethostname() == 'NERVA':
        verify = True
    else:
        verify = False

    delay_award = np.arange(500, 1000, 25)  # penalty for delay
    award_time_threshold = np.arange(3.5, 9, 0.25)  # np.arange(3.5, 9, 0.3)  # [4.5, 5.5, 6.5, 7.5]  # award time threshold
    awake_penalty = np.arange(5, 50, 5)  # penalty for unnecessary awake
    awake_award = np.arange(300, 700, 25)  # award for necessary awake
    e_greedy = [1]  # np.arange(0, 1.05, 0.05)  # eps for greedy strategy
    discount_factor = [0.45]  # np.arange(0, 1.05, 0.025)  # [0.05]
    learning_rate = [0.35]  # np.arange(0.05, 1.05, 0.025)
    parameters = cartesian(delay_award, awake_penalty, awake_award, e_greedy, award_time_threshold, discount_factor,
                           learning_rate)
    directory = 'results/' + datetime.datetime.now().strftime("%d_%B_%Y") + logs_name  # _%H_%M")

    elapsed_time = single_epoch_time * len(parameters) * test_count
    parameters = np.array(parameters)
    logging.info("Parameters number = " + str(len(parameters)))
    logging.info("Time elapse (sum) = " + str(int(elapsed_time/3600)) + ":" +
                 "{0:0>2}".format(int(elapsed_time/60) % 60) +
                 ":" + "{0:0>2}".format((int(elapsed_time % 60))))
    elapsed_time /= split_number
    logging.info("Time elapse (single core) = " + str(int(elapsed_time/3600)) + ":" +
                 "{0:0>2}".format(int(elapsed_time/60) % 60) +
                 ":" + "{0:0>2}".format((int(elapsed_time % 60))))
    if verify:
        sys.exit()
    parameters = np.array_split(parameters, split_number)

    for da, ap, aa, e, adt, df, lr in parameters[current_number]:
        # sys_engine = MultipleTests([SimulationProcess(TestPlan.FixedDelay(delay=e, epoch=_epoch_number))],
        #                            test_count=test_count)
        sys_engine = MultipleTests([SimulationProcess(
            TestPlan.QLearning(e_greedy=e, learning_rate=lr, discount_factor=df, awake_award=aa,
                               awake_penalty=ap, delay_award=da, delay_threshold=adt, epoch=_epoch_number))],
            test_count=test_count)
        sys_engine_result = sys_engine.get_result_data()
        result_tab.append([da, ap, aa, e, adt, df, lr] + sys_engine_result)
    print(result_tab)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = directory + "/" + datetime.datetime.now().strftime("%H_%M") + '_results_' + str(current_number) + '.csv'
    logging.info("write result to:" + str(file_name))
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result_tab)


def cartesian(*arrays):
    mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
    dim = len(mesh)  # number of dimensions
    elements = mesh[0].size  # number of elements, any index will do
    flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
    reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
    return reshape

if __name__ == "__main__":
    logging.info("Program start")
    if len(sys.argv) > 1:
        logging.info("This is the name of the script: " + str(sys.argv[0]))
        logging.info("Number of arguments: " + str(len(sys.argv)))
        logging.info("The arguments are: " + str(sys.argv))
        primitive_hill_climb(current_number=int(sys.argv[1]), split_number=int(sys.argv[2]), logs_name=sys.argv[3])
        sys.exit()

    #  generate plot for fixed parameters
    _epoch_number = 150
    _test_number = 10
    # _epoch_number = 100
    # _test_number = 3
    # env = MultipleTests([SimulationProcess(TestPlan.QLearning(e_greedy=0.1, learning_rate=.3, discount_factor=0.05,
    #                                                           awake_award=500, awake_penalty=21, delay_award=475,
    #                                                           delay_threshold=7.5, epoch=_epoch_number))]
    #                     , test_count=_test_number)

    # env = MultipleTests([SimulationProcess(TestPlan.QLearning(e_greedy=0.1, learning_rate=.5, discount_factor=0.05,
    #                                                           awake_award=600, awake_penalty=35, delay_award=450,
    #                                                           delay_threshold=5, epoch=_epoch_number)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.1, learning_rate=.85, discount_factor=0.05,
    #                                                           awake_award=600, awake_penalty=35, delay_award=450,
    #                                                           delay_threshold=5, epoch=_epoch_number)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.1, learning_rate=.85, discount_factor=0.05,
    #                                                           awake_award=625, awake_penalty=35, delay_award=450,
    #                                                           delay_threshold=6, epoch=_epoch_number)),
    #                      SimulationProcess(TestPlan.QLearning(e_greedy=0.1, learning_rate=.85, discount_factor=0.05,
    #                                                           awake_award=625, awake_penalty=40, delay_award=475,
    #                                                           delay_threshold=6, epoch=_epoch_number))]
    #                     , test_count=_test_number)

    # env = MultipleTests([SimulationProcess(TestPlan.QLearning(e_greedy=1, learning_rate=.6, discount_factor=0.5,
    #                                                           awake_award=500, awake_penalty=19, delay_award=875,
    #                                                           delay_threshold=7.5, epoch=_epoch_number,
    #                                                           states_number=72 * 2)),
    #                      SimulationProcess(TestPlan.FixedDelay(delay=9, epoch=_epoch_number))]
    #                     , test_count=_test_number)

    env = MultipleTests([SimulationProcess(TestPlan.QLearning(e_greedy=0.05, learning_rate=.35, discount_factor=0.45,
                                                              awake_award=575, awake_penalty=40, delay_award=500,
                                                              delay_threshold=8, epoch=_epoch_number,
                                                              states_number=40)),
                        SimulationProcess(TestPlan.QLearning(e_greedy=0.1, learning_rate=.35, discount_factor=0.45,
                                                             awake_award=575, awake_penalty=40, delay_award=500,
                                                             delay_threshold=8, epoch=_epoch_number, states_number=40)),
                        SimulationProcess(TestPlan.QLearning(e_greedy=0.15, learning_rate=.35, discount_factor=0.45,
                                                             awake_award=575, awake_penalty=40, delay_award=500,
                                                             delay_threshold=8, epoch=_epoch_number, states_number=40)),
                        SimulationProcess(TestPlan.QLearning(e_greedy=0.2, learning_rate=.35, discount_factor=0.45,
                                                             awake_award=575, awake_penalty=40, delay_award=500,
                                                             delay_threshold=8, epoch=_epoch_number, states_number=40)),
                        SimulationProcess(TestPlan.QLearning(e_greedy=0.25, learning_rate=.35, discount_factor=0.45,
                                                             awake_award=575, awake_penalty=40, delay_award=500,
                                                             delay_threshold=8, epoch=_epoch_number, states_number=40)),
                         SimulationProcess(TestPlan.FixedDelay(delay=9, epoch=_epoch_number))]
                        , test_count=_test_number)
    # env = MultipleTests([SimulationProcess(TestPlan.QLearning(e_greedy=1, learning_rate=.35, discount_factor=0.45,
    #                                                           awake_award=575, awake_penalty=40, delay_award=500,
    #                                                           delay_threshold=8, epoch=_epoch_number,
    #                                                           states_number=40))]
    #                     , test_count=_test_number)

    # env = MultipleTests([SimulationProcess(TransmissionLearning.QLearning(e_greedy=1, learning_rate=.35,
    #                                                                       discount_factor=0.45, lmode='ml',
    #                                                                       epoch_number=_epoch_number)),
    #                      SimulationProcess(TransmissionLearning.QLearning(e_greedy=.1, learning_rate=.5,
    #                                                                       discount_factor=0.5, lmode='sms',
    #                                                                       epoch_number=_epoch_number)),
    #                      SimulationProcess(TransmissionLearning.QLearning(e_greedy=.1, learning_rate=.5,
    #                                                                       discount_factor=0.5, lmode='timeout',
    #                                                                       epoch_number=_epoch_number))]
    #                     , test_count=_test_number)
    # env = MultipleTests([SimulationProcess(TransmissionLearning.QLearning(e_greedy=1, learning_rate=.35,
    #                                                                       discount_factor=0.45, lmode='ml',
    #                                                                       epoch_number=_epoch_number)),
    #                     SimulationProcess(TransmissionLearning.QLearning(e_greedy=.1, learning_rate=.5,
    #                                                                      discount_factor=0.5, lmode='sms',
    #                                                                      epoch_number=_epoch_number))]
    #                     , test_count=_test_number)

    # env = MultipleTests([SimulationProcess(TransmissionLearning.QLearning(e_greedy=.1, learning_rate=.5,
    #                                                                       discount_factor=0.5, lmode='sms'))]
    #                     , test_count=_test_number)

    env.start()
    env.display_results(['average_delay', 'power_left'])
