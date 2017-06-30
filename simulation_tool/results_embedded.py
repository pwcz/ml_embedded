#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
import matplotlib.pyplot as plt
import numpy as np

delays = [
            4.0095652174,
            4.5290909091,
            4.1845238095,
            4.2502173913,
            3.285952381,
            4.3186046512,
            4.6223809524,
            4.8715217391,
            3.5864285714,
            4.0026086957,
            3.4287804878,
            4.2208163265,
            3.7517021277,
            4.4114285714,
            3.9215384615,
            3.7511363636,
            3.6792857143,
            4.0912244898,
            3.9291111111,
            3.5802325581,
            3.5748837209,
            4.37,
            3.9955263158,
            3.6757407407,
            3.5620930233,
            3.6697619048,
            4.5197959184,
            4.4002040816,
            4.9036956522,
            3.66395833333
        ]

power = [
        109.8873,
        107.1668,
        110.0452,
        118.0639,
        125.1494,
        124.8191,
        123.1294,
        118.5876,
        117.28371647,  117.05895324,  117.17719966,  107.35541936,
        115.99745138,  110.76051547,  123.07036964,  105.35225389,
        132.24228626,  110.57303141,  116.3444648 ,  104.28232378,
        119.23237017,  115.05638576,  114.58078244,  124.37966968,
        114.26632604,  111.79101764,  116.77338882,  106.4370584 ,
        103.94402981,  118.32731822
]


plt.figure(1)
plt.rc('font', family='serif', size=12)
line_style = '.-'
plot_line_colors = ['r', 'b', 'g', 'k', 'm', 'c']
fill_colors = ['red', 'blue', 'green', 'black', 'magenta', 'cyan']
labels = {'average_delay': ['epoka', 'średnie opóźnienia [s]'],
          'power_left': ['epoka', 'zużycie energii [J]'],
          'delay_median': ['epoka', 'mediana opóźnienia [s]'],
          'button_push_counter': ['epoka', 'liczba kar']}

epoch = range(30)
ax = plt.subplot(211)
fixed_mean = [4.5 for x in range(30)]
fixed_top = [4.5 + 0.2596 for x in range(30)]
fixed_bottom = [4.5 - 0.2596 for x in range(30)]
plt.plot(epoch, delays, plot_line_colors[0] + line_style)
plt.plot(epoch, fixed_mean, plot_line_colors[1] + line_style)
plt.fill_between(epoch, fixed_top, fixed_bottom, alpha=0.15, facecolor=fill_colors[1])

plt.legend(["Q-learning embedded", "stałe wybudznie 9s"], loc='best')
plt.plot(epoch, fixed_top, plot_line_colors[1], linewidth=0.5)
plt.plot(epoch, fixed_bottom, plot_line_colors[1], linewidth=0.5)
plt.ylabel(u'średnie opóźnienia [s]')

plt.xlim([1, 30])
plt.grid(True)

ax = plt.subplot(212)
fixed_power = [223.93 for x in range(30)]

plt.plot(epoch, power, plot_line_colors[0] + line_style)
plt.plot(epoch, fixed_power, plot_line_colors[1] + line_style)
plt.legend(["Q-learning embedded", "stałe wybudznie 9s"], loc='best')
# plt.xlim([0, 30])
# plt.xticks(np.arange(1, 25, 2))
plt.xlabel('epoka')
plt.ylabel(u'zużycie energii [J]')
# ax.get_yaxis().set_label_coords(-0.1, 0.5)
plt.grid(True)
plt.savefig("reward_function.png")

print("power mean = " + str(np.mean(power)))
print("delay mean = " + str(np.mean(delays)))
plt.show()