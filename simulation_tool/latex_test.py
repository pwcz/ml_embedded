import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import pylab as plt
plt.figure(1)

string = r'z=${value}^{upper}_{lower}$'.format(
                value='{' + str(0.27) + '}',
                upper='{+' + str(0.01) + '}',
                lower='{-' + str(0.01) + '}')
print(string)

fig = plt.figure(figsize=(3,1))
fig.text(0.1, 0.5, string, size=24, va='center')
fig.show()
input()
