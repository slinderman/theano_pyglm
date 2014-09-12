import os
import numpy as np

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyglm.utils.io import parse_cmd_line_args, load_data
from pyglm.utils.sta import sta

# Parse command line args
(options, args) = parse_cmd_line_args()
# Load data and model
data = load_data(options)
stim = data['stim']
spks = data['S']

# Plot the STA at various lags
maxlag = 30
lags_to_plot = np.arange(maxlag, step=5)


# Downsample the spikes to the resolution of the stimulus
Tspks, N = spks.shape
Tstim = stim.shape[0]
# Flatten higher dimensional stimuli
if stim.ndim == 3:
    stimf = stim.reshape((Tstim, -1))
else:
    stimf = stim

s = sta(stimf,
        data,
        maxlag,
        Ns=np.arange(N))

# Get the limits by finding the max absolute value per neuron
s_max = np.amax(abs(s.reshape((N,-1))), axis=1)

plt.figure()
for n in range(min(N,5)):
    for j,l in enumerate(lags_to_plot):
        plt.subplot(N,len(lags_to_plot), n*len(lags_to_plot) + j + 1)
        # plt.title('N: %d, Lag: %d' % (n, j))
        plt.imshow(np.kron(s[n,l,:].reshape((10,10)), np.ones((10,10))),
                   vmin=-s_max[n], vmax=s_max[n],
                   cmap='RdGy')
        plt.xlabel(None)
        plt.ylabel(None)

        if j == len(lags_to_plot) - 1:
            plt.colorbar()

plt.savefig(os.path.join(options.resultsDir, 'sta.pdf'))