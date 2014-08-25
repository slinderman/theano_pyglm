"""
Build a class of plotting classes. These classes should be specific
to particular components or state variables. For example, we might
have a plotting class for the network. The classes should be initialized
with a model to determine how they should plot the results.

The classes should be able to:
 - plot either a single sample or the mean of a sequence of samples
   along with error bars.

 - take in an axis (or a figure) or create a new figure if not specified
 - take in a color or colormap for plotting


"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from hips.plotting.colormaps import gradient_cmap

from utils.theano_func_wrapper import seval

rwb_cmap = gradient_cmap([[1,0,0],
                          [1,1,1],
                          [0,0,0]])

class PlotProvider(object):
    """
    Abstract class for plotting a sample or a sequence of samples
    """
    def __init__(self, population):
        """
        Check that the model satisfies whatever criteria are appropriate
        for this model.
        """
        self.population = population

    def plot(self, sample, ax=None):
        """
        Plot the sample or sequence of samples
        """
        pass

class NetworkPlotProvider(PlotProvider):
    """
    Class to plot the connectivity network
    """

    def __init__(self, population):
        super(NetworkPlotProvider, self).__init__(population)

        # TODO: Check that the model has a network?
        # All models should have a network

    def plot(self, xs, ax=None, title=None, vmin=None, vmax=None, cmap=rwb_cmap):

        # Ensure sample is a list
        if not isinstance(xs, list):
            xs = [xs]

        # Get the weight matrix and adjacency matrix
        wvars = self.population.network.weights.get_variables()
        Ws = np.array([seval(self.population.network.weights.W,
                            wvars, x['net']['weights'])
                       for x in xs])

        gvars = self.population.network.graph.get_variables()
        As = np.array([seval(self.population.network.graph.A,
                             gvars,  x['net']['graph'])
                       for x in xs])


        # Compute the effective connectivity matrix
        W_inf = np.mean(Ws*As, axis=0)

        # Make sure bounds are set
        if None in (vmax,vmin):
            vmax = np.amax(np.abs(W_inf))
            vmin = -vmax

        # Create a figure if necessary
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        px_per_node = 10
        im = ax.imshow(np.kron(W_inf,np.ones((px_per_node,px_per_node))),
                       vmin=vmin, vmax=vmax,
                       extent=[0,1,0,1],
                       interpolation='nearest',
                       cmap=cmap)

        ax.set_title(title)


class LocationPlotProvider(PlotProvider):
    """
    Plot the latent locations of the neurons
    """
    def plot(self, xs, ax=None, name='location_provider', color='k'):
        """
        Plot a histogram of the inferred locations for each neuron
        """
        # Ensure sample is a list
        if not isinstance(xs, list):
            xs = [xs]

        if name not in xs[0]['latent']:
            return

        # Get the locations
        loccomp = self.population.latent.latentdict[name]
        locvars = loccomp.get_variables()
        Ls = np.array([seval(loccomp.Lmatrix,
                            locvars, x['latent'][name])
                       for x in xs])
        [N_smpls, N, D] = Ls.shape

        for n in range(N):
            # plt.subplot(1,N,n+1, aspect=1.0)
            # plt.title('N: %d' % n)

            if N_smpls == 1:
                if D == 1:
                    plt.plot([Ls[0,n,0], Ls[0,n,0]],
                             [0,2], color=color, lw=2)
                elif D == 2:
                    ax.plot(Ls[0,n,1], Ls[0,n,0], 's',
                             color=color, markerfacecolor=color)
                    ax.text(Ls[0,n,1]+0.25, Ls[0,n,0]+0.25, '%d' % n,
                             color=color)

                    # TODO: Fix the limits!
                    ax.set_xlim((-0.5, 4.5))
                    ax.set_ylim((4.5, -0.5))
                else:
                    raise Exception("Only plotting locs of dim <= 2")
            else:
                # Plot a histogram of samples
                if D == 1:
                    ax.hist(Ls[:,n,0], bins=20, normed=True, color=color)
                elif D == 2:
                    ax.hist2d(Ls[:,n,1], Ls[:,n,0], bins=np.arange(-0.5,5), cmap='Reds', alpha=0.5, normed=True)
                    ax.set_xlim((-0.5, 4.5))
                    ax.set_ylim((4.5, -0.5))

                    # ax.colorbar()
                else:
                    raise Exception("Only plotting locs of dim <= 2")



