""" Plot the inference results
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')       # To enable saving remotely
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Get a red-white-black cmap
cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.5, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.5, 1.0, 1.0),
                 (1.0, 0.0, 0.0))}
cmap = LinearSegmentedColormap('my_colormap',cdict,256)

from utils.avg_dicts import average_list_of_dicts, std_list_of_dicts

import cPickle

def plot_connectivity_matrix(s_smpls, s_true=None):
    
    # Compute the inferred connectivity matrix
    W_inf = np.zeros_like(s_smpls[0]['net']['weights']['W'])
    for smpl in s_smpls:
        W_inf += smpl['net']['weights']['W'] * smpl['net']['graph']['A']
    W_inf /= len(s_smpls)

    true_given = s_true is not None
    if true_given:
        plt.subplot(1,2,1)
        W_true = s_true['net']['weights']['W'] * s_true['net']['graph']['A']
        W_max = np.amax(np.maximum(np.abs(W_true),np.abs(W_inf)))
    else:
        W_max = np.amax(np.abs(W_inf))
        
    px_per_node = 10

    if true_given:
        plt.imshow(np.kron(W_true,np.ones((px_per_node,px_per_node))),
                   vmin=-W_max,vmax=W_max,
                   extent=[0,1,0,1],
                   interpolation='nearest',
                   cmap=cmap)
        plt.colorbar()
        plt.title('True Network')
        plt.subplot(1,2,2)

    # Plot the inferred network
    plt.imshow(np.kron(W_inf,np.ones((px_per_node,px_per_node))),
               vmin=-W_max,vmax=W_max,
               extent=[0,1,0,1],
               interpolation='nearest',
               cmap=cmap)
    plt.colorbar()
    plt.title('Inferred Network')
    
def plot_stim_response(s_glm, s_glm_std=None, color=None):
    if 'stim_response_t' in s_glm['bkgd'].keys() and \
       'stim_response_x' in s_glm['bkgd'].keys():
      
        # Get the stimulus responses
        stim_x = s_glm['bkgd']['stim_response_x']
        stim_t = s_glm['bkgd']['stim_response_t']
            
        # Plot the spatial component of the stimulus response
        plt.subplot(1,2,1)
        if len(stim_x.shape) >= 2:
            px_per_node = 10
            stim_x_max = np.amax(np.abs(stim_x))
            plt.imshow(np.kron(stim_x,np.ones((px_per_node,px_per_node))),
                       vmin=-stim_x_max,vmax=stim_x_max,
                       extent=[0,1,0,1],
                       interpolation='nearest')
            plt.colorbar()
        else:
            plt.plot(stim_x, color=color, linestyle='-')
            plt.hold(True)

            # If standard deviation is given, plot that as well
            if s_glm_std is not None:
                stim_x_std = s_glm_std['bkgd']['stim_response_x']
                plt.plot(stim_x + 2*stim_x_std, color=color, linestyle='--') 
                plt.plot(stim_x - 2*stim_x_std, color=color, linestyle='--')
            
        plt.subplot(1,2,2)
        plt.plot(stim_t,'-r')
        plt.hold(True)
        if s_glm_std is not None:
            stim_t_std = s_glm_std['bkgd']['stim_response_t']
            plt.plot(stim_t + 2*stim_t_std, color=color, linestyle='--') 
            plt.plot(stim_t - 2*stim_t_std, color=color, linestyle='--')
            
    elif 'stim_response' in s_glm['bkgd'].keys():
        stim_t = s_glm['bkgd']['stim_response']
        plt.plot(stim_t, color=color, linestyle='-')
        plt.hold(True)
        if s_glm_std is not None:
            stim_t_std = s_glm_std['bkgd']['stim_response']
            plt.plot(stim_t + 2*stim_t_std, color=color, linestyle='--') 
            plt.plot(stim_t - 2*stim_t_std, color=color, linestyle='--')

def plot_imp_responses(s_inf, s_std=None, fig=None, color=None, use_bgcolor=False, linestyle='-', dt=0.001):
    """ Plot the impulse responses plus or minus two standard devs
    """ 


    # Get the weights of the impulse responses
    W_inf = s_inf['net']['weights']['W'] * s_inf['net']['graph']['A']
    N = W_inf.shape[0]
    
    s_imps = []
    s_imps_std = []
    for n_post in np.arange(N):
        s_imp_row = []
        s_imp_std_row = []
        
        s_imp_n = s_inf['glms'][n_post]['imp']['impulse']
        if s_std is not None:
            s_imp_std_n = s_std['glms'][n_post]['imp']['impulse']

        for n_pre in np.arange(N):
            w = W_inf[n_pre,n_post]
            s_imp_row.append(w*s_imp_n[n_pre,:])
            if s_std is not None:
                s_imp_std_row.append(w*s_imp_std_n[n_pre,:])

        s_imps.append(s_imp_row)
        s_imps_std.append(s_imp_std_row)
        
    s_imps = np.array(s_imps)    
    s_imps_std = np.array(s_imps_std)
    
    # Transpose so that pre is row index 
    s_imps = np.transpose(s_imps, [1,0,2])
    if s_std is not None:
        s_imps_std = np.transpose(s_imps_std, [1,0,2])
    else:
        s_imps_std = np.zeros_like(s_imps)
    imp_max = np.amax(np.abs(s_imps+2*s_imps_std))

    t_imp = dt*np.arange(s_imps.shape[2])
    W_imp = np.trapz(s_imps,t_imp, axis=2)
    W_imp_max = np.amax(W_imp)

    # Create a figure if necessary
    if fig is None:
        fig = plt.figure()

    for n_pre in np.arange(N):
        for n_post in np.arange(N):
            ax = fig.add_subplot(N,N,n_pre*N+n_post + 1)
            if use_bgcolor:
                # Set background color based on weight of impulse
                bkgd_color = cmap((W_imp[n_pre,n_post] -(-W_imp_max))/(2*W_imp_max))
                # Set it slightly transparent
                tcolor = list(bkgd_color)
                tcolor[3] = 0.75
                tcolor = tuple(tcolor)
                ax.set_axis_bgcolor(tcolor)


            # Plot the inferred impulse response
            ax.hold(True)
            ax.plot(np.squeeze(s_imps[n_pre,n_post,:]),color=color, linestyle=linestyle)

            # Plot plus or minus 2 stds
            if s_std is not None:
                ax.plot(np.squeeze(s_imps[n_pre,n_post,:] +
                                    2*s_imps_std[n_pre,n_post,:]),
                                    color=color, 
                                    linestyle='--')
                ax.plot(np.squeeze(s_imps[n_pre,n_post,:] -
                                    2*s_imps_std[n_pre,n_post,:]),
                                    color=color, 
                                    linestyle='--')

            ax.plot(np.zeros_like(np.squeeze(s_imps[n_pre,n_post,:])),
                     color='k', linestyle=':')

            # Set labels 
            #if not (n_pre == N-1 and n_post == 0):
            if True:
                ax.set_xlabel("")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel("")
            ax.set_ylim(-imp_max,imp_max)

    # Add a colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])

    # Rather than using the colorbar method, directly
    # instantiate a colorbar
    from matplotlib.colorbar import ColorbarBase
    cbar_ticks = np.array([-0.9*W_imp_max, 0.0, 0.9*W_imp_max]).round(2)
    cbar = ColorbarBase(cbar_ax, cmap=cmap,
                        values=np.linspace(-W_imp_max, W_imp_max, 500),
                        boundaries=np.linspace(-W_imp_max, W_imp_max, 500),
                        ticks=cbar_ticks)

    return fig

def plot_imp_responses_fast(s_inf, s_std=None, fig=None, color=None, use_bgcolor=False, linestyle='-'):
    """ Plot the impulse responses plus or minus two standard devs
        In this case we use a single axes rather than multiple subplots since
        matplotlib is, unfortunately, ridiculously slow when it comes to large
        numbers of subplots
    """
    # Get the weights of the impulse responses
    W_inf = s_inf['net']['weights']['W'] * s_inf['net']['graph']['A']
    N = W_inf.shape[0]
    
    s_imps = []
    s_imps_std = []
    for n_post in np.arange(N):
        s_imp_row = []
        s_imp_std_row = []
        
        s_imp_n = s_inf['glms'][n_post]['imp']['impulse']
        if s_std is not None:
            s_imp_std_n = s_std['glms'][n_post]['imp']['impulse']

        for n_pre in np.arange(N):
            w = W_inf[n_pre,n_post]
            s_imp_row.append(w*s_imp_n[n_pre,:])
            if s_std is not None:
                s_imp_std_row.append(w*s_imp_std_n[n_pre,:])

        s_imps.append(s_imp_row)
        s_imps_std.append(s_imp_std_row)
        
    s_imps = np.array(s_imps)    
    s_imps_std = np.array(s_imps_std)
    
    # Transpose so that pre is row index 
    s_imps = np.transpose(s_imps, [1,0,2])
    if s_std is not None:
        s_imps_std = np.transpose(s_imps_std, [1,0,2])
    else:
        s_imps_std = np.zeros_like(s_imps)
    imp_max = np.amax(np.abs(s_imps+2*s_imps_std))
        
    W_imp = np.sum(s_imps,2)
    W_imp_max = np.amax(W_imp)

    # Create a figure if necessary
    if fig is None:
        fig = plt.figure()

    # Get subplot sizes
    x_sz = s_imps.shape[2]     # Number of time bins per impulse response
    y_sz = 2*W_imp_max         # Amplitude of impulse responses
    x_buf = .05*x_sz            # x buffer on either side of figure 
    y_buf = .05*y_sz            # y buffer on top and bottom of figure
    
    ax = plt.subplot(111)
    plt.setp(ax, 'frame_on', False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.hold(True)

    from matplotlib.patches import Rectangle

    for n_pre in np.arange(N):
        for n_post in np.arange(N):
            x_foff = (x_sz + 2*x_buf) * n_post   # Offset of this subfigure
            y_foff = (y_sz + 2*y_buf) * n_pre 
            x_aoff = x_foff + x_buf              # Offset of this axis
            y_aoff = y_foff + y_buf
            if use_bgcolor:
                # Add a semitransparent patch for the background
                # Set background color based on weight of impulse
                bkgd_color = cmap((W_imp[n_pre,n_post] -(-W_imp_max))/(2*W_imp_max))
                # Set it slightly transparent
                tcolor = list(bkgd_color)
                tcolor[3] = 0.75
                tcolor = tuple(tcolor)
                #ax.set_axis_bgcolor(tcolor)
                ax.add_patch(Rectangle((x_foff,y_foff+y_sz+2*y_buf),  # Lower left coordinate
                                       x_sz+2*x_buf,                  # width
                                       y_sz+2*y_buf,                  # height
                                       alpha=0.75,
                                       color=tcolor,
                                       fill=True))


            # Plot the inferred impulse response
            ax.plot(x_aoff + np.arange(x_sz),
                    y_aoff + np.squeeze(s_imps[n_pre,n_post,:]),
                    color=color, linestyle=linestyle)

            # Plot plus or minus 2 stds
            if s_std is not None:
                ax.plot(x_aoff + np.arange(x_sz),
                        y_aoff + np.squeeze(s_imps[n_pre,n_post,:] +
                                            2*s_imps_std[n_pre,n_post,:]),
                        color=color, 
                        linestyle='--')
                ax.plot(x_aoff + np.arange(x_sz),
                        y_aoff + np.squeeze(s_imps[n_pre,n_post,:] -
                                            2*s_imps_std[n_pre,n_post,:]),
                        color=color, 
                        linestyle='--')

            ax.plot(x_aoff + np.arange(x_sz),
                    y_aoff + np.zeros_like(np.squeeze(s_imps[n_pre,n_post,:])),
                    color='k', linestyle=':')

    return fig


def plot_firing_rate(s_glm, s_glm_std=None, color=None, tt=None, T_lim=None):
    if tt is None:
        tt = np.arange(np.size(s_glm['lam']))
    if T_lim is None:
        T_lim = slice(0,np.size(s_glm['lam']))
    plt.plot(tt[T_lim], s_glm['lam'][T_lim],
             color=color)
    plt.hold(True)
    
    if s_glm_std is not None:
        # Make a shaded patch for the error bars
        from matplotlib.patches import Polygon
        verts = list(zip(tt[T_lim], s_glm['lam'][T_lim] + 2*s_glm_std['lam'][T_lim])) + \
                list(zip(tt[T_lim][::-1],s_glm['lam'][T_lim][::-1] - 2*s_glm_std['lam'][T_lim][::-1]))
        poly = Polygon(verts, facecolor=color, edgecolor=color, alpha=0.5)
        plt.gca().add_patch(poly)

#        plt.plot(s_glm['lam'] + 2*s_glm_std['lam'],
#                 color=color,
#                 linestyle='--')
#        plt.plot(s_glm['lam'] - 2*s_glm_std['lam'],
#                 color=color,
#                 linestyle='--')

def plot_ks(s_glm, S, dt, s_glm_std=None, color=None):
    """ Plot a Kolmogorov-Smirnov goodness of fit test..
    """
    lam = s_glm['lam']
    # Cumulative integral of fr
    I = dt * np.cumsum(lam);

    # Find rescaled spike times
    rescaled_isi = np.diff(I[S])

    # For a PP the cdf is of the exponential distribution
    z = 1-np.exp(-rescaled_isi);
    z = np.sort(z)
    N = len(z)

    ez = (np.arange(1,N+1)-.5)/N
    plt.plot(ez,ez,'k');
    plt.hold(True)

    # The 95% confidence interval is approximately ez+-1.36/sqrt(N)
    plt.plot(ez,ez+1.36/np.sqrt(N),'--k')
    plt.plot(ez,ez-1.36/np.sqrt(N),'--k')

    # Plot the actual statistic
    plt.plot(z,ez,'-b');

    plt.ylim([0,1])
    plt.xlim([0,1])
    
    # Check if the test passes
    test_passed = np.all(np.abs(z-ez)<1.36/np.sqrt(N))
    return test_passed

def plot_basis(s_glm, color='k'):
    if 'basis' in s_glm['glms'][0]['imp']:
        plt.plot(s_glm['glms'][0]['imp']['basis'],
                 color=color)

def plot_log_prob(s_inf, key='logp', s_true=None, color='r'):
    inf_lp_trace = np.array([s[key] for s in s_inf])

    if len(inf_lp_trace) > 1:
        plt.plot(inf_lp_trace, color=color)
        plt.xlabel('Iteration')
    else:
        plt.bar(0, inf_lp_trace[0], color=color)

    if s_true is not None:
        plt.hold(True)
        true_lp_trace = s_true[key] * np.ones_like(inf_lp_trace)
        if len(inf_lp_trace) > 1:
            plt.plot(true_lp_trace, color='k')
        else:
            plt.bar(1, true_lp_trace[0], color='k')

    plt.ylabel('Log probability')

def plot_log_lkhd(s_inf, s_true=None,  color='k'):
    plot_log_prob(s_inf, key='ll', s_true=s_true, color=color)
    plt.ylabel('Log likelihood')

def plot_results(population, 
                 x_inf, 
                 popn_true=None, 
                 x_true=None, 
                 resdir=None,
                 do_plot_connectivity=True,
                 do_plot_stim_resp=True,
                 do_plot_imp_responses=True,
                 do_plot_firing_rates=True,
                 do_plot_ks=True,
                 do_plot_logpr=True):
    """ Plot the inferred stimulus tuning curves and impulse responses
    """
    if not resdir:
        resdir = '.'

    true_given = x_true is not None and popn_true is not None
    
    # Make sure we have a list of x's
    if not isinstance(x_inf, list):
        x_inf = [x_inf]

    # Evaluate the state for each of the parameter settings
    N_samples = len(x_inf)
    s_inf = []
    for x in x_inf:
        s_inf.append(population.eval_state(x))
    
    s_true = None
    if true_given:
        s_true = popn_true.eval_state(x_true)

    # Average the inferred states
    s_avg = average_list_of_dicts(s_inf)
    s_std = std_list_of_dicts(s_inf, s_avg)
    N = population.N

    # TODO Fix the averaging of W and A
    # E[W] * E[A] != E[W*A]
    # Plot the inferred connectivity matrix
    if do_plot_connectivity:
        print "Plotting connectivity matrix"
        f = plt.figure()
        plot_connectivity_matrix(s_inf, s_true)
        f.savefig(os.path.join(resdir,'conn.pdf'))
        plt.close(f)

    # Plot stimulus response functions
    if do_plot_stim_resp:
        print "Plotting stimulus response functions"
        for n in range(N):
            f = plt.figure()
            plot_stim_response(s_avg['glms'][n], 
                               s_glm_std=s_std['glms'][n],
                               color='r')
            if true_given:
                plot_stim_response(s_true['glms'][n], 
                                   color='k')
        
            f.savefig(os.path.join(resdir,'stim_resp_%d.pdf' % n))
            plt.close(f)
        
    # Plot the impulse responses
    if do_plot_imp_responses:
        print "Plotting impulse response functions"
        f = plt.figure()
        plot_imp_responses(s_avg,
                           s_std,
                           fig=f,
                           color='r',
                           use_bgcolor=True)
        if true_given:
            plot_imp_responses(s_true,
                               fig=f,
                               color='k',
                               linestyle='--',
                               use_bgcolor=False)
            
        f.savefig(os.path.join(resdir,'imp_resp.pdf'))
        plt.close(f)
    
    # Plot the impulse response basis
    if do_plot_imp_responses:
        f = plt.figure()
        plot_basis(s_avg)
        f.savefig(os.path.join(resdir,'imp_basis.pdf'))
        plt.close(f)
    

    # Plot the firing rates
    if do_plot_firing_rates:
        print "Plotting firing rates"
        T_lim = slice(0,2000)
        for n in range(N):
            f = plt.figure()
            plot_firing_rate(s_avg['glms'][n], 
                             s_std['glms'][n], 
                             color='r',
                             T_lim=T_lim)
            if true_given:
                plot_firing_rate(s_true['glms'][n], color='k', T_lim=T_lim)
            
            # Plot the spike times
            St = np.nonzero(population.glm.S.get_value()[T_lim,n])[0]
            plt.plot(St,s_avg['glms'][n]['lam'][T_lim][St],'ko')
            
            plt.title('Firing rate %d' % n)
            
            f.savefig(os.path.join(resdir,'firing_rate_%d.pdf' % n))
            plt.close(f)

    if do_plot_ks:
        print "Plotting KS test results"
        for n in range(N):
            f = plt.figure()
            St = np.nonzero(population.glm.S.get_value()[:,n])[0]
            plot_ks(s_avg['glms'][n], St, population.glm.dt.get_value())
            f.savefig(os.path.join(resdir, 'ks_%d.pdf' %n))
            plt.close(f)

    if do_plot_logpr:
        print "Plotting log probability and log likelihood trace"
        f = plt.figure()
        plot_log_prob(s_inf, s_true=s_true, color='r')
        f.savefig(os.path.join(resdir, 'log_prob.pdf'))
        plt.close(f)
        
        f = plt.figure()
        plot_log_lkhd(s_inf, s_true=s_true, color='r')
        f.savefig(os.path.join(resdir, 'log_lkhd.pdf'))
        plt.close(f)

        if 'logprior' in s_inf[0]:
            f = plt.figure()
            plot_log_prob(s_inf, key='logprior', s_true=s_true, color='r')
            plt.ylabel('Log prior')
            f.savefig(os.path.join(resdir, 'log_prior.pdf'))
            plt.close(f)

        if 'predll' in x_inf[0]:
            f = plt.figure()
            plot_log_prob(x_inf, key='predll', s_true=x_true, color='r')
            plt.ylabel('Pred. Log Likelihood')
            f.savefig(os.path.join(resdir, 'pred_ll.pdf'))
            plt.close(f)

    print "Plots can be found in directory: %s" % resdir

def parse_cmd_line_args():
    """
    Parse command line parameters
    """
    from optparse import OptionParser

    parser = OptionParser()
    
    parser.add_option("-d", "--dataFile", dest="dataFile",
                      help="Data file to load")

    parser.add_option("-r", "--resultsFile", dest="resultsFile", default='.',
                      help="Results file to plot.")

    (options, args) = parser.parse_args()

    # Check if specified files exist
    if options.resultsFile is None or not os.path.exists(options.resultsFile):
        raise Exception("Invalid results file: %s" % options.resultsFile)

    return (options, args)


if __name__ == "__main__":
    from test.synth_harness import initialize_test_harness
    options, popn, data, popn_true, x_true = initialize_test_harness()
    
    # Load the results
    with open(options.x0_file, 'r') as f:
        print "Loading results from: %s" % options.x0_file
        x = cPickle.load(f)
        # If x is a list of samples, only keep the last (burned-in) fraction
        if isinstance(x, list):
            smpl_frac = 0.5
            x = x[-1*int(smpl_frac*len(x)):]

    print "Plotting results"
    plot_results(popn, 
                 x,
                 popn_true=popn_true,
                 x_true=x_true,
                 resdir=options.resultsDir,
                 do_plot_connectivity=False,
                 do_plot_stim_resp=False,
                 do_plot_imp_responses=True,
                 do_plot_firing_rates=False,
                 do_plot_ks=False,
                 do_plot_logpr=False)
