# Run as script using 'python -m test.synth'
import os
import cPickle
import time

import numpy as np

from generate_synth_data import gen_synth_data
from models.model_factory import make_model, convert_model
from population import Population
from pyglm.inference.parallel_gibbs import parallel_gibbs_sample
from parallel_harness import initialize_parallel_test_harness
from pyglm.plotting import plot_results


def run_synth_test():
    """ Run a test with synthetic data and MCMC inference
    """
    options, popn, data, client, popn_true, x_true = initialize_parallel_test_harness()

    raise Exception("Make sur ethe sparsity is set properly!")

    # If x0 specified, load x0 from file
    x0 = None
    if options.x0_file is not None:
        with open(options.x0_file, 'r') as f:
            print "Initializing with state from: %s" % options.x0_file
            prev_x0 = cPickle.load(f)
            if isinstance(prev_x0, list):

                x0 = prev_x0[-1]
            else:
                mle_x0 = prev_x0
                # HACK: We're assuming x0 came from a standard GLM
                mle_model = make_model('standard_glm', N=data['N'])
                mle_popn = Population(mle_model)
                mle_popn.set_data(data)

                x0 = popn.sample(None)
                x0 = convert_model(mle_popn, mle_model, mle_x0, popn, popn.model, x0)

    # !!!!DEBUG!!!!!
    # Initialize with true variables
    # import copy
    # x0 = copy.deepcopy(x_true)

    use_existing = False
    
    if use_existing and  \
       os.path.exists(os.path.join(options.resultsDir, 'results.pkl')):

        print "Found existing results"
        with open(os.path.join(options.resultsDir, 'results.pkl')) as f:
            x_smpls = cPickle.load(f)
            N_samples = len(x_smpls)
    else:
        N_samples = 1000

        # Define a callback to evaluate log likelihoods and predictive log likelihoods
        print "Creating synthetic test data"
        T_test = 15
        popn_test = Population(popn.model)
        test_data = gen_synth_data(data['N'], T_test, popn_true, x_true)
        popn_test.set_data(test_data)

        # Compute pred ll under true model
        popn_true.set_data(test_data)
        x_true['predll'] = popn_true.compute_ll(x_true)
        popn_true.set_data(data)

        # Compute the predictive log likelihood under a homogeneous PP model wiht MLE
        # homog_pred_lls[j] = compute_homog_pp(train_data, test_data)

        pred_lls = np.zeros(N_samples)
        smpl = [0]
        def pred_ll_cbk(x):
            pred_ll = popn_test.compute_ll(x)
            pred_lls[smpl[0]] = pred_ll
            x['predll'] = pred_ll
            smpl[0] += 1
            print "Pred LL: %.2f" % pred_ll
        pred_ll_cbk = None

        # Perform inference
        print "Performing parallel inference"
        start_time = time.time()
        x_smpls = parallel_gibbs_sample(client, data,
                                        x0=x0, N_samples=N_samples,
                                        save_interval=50, results_dir=options.resultsDir,
                                        callback=pred_ll_cbk)
        stop_time = time.time()

        # Save results
        print "Saving results to %s" % os.path.join(options.resultsDir, 'results.pkl')
        with open(os.path.join(options.resultsDir, 'results.pkl'),'w') as f:
            cPickle.dump(x_smpls, f, protocol=-1)

        # Save runtime
        with open(os.path.join(options.resultsDir, 'runtime.pkl'),'w') as f:
            cPickle.dump(stop_time-start_time, f, protocol=-1)


    # Plot average of last 20% of samples
    print "Plotting results"
    smpl_frac = 1.0

    # Only plot the impulse response matrix for small N
    do_plot = data['N'] < 20
    do_plot_imp_responses = data['N'] < 30

    if do_plot:
        plot_results(popn,
                    x_smpls[-1*int(smpl_frac*len(x_smpls)):],
                    popn_true,
                    x_true,
                    do_plot_imp_responses=do_plot_imp_responses,
                    resdir=options.resultsDir)

if __name__ == "__main__":
    run_synth_test()
