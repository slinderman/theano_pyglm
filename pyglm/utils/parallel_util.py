import os
import cPickle

from IPython.parallel.util import interactive

from pyglm.models.model_factory import *

def parallel_compute_ll(dview,
                        v,
                        N):
    """ Compute the log prob in parallel
    """
    ll_tot = 0
    # Decorate with @interactive to ensure that the function runs
    # in the __main__ namespace that contains 'popn'
    @interactive
    def _compute_glm_ll(n, vs):
        print "Computing log lkhd for GLM %d" % n
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs, n)
        lp = seval(popn.glm.ll,
                   syms,
                   nvars)
        return lp

    ll_glms = dview.map_async(_compute_glm_ll,
                              range(N),
                              [v]*N)

    ll_tot += sum(ll_glms.get())
    return ll_tot


def parallel_compute_log_p(dview,
                           master,
                           v,
                           N):
    """ Compute the log prob in parallel
    """

    # Compute the log probabaility of global variables
    # (e.g. the network) on the first node
    lp_tot = 0

    @interactive
    def _compute_network_lp(vs):
        print "Computing log prob for network"
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs,0)
        lp = seval(popn.network.log_p,
                   syms,
                   nvars)
        #lp = popn.network.log_p.eval(dict(zip(_flatten(tmpsyms),
                                     #        _flatten(tmpnvars))),
                                     #on_unused_input='ignore')
        return lp

    lp_tot += master.apply_sync(_compute_network_lp, v)

    # Decorate with @interactive to ensure that the function runs
    # in the __main__ namespace that contains 'popn'
    @interactive
    def _compute_glm_lp(n, vs):
        print "Computing lp for GLM %d" % n
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs, n)
        lp = seval(popn.glm.log_p,
                   syms,
                   nvars)
        return lp

    lp_glms = dview.map_async(_compute_glm_lp,
                              range(N),
                              [v]*N)
    # print lp_glms.get()
    # lp_glms.display_outputs()

    lp_tot += sum(lp_glms.get())
    return lp_tot


def parallel_compute_log_prior(dview,
                               master,
                               v,
                               N):
    """ Compute the log prob in parallel
    """

    # Compute the log probabaility of global variables
    # (e.g. the network) on the first node
    lp_tot = 0

    @interactive
    def _compute_network_lp(vs):
        print "Computing log prob for network"
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs,0)
        lp = seval(popn.network.log_p,
                   syms,
                   nvars)
        #lp = popn.network.log_p.eval(dict(zip(_flatten(tmpsyms),
                                     #        _flatten(tmpnvars))),
                                     #on_unused_input='ignore')
        return lp

    lp_tot += master.apply_sync(_compute_network_lp, v)

    # Decorate with @interactive to ensure that the function runs
    # in the __main__ namespace that contains 'popn'
    @interactive
    def _compute_glm_lp(n, vs):
        print "Computing log prior for GLM %d" % n
        syms = popn.get_variables()
        nvars = popn.extract_vars(vs, n)
        lp = seval(popn.glm.log_prior,
                   syms,
                   nvars)
        return lp

    lp_glms = dview.map_async(_compute_glm_lp,
                              range(N),
                              [v]*N)
    # print lp_glms.get()
    # lp_glms.display_outputs()

    lp_tot += sum(lp_glms.get())
    return lp_tot


def initialize_imports(dview):
    """ Import required model code on the clients.
        Initialize global variable for each client's local population obj
    """
    dview.execute('from pyglm.population import Population')
    dview.execute('from pyglm.models.model_factory import make_model')
    dview.execute('from pyglm.utils.theano_func_wrapper import seval')
    dview.execute('import cPickle')

    # Make sure that each node uses a different Theano compile directory
    @interactive
    def _set_theano_base_compiledir(n):
        theano.config.base_compiledir = \
            os.path.join(theano.config.base_compiledir, 'engine%d' % n)
    dview.map(_set_theano_base_compiledir,
              range(len(dview)))

def add_data_on_engines(dview, d,
                        use_shared_filesystem=False):
    """ Send the data to each engine
    """
    # Initialize the GLM with the data
    if not use_shared_filesystem:
        # This seems to be the bottleneck for large datasets!
        # The data is sent to each worker sequentially (not broadcast)
        dview['data'] = d

    else:
        # Pickle the data and save it to the shared file system
        with open('.temp_data.pkl', 'w') as f:
            cPickle.dump(d, f, protocol=-1)

        # Initialize the data globa
        dview['data'] = 'tmp'
        @interactive
        def _load_data():
            global data
            with open('.temp_data.pkl', 'r') as f:
                data = cPickle.load(f)

        dview.apply_sync(_load_data)

        # Delete the temp data
        os.remove('.temp_data.pkl')

    # Condition the population on the data
    dview.execute("popn.add_data(data)", block=True)

def set_hyperparameters_on_engines(dview, m):
    """ Send the hyperparameters to each engine
    """
    dview['model'] = m
    dview.execute("popn.set_hyperparameters(model)", block=True)


def create_population_on_engines(dview,
                                 data_sequences,
                                 model_type
                                 ):
    """ Initialize a model with N neurons. Use the data if specified on the
        command line, otherwise sample new data from the model.
        Return a population object, the data, and a set of true parameters
        which is expected for synthetic tests
    """
    # Make sure data sequences is a list
    if not isinstance(data_sequences, list):
        data_sequences = [data_sequences]

    # Initialize a model with N neurons
    N = data_sequences[0]['N']
    dt = data_sequences[0]['dt']
    if isinstance(model_type, dict):
        m = model_type
    elif isinstance(model_type, str):
        m = make_model(model_type, N=N, dt=dt)
    else:
        raise Exception("Create_population_on_engines requires model to be either str type or dict")

    # We only want to do this for synthetic tests.
    # stabilize_sparsity(m)
    dview['model'] = m
    

    # Create a population object on each engine
    #@interactive
    #def _create_population():
    #    global popn
    #    popn = Population(model)
    #dview.apply(_create_population, block=True)

    dview.execute('popn = Population(model)', block=True)

    # Initialize the GLM with the data
    #dview['data'] = data
    #dview.execute("popn.set_data(data)", block=True)
    for data in data_sequences:
        add_data_on_engines(dview, data)
