pyglm
=====

Generalized linear models for neural spike train modeling, in Python! With GPU-accelerated fully-Bayesian inference, MAP inference, and network priors.

usage
-
First, install Theano version 0.6.0 from here:
http://deeplearning.net/software/theano/install.html#install

Make sure you have g++ installed, otherwise performance will be
terrible! Theano will warn you if it cannot find g++.

Then, you can quickly demo the code by running:

1.    cd pyglm
2.    mkdir data
3.    mkdir results
4.    ``python -m test.generate_synth_data -N 4 -T 60 -m standard_glm -r data``

      *(outputs something like 'Creating results directory: data\/2014_02_18-21_40')*
5.    ``python -m test.synth_map -d data/2014_02_18-21_40/data.pkl -r results``

      *(outputs something like 'Plots can be found in directory: results/2014_02_18-21_43')*
6.    open the PDFs in the results directory. Hopefully they look good!

overview
-

GLMs are one of the most popular models for neural spike trains, yet
there is no industry standard implementation. This package is an
attempt to provide a simple, extensible framework for building GLMs,
simulating spike trains, and fitting them using off-the-shelf
optimization tools.

We build upon Theano, a package that offloads much of the work of
deriving and verifying gradients and Hessians using automatic
differentiation, and then compile the gradients into optimized C or
CUDA code for fast execution. This makes model development quite easy -
we work entirely with symbolic expressions. We pay a bit of a penalty
by rederiving gradients, but in many cases this penalty is minor
relative to the benefits of rapid development.

One of the primary motivations for using Python is the potential for
easily parallelizing the code and offloading to Amazon EC2. This is
still a work in progress, but should be coming soon!

code structure
-

The code is object oriented to mimic our modeling abstractions. We
have a single GLM class which contains subclasses for the impulse
responses, the stimulus response (via the background model), the bias,
and the nonlinearity.

GLMs are bound together in a Population with shared variables. This
allows us to model the underlying network of functional correlations
and other variables that may be shared among individual neurons.

A more thorough code overview is in the works, but for now feel free
to browse!

parallel
-

Some basic instructions for running with IPython.parallel (note,
these instructions are specifically for my cluster with SGE queueing system.
For your own cluster, consult the IPython.parallel tutorial on starting
the controller and engines).

To start the controller (on a node visible from all engines):
`` ipcontroller --profile=sge --ip='*'``

To start the engines (on cluster worker nodes):
``ipcluster engines --profile=sge --n=8``

Then, run the parallel test script:
``python -m test.parallel_coord_descent -m standard_glm -d data/synth.mat -p sge``

contributing
-

Hopefully this project will offer a substrate for sharing our expertise
in GLMs. Feel free to contact me if you'd like to contribute!

