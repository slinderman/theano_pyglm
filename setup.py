#!/usr/bin/env python

from distutils.core import setup

setup(name='PyGLM',
      version='0.1',
      description='Bayesian inference in GLMs for neural spike trains',
      author='Scott Linderman',
      author_email='slinderman@seas.harvard.edu',
      url='http://www.github.com/slinderman/pyglm',
      packages=['pyglm',
                'pyglm.components',
                'pyglm.inference',
                'pyglm.models',
                'pyglm.plotting',
                'pyglm.utils'],
     )