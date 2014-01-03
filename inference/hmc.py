"""
Implementation of Hybrid Monte Carlo (HMC) sampling algorithm following Neal (2010).
Use the log probability and the gradient of the log prob to navigate the distribution.
"""
import numpy as np
import logging
log = logging.getLogger("global_log")

def hmc(U, 
        grad_U, 
        step_sz, 
        n_steps, 
        q_curr, 
        adaptive_step_sz=False,
        tgt_accept_rate=0.9,
        avg_accept_time_const=0.95,
        avg_accept_rate=0.9,
        min_step_sz=0.001,
        max_step_sz=1.0):
    """
    U       - function handle to compute log probability we are sampling
    grad_U  - function handle to compute the gradient of the density with respect 
              to relevant params
    step_sz - step size
    n_steps       - number of steps to take
    q_curr  - current state
    
    """
    # Start at current state
    q = np.copy(q_curr)
    # Moment is simplest for a normal rv
    p = 0.25 * np.random.randn(np.size(q))
    p_curr = p
    
    # Evaluate potential and kinetic energies at start of trajectory
    U_curr = U(q_curr)
    K_curr = np.sum(p_curr**2)/2
    
    # Make a half step in the momentum variable
    p -= step_sz*grad_U(q)/2
    
    # Alternate L full steps for position and momentum
    for i in np.arange(n_steps):
        q += step_sz*p
        
        # Full step for momentum except for last iteration
        if i < n_steps-1:
            p -= step_sz*grad_U(q)
        else:
            p -= step_sz*grad_U(q)/2
    
    # Negate the momentum at the end of the trajectory to make proposal symmetric?
    p = -p
    
    # Evaluate potential and kinetic energies at end of trajectory
    U_prop = U(q)
    K_prop = np.sum(p**2)/2
    
    # Accept or reject new state with probability proportional to change in energy.
    # Ideally this will be nearly 0, but forward Euler integration introduced errors.
    # Exponentiate a value near zero and get nearly 100% chance of acceptance.
    accept = np.log(np.random.rand()) < U_curr-U_prop + K_curr-K_prop
    if accept:
        q_next = q
    else:
        q_next = q_curr

    q_next = np.reshape(q_next, np.shape(q))
        
    # Do adaptive step size updates if requested
    if adaptive_step_sz:
        new_accept_rate = avg_accept_time_const * avg_accept_rate + \
                          (1.0-avg_accept_time_const) * accept
        if avg_accept_rate > tgt_accept_rate:
            new_step_sz = step_sz * 1.02
        else:
            new_step_sz = step_sz * 0.98

        new_step_sz = np.clip(new_step_sz, min_step_sz, max_step_sz)

        return (q_next, new_step_sz, new_accept_rate)
    else:
        return q_next
