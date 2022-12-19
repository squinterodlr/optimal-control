'''Methods for rollout (n-step lookahead) optimal control.'''
import numpy as np


def get_optimal_step_rollout(prog, step, state, horizon=None):
    '''Returns the optimal control and the next state for the given state and step.
    Using a (one-step) lookahead policy.
    Takes the following arguments:
        step: The current timestep.
        state: The current state. Must be of shape (num_state_vars,).
        horizon: The horizon to use for the rollout policy. Default None, which is taken as
                the maximum timesteps in the program.
    returns:
        next_state: ndarray of dimension (num_state_vars,)
        next_state_idx: int ndarray of dimension (num_state_vars,)
        opt_ctrl: ndarray of dimension (num_ctrl_vars,)
        opt_ctrl_idx: int ndarray of dimension (num_ctrl_vars,)
    '''
    # TODO: Implement multi-step lookahead.

    if horizon is None:
        horizon = prog.timesteps

    (next_states, next_indices,
        allowed_ctrls, allowed_ctrls_bool) = prog.get_all_next_states(step, state)
    
    next_costs = []

    for next_state, ctrl in zip(next_states.T, allowed_ctrls.T):
        evolution = prog.get_optimal_evolution(next_state,
                                               init_step=step + 1,
                                               horizon=horizon,
                                               policy='greedy')
        cost = prog.lagrangian(step, state, ctrl) + evolution.cum_cost[-1]
        next_costs.append(cost)

    next_costs = np.array(next_costs)
    opt_idx = next_costs.argmin()
    opt_ctrl = allowed_ctrls[:, opt_idx]
    opt_ctrl_idx = prog.get_closest_ctrl_idx(opt_ctrl)
    next_state = next_states[:, opt_idx]
    next_state_idx = prog.get_closest_state_idx(next_state)

    return next_state, next_state_idx, opt_ctrl, opt_ctrl_idx