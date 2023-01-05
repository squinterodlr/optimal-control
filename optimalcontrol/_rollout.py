'''Methods for rollout (n-step lookahead) optimal control.'''
import numpy as np
from tqdm import tqdm
from ._helpers import get_closest_idx

#UHHH THIS IS SUPER SLOW

def get_optimal_step_rollout(prog, step, state, horizon=None, progress_bar=True):
    '''Returns the optimal control and the next state for the given state and step.
    Using a (one-step) lookahead policy.
    Takes the following arguments:
        step: The current timestep.
        state: The current state. Must be of shape (num_state_vars,).
        horizon: The horizon to use for the rollout policy. Default None, which is taken as
                the number of steps until the end.
    returns:
        next_state: ndarray of dimension (num_state_vars,)
        next_state_idx: int ndarray of dimension (num_state_vars,)
        opt_ctrl: ndarray of dimension (num_ctrl_vars,)
        opt_ctrl_idx: int ndarray of dimension (num_ctrl_vars,)
    '''
    # TODO: Implement multi-step lookahead.

    if horizon is None:
        horizon = prog.timesteps - 2 - step

    (next_states, next_indices,
        allowed_ctrls, allowed_ctrls_bool) = prog.get_all_next_states(step, state)
    
    next_costs = []

    iterator = tqdm(zip(next_states.T, allowed_ctrls.T)) if progress_bar else zip(next_states.T, allowed_ctrls.T)
    print("Number of next states: ", len(next_states.T))
    for next_state, ctrl in iterator:
        evolution = prog.get_optimal_evolution(next_state,
                                               init_step=step + 1,
                                               end_step=step + 1 + horizon,
                                               policy='greedy',
                                               verbose=False,
                                               progress_bar=False)
        cost = prog.lagrangian(step, state, ctrl) + evolution.cum_cost[-1]
        next_costs.append(cost)

    next_costs = np.array(next_costs)
    opt_idx = next_costs.argmin()
    opt_ctrl = allowed_ctrls[:, opt_idx]
    opt_ctrl_idx = get_closest_idx(opt_ctrl, prog.ctrl_grid)
    next_state = next_states[:, opt_idx]
    next_state_idx = get_closest_idx(next_state, prog.state_grid)

    return next_state, next_state_idx, opt_ctrl, opt_ctrl_idx