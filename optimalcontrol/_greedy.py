'''
Greedy methods for optimal control'''

import numpy as np
from ._helpers import get_closest_idx

def get_optimal_step_greedy(prog, step, state=None, state_idx=None):
    '''Get the control and next state at a given time step and state, with
    a greedy policy.
    Returns the control and next state.
    Takes the following arguments:
        step: int. The time step.
        state: ndarray of dimension (num_state_vars,). Must be specified if state_idx is not, is
            ignored if state_idx is specified.
        state_idx: int ndarray of dimension (num_state_vars,). Must be specified if state is not.
    Returns:
        next_state: ndarray of dimension (num_state_vars,)
        next_state_idx: int ndarray of dimension (num_state_vars,)
        opt_ctrl: ndarray of dimension (num_ctrl_vars,)
        opt_ctrl_idx: int ndarray of dimension (num_ctrl_vars,)
    '''
    if state_idx is not None:
        state = prog.get_state_from_idx(state_idx)
    elif state is not None:
        state = np.array([state]).reshape((prog.num_state_vars,))
        state_idx = get_closest_idx(state, prog.state_grid)
    else:
        raise ValueError("Either state or state_idx must be specified.")

    next_states, next_states_idx, allowed_ctrl, allowed_ctrls_bool = prog.get_all_next_states(
        step, state)
    lagrangian = prog.lagrangian(step, state, allowed_ctrl)
    idx = lagrangian.argmin()
    opt_ctrl = allowed_ctrl[idx]
    opt_ctrl_idx = prog.get_ctrl_idx(opt_ctrl)
    next_state = next_states[idx]
    next_state_idx = next_states_idx[idx]

    return next_state, next_state_idx, opt_ctrl, opt_ctrl_idx