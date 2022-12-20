'''
Implementation of exact dynamic programming
'''
from ._helpers import get_closest_idx
from .systemtrajectory import SystemTrajectory
from ._constraints import satisfies_state_constraints
import numpy as np
import tqdm
import warnings

def calculate_q_factor(prog, step, state):
        '''
        Calculates the Q-factor of a state, given all admissible controls. Returns
        array in the shape of the control grid.
        '''

        # Shapes of things here:
        # next_states: (num_state_vars, ctrl_gridlenghts[0], ..., ctrl_gridlenghts[-1])
        # next_states_idx: (num_state_vars, ctrl_gridlenghts[0], ..., ctrl_gridlenghts[-1]) type int
        # allowed_ctrls_bool: (ctrl_gridlenghts[0], ..., ctrl_gridlenghts[-1]) type bool
        next_states, next_states_idx, allowed_ctrl, allowed_ctrls_bool = prog.get_all_next_states(
            step, state)

        next_vf = prog.valuefunction[(step + 1,) + tuple(next_states_idx)]
        q_factor = prog.lagrangian(step, state, allowed_ctrl) + next_vf

        return q_factor, next_states, allowed_ctrl

def optimize_q_factor(prog, step, state):

    q_factor, next_states, allowed_ctrl = calculate_q_factor(prog, step, state)

    opt_idx = q_factor.argmin()

    opt_q_factor = q_factor[opt_idx]
    opt_ctrl = allowed_ctrl[:,opt_idx]
    opt_ctrl_idx = get_closest_idx(opt_ctrl, prog.ctrl_grid)
    next_state = next_states[:,opt_idx]
    next_state_idx = get_closest_idx(next_state, prog.state_grid)

    return opt_ctrl_idx, opt_q_factor, next_state, next_state_idx


def calculate_valuefunction_exact(prog):
    '''Calculate the value function using exact dynamic programming.
    Updates the value function in prog.valuefunction, writes
    the indices of the optimal controls in prog.opt_ctrl_idx, 
    and the indices of the optimal next states in prog.next_opt_state_idx.
    '''
    if prog.num_state_vars == 1:
        flat_range_state = prog.state_gridlengths
    else:
        flat_range_state = len(prog.state_mesh[0].flatten())

    for step in tqdm.tqdm(range(prog.timesteps - 2, -1, -1), desc='Step'):

        # TODO change to multi-index iterator
        for flat_idx in tqdm.tqdm(range(flat_range_state)):

            if prog.num_state_vars == 1:  # convert state to array

                # state = prog.state_grid[flat_idx]
                state = np.array([prog.state_grid[flat_idx]])
                # print("Index: {}".format(flat_idx))
            else:
                # note that this index is reversed: the first component indexes the last variable in the mesh
                # We can access the state with [X[unraveled_ix] for X in prog.state_mesh]
                unraveled_ix = np.unravel_index(
                    flat_idx, prog.state_mesh[0].shape)
                # print("Index: {}\nState: {}".format(unraveled_ix, [X[unraveled_ix] for X in prog.state_mesh]))
                state = np.array([X[unraveled_ix]
                                    for X in prog.state_mesh])

            # at this point, state is an ndarray of dimension (num_state_vars,)

            if not satisfies_state_constraints(prog, step, state):
                opt_q_factor = np.inf
                opt_ctrl_idx = np.full((prog.num_ctrl_vars,), np.nan)
                next_opt_state_idx = np.full(
                    (prog.num_state_vars,), np.nan)

            
            else:
                opt_ctrl_idx, opt_q_factor, next_opt_state, next_opt_state_idx = optimize_q_factor(prog,
                    step, state)

            if prog.num_state_vars == 1:

                prog.valuefunction[step, flat_idx] = opt_q_factor
                prog.next_optimal_state_idx[step,
                                            flat_idx] = next_opt_state_idx
                prog.opt_policy_idx[step, flat_idx] = opt_ctrl_idx

            else:
                prog.valuefunction[step][unraveled_ix[::-1]] = opt_q_factor
                prog.opt_policy_idx[step][unraveled_ix[::-1]
                                            ] = opt_ctrl_idx
                prog.next_optimal_state_idx[step][unraveled_ix[::-1]
                                                    ] = next_opt_state_idx

def get_optimal_step_exact(prog, step, state=None, state_idx=None):
    '''Get the optimal control and next state at a given time step and state,
    assuming the value function has been calculated using exact dynamic programming.
    Returns the optimal control and the optimal next state.
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

    if prog.valuefunction is None:
        warnings.warn("Value function not calculated. Calculating now.", RuntimeWarning)
        calculate_valuefunction_exact(prog)


    opt_ctrl_idx = prog.opt_policy_idx[step][tuple(state_idx)]
    next_state_idx = prog.next_optimal_state_idx[step][tuple(state_idx)]
    next_state = prog.get_state_from_idx(next_state_idx)
    opt_ctrl = prog.get_ctrl_from_idx(opt_ctrl_idx)

    return next_state, next_state_idx, opt_ctrl, opt_ctrl_idx

def get_optimal_evolution_exact(prog, initial_state, init_step=0):
    '''Get the optimal evolution of the system starting from initial_state at time init_step.
    Returns the state trajectory and the control trajectory.
    Takes the following arguments:
        initial_state: ndarray of dimension (num_state_vars,)
        init_step: int, default 0. The initial time step.
    Returns:
        system_traj: Object of class SystemTrajectory containing the state and control trajectories.
    '''
    if prog.valuefunction is None:
        warnings.warn("Value function not calculated. Calculating now.", RuntimeWarning)
        calculate_valuefunction_exact(prog)

    initial_state = np.array([initial_state]).reshape(
        (prog.num_state_vars,))
    initial_state_idx = get_closest_idx(initial_state, prog.state_grid)
    state_trajectory_idx = [initial_state_idx]
    state_trajectory = [initial_state]
    ctrl_trajectory = []

    # NOTE: if num_ctrl_vars = 1 then opt_policy_idx has an annoying extra superfluous dimension
    # and I can account for it by ignoring it and reshaping at the end? probably that's the safest bet

    for count, step in enumerate(range(init_step, prog.timesteps - 1)):
        state_idx = state_trajectory_idx[count]

        next_state, next_state_idx, opt_ctrl, opt_ctrl_idx = get_optimal_step_exact(prog, step=step,
            state_idx=state_idx)

        state_trajectory.append(next_state)
        state_trajectory_idx.append(next_state_idx)
        ctrl_trajectory.append(opt_ctrl)

    # Flatten trajectories in the case of 1D state or control variables
    if prog.num_ctrl_vars == 1:
        ctrl_trajectory = np.array(ctrl_trajectory).reshape(
            (prog.timesteps - 1,))
    else:
        ctrl_trajectory = np.array(ctrl_trajectory).reshape(
            (prog.timesteps - 1, prog.num_ctrl_vars))
    
    if prog.num_state_vars == 1:
        state_trajectory = np.array(state_trajectory).reshape(
            (prog.timesteps,))
    else:
        state_trajectory = np.array(state_trajectory).reshape(
            (prog.timesteps, prog.num_state_vars))

    system_traj = SystemTrajectory()
    system_traj.set_ctrl_trajectory(np.array(ctrl_trajectory))
    system_traj.set_state_trajectory(np.array(state_trajectory))
    system_traj.calculate_cost(prog.lagrangian, prog.end_cost)

    return system_traj