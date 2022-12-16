'''
Implementation of exact dynamic programming
'''
from ._helpers import get_closest_idx
from .systemtrajectory import SystemTrajectory
import numpy as np
import tqdm

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
    next_state = next_states[:,opt_idx]
    next_state_idx = get_closest_idx(next_state, prog.state_grid)

    return opt_idx, opt_q_factor, next_state, next_state_idx


def calculate_valuefunction_exact(prog):
    '''Calculate the value function using exact dynamic programming
    '''
    if prog.num_state_vars == 1:
        flat_range_state = prog.state_gridlengths
    else:
        flat_range_state = len(prog.state_mesh[0].flatten())

    for step in tqdm.tqdm(range(prog.timesteps - 2, -1, -1), desc='Step'):

        # TODO change to multi-index iterator
        for flat_idx in tqdm.tqdm(range(flat_range_state)):

            if prog.num_state_vars == 1:  # convert state to array

                # state = self.state_grid[flat_idx]
                state = np.array([prog.state_grid[flat_idx]])
                # print("Index: {}".format(flat_idx))
            else:
                # note that this index is reversed: the first component indexes the last variable in the mesh
                # We can access the state with [X[unraveled_ix] for X in self.state_mesh]
                unraveled_ix = np.unravel_index(
                    flat_idx, prog.state_mesh[0].shape)
                # print("Index: {}\nState: {}".format(unraveled_ix, [X[unraveled_ix] for X in self.state_mesh]))
                state = np.array([X[unraveled_ix]
                                    for X in prog.state_mesh])

            # at this point, state is an ndarray of dimension (num_state_vars,)

            if not prog.satisfies_state_constraints(step, state):
                opt_q_factor = np.inf
                opt_ctrl_idx = np.full((prog.num_ctrl_vars,), np.nan)
                next_opt_state_idx = np.full(
                    (prog.num_state_vars,), np.nan)

            # TODO fix this hacky shit
            else:
                opt_ctrl_idx, opt_q_factor, next_opt_state, next_opt_state_idx = optimize_q_factor(prog,
                    step, state)

            if prog.num_state_vars == 1:

                prog.valuefunction[step, flat_idx] = opt_q_factor
                prog.next_optimal_state_idx[step,
                                            flat_idx] = next_opt_state_idx
                if prog.num_ctrl_vars > 1:
                    # please work[1:] #first entry is x index
                    prog.opt_policy_idx[step, flat_idx] = opt_ctrl_idx
                else:
                    prog.opt_policy_idx[step, flat_idx] = opt_ctrl_idx

            else:
                prog.valuefunction[step][unraveled_ix[::-1]] = opt_q_factor
                prog.opt_policy_idx[step][unraveled_ix[::-1]
                                            ] = opt_ctrl_idx
                prog.next_optimal_state_idx[step][unraveled_ix[::-1]
                                                    ] = next_opt_state_idx

def get_optimal_evolution_exact(self, initial_state, init_step=0):
    if self.valuefunction is None:
        print("Value function not calculated yet.")
        self.calculate_valuefunction()

    initial_state = np.array([initial_state]).reshape(
        (self.num_state_vars,))
    initial_state_idx = get_closest_idx(initial_state, self.state_grid)
    state_trajectory_idx = [initial_state_idx]
    state_trajectory = [initial_state]
    ctrl_trajectory = []

    # NOTE: if num_ctrl_vars = 1 then opt_policy_idx has an annoying extra superfluous dimension
    # and I can account for it by ignoring it and reshaping at the end? probably that's the safest bet

    for count, step in enumerate(range(init_step, self.timesteps - 1)):
        state_idx = state_trajectory_idx[count]
        if self.num_state_vars == 1:
            ctrl_idx = self.opt_policy_idx[step, state_idx][0]
            next_state_idx = self.next_optimal_state_idx[step,
                                                            state_idx][0]
        else:
            # might be array even in 1d of controls
            ctrl_idx = self.opt_policy_idx[(step,)+tuple(state_idx)]
            next_state_idx = self.next_optimal_state_idx[(
                step,)+tuple(state_idx)]

        next_state = self.get_state_from_idx(next_state_idx)
        ctrl = self.get_ctrl_from_idx(ctrl_idx)  # is array even in 1d

        # print(next_state)

        state_trajectory.append(next_state)
        state_trajectory_idx.append(next_state_idx)
        ctrl_trajectory.append(ctrl)

    system_traj = SystemTrajectory()
    system_traj.set_ctrl_trajectory(np.array(ctrl_trajectory))
    system_traj.set_state_trajectory(np.array(state_trajectory))
    system_traj.calculate_cost(self.lagrangian, self.end_cost)

    return system_traj