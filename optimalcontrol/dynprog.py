"""
Dynamic programming
"""
import numpy as np
from .systemtrajectory import SystemTrajectory
from ._helpers import get_closest_idx, make_grid, is_array
from ._exact_dp import calculate_valuefunction_exact, get_optimal_evolution_exact, calculate_q_factor
class DynamicProgram:
    '''A class that solves a dynamic program. Takes the following arguments:
    evolution_fun: A function that takes a step, a state, and a control and returns the next state.
    lagrangian: A function that takes a step, a state, and a control and returns the lagrangian (or cost) at that point.
    timesteps: The number of timesteps in the trajectory.
    end_cost: A function that takes a state and returns the cost at the end of the trajectory.
    '''
    # TODO: Account for things being one-dimensional. Done, I think? Needs more tests.

    default_state_grid_num = 32
    default_ctrl_grid_num = 32

    def __init__(self, evolution_fun=None, lagrangian=None, timesteps=None, end_cost=None) -> None:
        """Initializes the dynamic program. Takes the following arguments:
        evolution_fun: A function that takes a step, a state, and a control and returns the next state.
        lagrangian: A function that takes a step, a state, and a control and returns the lagrangian (or cost) at that point.
        timesteps: The number of timesteps in the trajectory.
        end_cost: A function that takes a state and returns the cost at the end of the trajectory.
        """
        self.evolution_fun = evolution_fun
        self.lagrangian = lagrangian
        self.end_cost = end_cost
        self.timesteps = timesteps

        self.state_stepsize = None
        self.ctrl_stepsize = None

        self.vf_is_initialized = False
        self.valuefunction = None
        self.grids_are_initd = False

        self.state_constraints = ()
        self.ctrl_constraints = ()

    def set_evolution_fun(self, fun):
        '''Sets the evolution function of the system.
        Fun must be of the form fun(step, state, ctrl) = next_state
        Should be vectorized at least in controls
        '''
        self.evolution_fun = fun

    def set_lagrangian(self, fun):
        '''Sets the lagrangian of the system.
        '''
        self.lagrangian = fun

    def set_state_constraints(self, constraints):
        '''Sets the state constraints of the system. Takes a dictionary or tuple of dictionaries with the following entries:
        'key': 'eq' or 'ineq'. Determines whether the constraint is equality or an inequality
                constraint of the form f(x) >= 0.
        'fun': Callable. Single function f(step, x), and returns an ndarray of any dimension (number of constraints).
        'jac': Optional, Jacobian of f. Currently does nothing but will be useful for numerical optimization in the future.
        '''
        self.state_constraints = constraints

    def set_ctrl_constraints(self, constraints):
        '''Sets the control constraints of the system. Takes a dictionary or tuple of dictionaries with the following entries:
        'key': 'eq' or 'ineq'. Determines whether the constraint is equality or an inequality
                constraint of the form f(u) >= 0.
        'fun': Callable. Single function f(step, u), and returns an ndarray of any dimension (number of constraints).
        'jac': Optional, Jacobian of f. Currently does nothing but will be useful for numerical optimization in the future.
        '''
        self.ctrl_constraints = constraints

    def satisfies_state_constraints(self, step, state):
        '''Checks whether the state satisfies the state constraints.
        Takes the following arguments:
        step: The current step of the trajectory.
        state: The current state of the system.
        '''
        # Check whether constraints is a tuple or a dictionary
        if isinstance(self.state_constraints, dict):
            state_constraints = [self.state_constraints]
        else:
            state_constraints = self.state_constraints

        for constraint in state_constraints:

            constr_eval = constraint['fun'](step, state)
            # print("Constraint evaluation:\n{}".format(constr_eval))
            if constraint['type'] == 'ineq':
                constr_bool = (constr_eval >= 0)
            else:
                try:
                    constr_bool = np.isclose(constr_eval, np.zeros_like(
                        constr_eval), atol=constraint['tol'])
                except KeyError:
                    constr_bool = np.isclose(
                        constr_eval, np.zeros_like(constr_eval))

            if not np.array([constr_bool]).all():
                return False
            # print("State {} constraint:\n{}".format(constraint['type'],constr_bool))
        return True

    def set_default_state_stepsize(self):
        '''Sets the default state stepsize. This is the stepsize used if the user does not specify a stepsize.
        '''
        bounds = self.state_bounds

        if self.num_state_vars == 1:
            h = (bounds[1]-bounds[0])/(self.default_state_grid_num - 1)
            self.set_state_stepsize(h)
        else:
            h = [(upper - lower)/(self.default_state_grid_num - 1)
                 for (lower, upper) in bounds]
            self.set_state_stepsize(h)

    def set_default_ctrl_stepsize(self):
        '''Sets the default control stepsize. This is the stepsize used if the user does not specify a stepsize.
        '''
        bounds = self.ctrl_bounds

        if self.num_ctrl_vars == 1:
            h = (bounds[1]-bounds[0])/(self.default_ctrl_grid_num - 1)
            self.set_ctrl_stepsize(h)
        else:
            h = [(upper - lower)/(self.default_ctrl_grid_num - 1)
                 for (lower, upper) in bounds]
            self.set_ctrl_stepsize(h)

    def set_state_bounds(self, bounds):
        '''Sets the state bounds of the system. Takes the following arguments:
        bounds: A tuple, or a tuple of tuples. The first element of the tuple is the lower bound, and the second element is the upper bound.
        '''
        if is_array(bounds[0]):
            self.num_state_vars = len(bounds)
        else:
            self.num_state_vars = 1

        self.state_bounds = bounds

        if self.state_stepsize is None:

            self.set_default_state_stepsize()

    def set_ctrl_bounds(self, bounds):
        '''Sets the control bounds of the system. Takes the following arguments:
        bounds: A tuple, or a tuple of tuples. The first element of the tuple is the lower bound, and the second element is the upper bound.
        '''
        if is_array(bounds[0]):
            self.num_ctrl_vars = len(bounds)
        else:
            self.num_ctrl_vars = 1

        self.ctrl_bounds = bounds

        if self.ctrl_stepsize is None:
            self.set_default_ctrl_stepsize()

    def set_state_stepsize(self, stepsize):
        '''Sets the state stepsize of the system. Takes the following arguments:
        stepsize: A float, or a tuple of floats. The stepsize of the state grid. If it is a single float, then the stepsize is the same for all state variables.
        '''
        # TODO implement shape consistency checks
        if (not is_array(stepsize)) and self.num_state_vars > 1:
            self.state_stepsize = [stepsize]*self.num_state_vars
        else:
            self.state_stepsize = stepsize

    def set_ctrl_stepsize(self, stepsize):
        '''Sets the control stepsize of the system. Takes the following arguments:
        stepsize: A float, or a tuple of floats. The stepsize of the control grid. If it is a single float, then the stepsize is the same for all control variables.
        '''
        # TODO implement shape consistency checks
        if (not is_array(stepsize)) and self.num_ctrl_vars > 1:
            self.ctrl_stepsize = [stepsize]*self.num_ctrl_vars
        else:
            self.ctrl_stepsize = stepsize

    def set_state_grid(self, alignment='left'):
        '''Sets the state grid of the system. Takes the following arguments:
        alignment: A string, or a tuple of strings. The alignment of the grid. If it is a single string, then the alignment is the same for all state variables.
        '''
        if is_array(self.state_bounds[0]):

            if not is_array(self.state_stepsize):
                self.state_stepsize = [self.state_stepsize]*self.num_state_vars

            if not is_array(alignment):
                alignment = [alignment]*len(self.state_bounds)
                # print(alignment)

            self.state_grid = [make_grid(lower, upper, step, alignment=align)
                               for ((lower, upper), step, align)
                               in zip(self.state_bounds, self.state_stepsize, alignment)]

            self.state_gridlengths = [len(grid) for grid in self.state_grid]

        else:

            self.state_grid = make_grid(
                self.state_bounds[0], self.state_bounds[1], self.state_stepsize, alignment=alignment)
            self.state_gridlengths = len(self.state_grid)

    def set_ctrl_grid(self, alignment='left'):
        '''Sets the control grid of the system. Takes the following arguments:
        alignment: A string, or a tuple of strings. The alignment of the grid. If it is a single string, then the alignment is the same for all control variables.
        '''
        if is_array(self.ctrl_bounds[0]):

            if not is_array(self.ctrl_stepsize):
                self.state_stepsize = [self.ctrl_stepsize]*self.num_state_vars

            if not is_array(alignment):
                alignment = [alignment]*len(self.ctrl_bounds)

            self.ctrl_grid = [make_grid(lower, upper, step, alignment=align)
                              for ((lower, upper), step, align)
                              in zip(self.ctrl_bounds, self.ctrl_stepsize, alignment)]

            self.ctrl_gridlengths = [len(grid) for grid in self.ctrl_grid]

        else:

            self.ctrl_grid = make_grid(
                self.ctrl_bounds[0], self.ctrl_bounds[1], self.ctrl_stepsize, alignment)
            self.ctrl_gridlengths = len(self.ctrl_grid)

    def update_griddata(self):
        '''Updates the grid data of the system. This is called automatically when the grids are set.
        '''
        if self.num_state_vars > 1:
            self.state_mesh = np.meshgrid(*self.state_grid)
            self.state_gridlengths = [len(grid) for grid in self.state_grid]
        else:
            self.state_mesh = self.state_grid
            self.state_gridlengths = len(self.state_grid)

        if self.num_ctrl_vars > 1:
            self.ctrl_mesh = np.meshgrid(*self.ctrl_grid)
            self.ctrl_gridlengths = [len(grid) for grid in self.ctrl_grid]
        else:
            self.ctrl_mesh = self.ctrl_grid
            self.ctrl_gridlengths = len(self.ctrl_grid)

        self.grids_are_initd = True

    def set_grids(self, state_align='left', ctrl_align='left'):
        '''Sets the state and control grids of the system. Takes the following arguments:
        state_align: A string, or a tuple of strings. The alignment of the state grid. If it is a single string, then the alignment is the same for all state variables.
        ctrl_align: A string, or a tuple of strings. The alignment of the control grid. If it is a single string, then the alignment is the same for all control variables.
        '''
        self.set_state_grid(state_align)
        self.set_ctrl_grid(ctrl_align)

        self.update_griddata()

    def initialize_valuefunction(self):
        '''Initializes the value function and the optimal policy.
        '''
        if self.timesteps is None:
            raise ValueError("Timesteps not set.")

        if self.vf_is_initialized:
            pass

        if not self.grids_are_initd:
            self.set_grids()

        if self.end_cost is None:
            self.end_cost = lambda *x: 0

        if self.num_state_vars == 1:
            vf_shape = (self.timesteps, self.state_gridlengths)
        else:
            vf_shape = tuple([self.timesteps] + self.state_gridlengths)

        self.valuefunction = np.full(vf_shape, np.inf)
        self.opt_policy_idx = np.zeros(
            vf_shape + (self.num_ctrl_vars,), dtype='int')
        self.next_optimal_state_idx = np.zeros(
            vf_shape + (self.num_state_vars,), dtype='int')
        # * unpacks the list into each of the arguments of end_cost, works even in 1D!
        self.valuefunction[-1] = self.end_cost(np.array(self.state_mesh))

        self.vf_is_initialized = True

    def get_state_from_idx(self, state_idx):
        '''Returns the state corresponding to the given index. Takes the following arguments:
        state_idx: The index of the state.
        Returns:
        The state corresponding to the given index.
        '''
        if self.num_state_vars == 1:
            return self.state_grid[state_idx]
        else:

            return np.array([self.state_grid[i][state_idx[i]] for i in range(self.num_state_vars)])

    def get_ctrl_from_idx(self, ctrl_idx):
        '''Returns the control corresponding to the given index. Takes the following arguments:
        ctrl_idx: The index of the control.
        Returns:
        The control corresponding to the given index.
        '''
        if self.num_ctrl_vars == 1:
            return self.ctrl_grid[ctrl_idx]
        else:
            return np.array([self.ctrl_grid[i][ctrl_idx[i]] for i in range(self.num_ctrl_vars)])

    def get_allowed_constraints_bool(self, step, state, next_states=None):
        '''Returns a boolean array of the allowed controls. Takes the following arguments:
        step: The current timestep.
        state: The current state.
        next_states: The next states. If not provided, they are calculated.
        Returns:
        A boolean array of the allowed controls.
        '''
        if self.num_state_vars == 1:

            if not is_array(state):
                x = np.array([state])
            else:
                x = state

        else:
            x = state

        state_shape = tuple([self.num_state_vars]) + \
            tuple([1]*self.num_ctrl_vars)

        if next_states is None:
            next_states = self.evolution_fun(step,
                                             # *([1]*self.num_ctrl_vars) will unpack as 1,1,1,1...,1 to reshape array
                                             x.reshape(state_shape),
                                             np.array(self.ctrl_mesh))

        # bounds check
        lower_bds, upper_bds = np.array(self.state_bounds).T

        allowed_ctrls_bool = (((next_states >= lower_bds.reshape(state_shape))
                               & (next_states <= upper_bds.reshape(state_shape)))
                              .all(axis=0))
        # check ctrl constraints
        if isinstance(self.ctrl_constraints, dict):
            ctrl_constraints = [self.ctrl_constraints]
        else:
            ctrl_constraints = self.ctrl_constraints

        for constraint in ctrl_constraints:

            constr_eval = constraint['fun'](step, np.array(self.ctrl_mesh))

            if constraint['type'] == 'ineq':
                constr_bool = (constr_eval >= 0)
            else:
                try:
                    constr_bool = np.isclose(constr_eval, np.zeros_like(
                        constr_eval), atol=constraint['tol'])
                except KeyError:
                    constr_bool = np.isclose(
                        constr_eval, np.zeros_like(constr_eval))

            if constr_bool.ndim > self.num_ctrl_vars:
                constr_bool = constr_bool.all(axis=0)
            allowed_ctrls_bool = (allowed_ctrls_bool & constr_bool)

        # check state constraints
        if isinstance(self.state_constraints, dict):
            state_constraints = [self.state_constraints]
        else:
            state_constraints = self.state_constraints

        for constraint in state_constraints:

            constr_eval = constraint['fun'](step, next_states)

            if constraint['type'] == 'ineq':
                constr_bool = (constr_eval >= 0)
            else:
                try:
                    constr_bool = np.isclose(constr_eval, np.zeros_like(
                        constr_eval), atol=constraint['tol'])
                except KeyError:
                    constr_bool = np.isclose(
                        constr_eval, np.zeros_like(constr_eval))

            constr_bool = constr_bool.all(axis=0)

            allowed_ctrls_bool = (allowed_ctrls_bool & constr_bool)

        return allowed_ctrls_bool

    def get_all_next_states(self, step, state):
        '''
        Returns a grid of all next states based on the evolution equation and the
        grid of controls. Assumes that state is a *single* state, i.e. of shape (num_state_vars,)
        This is very inefficient and should be replaced with a more efficient method.
        '''

        if self.num_state_vars == 1:

            x = np.array([state]).reshape((self.num_state_vars,))
            state_grid = self.state_grid.reshape((self.num_state_vars,
                                                  self.state_gridlengths))
            state_gridlengths = np.array([self.state_gridlengths])

        else:
            x = state
            state_grid = self.state_grid
            state_gridlengths = self.state_gridlengths

        # Boolean array of allowed controls of shape (num_ctrl_vars, m),
        # where m is the number of allowed controls which I think
        # we can't perfectly know a priori
        allowed_ctrls_bool = self.get_allowed_constraints_bool(
            step, state)

        ctrls = np.array(self.ctrl_mesh)[:, allowed_ctrls_bool]
        num_allowed_ctrls = ctrls.shape[1]
        # At this point, here are the shapes of things:
        # state: (num_state_vars,)
        # state_gridlengths: (num_state_vars, )
        # state_grid: (num_state_vars, state_gridlengths[0], ..., state_gridlengths[-1])
        #             = (num_state_vars, ) + tuple(state_gridlengths)
        # ctrl_mesh: (num_ctrl_vars, ctrl_gridlengths[0], ..., ctrl_gridlengths[-1])
        # allowed_ctrls_bool: (num_ctrl_vars, m)
        # ctrls: (num_ctrl_vars, m)

        state_shape = (self.num_state_vars, 1)
        next_states = self.evolution_fun(step,
                                         x.reshape(state_shape),
                                         ctrls)

        next_indices = np.array([(np.abs(next_states[i].reshape(num_allowed_ctrls, 1)
                                         - state_grid[i].reshape(1, state_gridlengths[i]))
                                  .argmin(axis=1))
                                 for i in range(self.num_state_vars)])

        return next_states, next_indices, ctrls, allowed_ctrls_bool

    def calculate_q_factor(self, step, state):
        '''
        Calculates the Q-factor of a state, given all admissible controls. Returns
        array in the shape of the control grid.
        '''

        # Shapes of things here:
        # next_states: (num_state_vars, ctrl_gridlenghts[0], ..., ctrl_gridlenghts[-1])
        # next_states_idx: (num_state_vars, ctrl_gridlenghts[0], ..., ctrl_gridlenghts[-1]) type int
        # allowed_ctrls_bool: (ctrl_gridlenghts[0], ..., ctrl_gridlenghts[-1]) type bool
        next_states, next_states_idx, allowed_ctrl, allowed_ctrls_bool = self.get_all_next_states(
            step, state)

        next_vf = self.valuefunction[(step + 1,) + tuple(next_states_idx)]
        q_factor = self.lagrangian(step, state, allowed_ctrl) + next_vf

        return q_factor, next_states, allowed_ctrl

    def calculate_optimal_step(self, step, state, policy='exact'):

        if policy == 'exact':
            # Assumes that we have calculated the value function for the next step
            state_idx = get_closest_idx(state, self.state_grid)
            opt_ctrl_idx = self.opt_policy_idx[step][state_idx]
            next_opt_state_idx = self.next_optimal_state_idx[step][state_idx]
            next_opt_state = self.get_state_from_idx(next_opt_state_idx)

        elif policy == 'greedy':

            next_states, next_states_idx, allowed_ctrl, allowed_ctrls_bool = self.get_all_next_states(
                step, state)
            lagrangian = self.lagrangian(step, state, allowed_ctrl)
            idx = lagrangian.argmin()
            opt_q_factor = lagrangian[idx]

        elif policy == 'rollout':
            raise NotImplementedError('Rollout policy not implemented yet.')

        else:
            raise ValueError('Unknown policy {}.'.format(policy))

        opt_ctrl = allowed_ctrl[:, idx].reshape((self.num_ctrl_vars,))
        next_opt_state = next_states[:, idx].reshape((self.num_state_vars,))
        next_opt_state_idx = get_closest_idx(next_opt_state, self.state_grid)
        opt_ctrl_idx = get_closest_idx(opt_ctrl, self.ctrl_grid)

        return opt_ctrl_idx, opt_q_factor, next_opt_state, next_opt_state_idx

    def calculate_valuefunction(self, policy='exact'):
        '''Calculate the value function recursively using the dynamic programming equation.
        policy='exact' means that we find the minimum of the Q-factor by looking over all elements
        of the array. It's very slow but guaranteed to work.        
        '''

        # TODO: Implement interpolation methods of value function and more clever optimizations
        # TODO: Check parameters
        # TODO: Check when there are several states but just one control
        self.initialize_valuefunction()
        # print("bloop")

        if policy == 'exact':
            print("Using exact policy. This might take time...")
            calculate_valuefunction_exact(self)

        else:
            raise ValueError("Policy {} not recognized.".format(policy))

    def get_optimal_evolution(self, initial_state, init_step=0, policy='exact'):
        current_state = np.array([initial_state]).reshape(
            (self.num_state_vars,))

        if policy == 'exact':
            return get_optimal_evolution_exact(self, current_state, init_step)
        elif policy == 'greedy':
            return self.get_optimal_evolution_greedy(self, current_state, init_step)
        else:
            raise ValueError("Policy {} not recognized.".format(policy))

    def get_optimal_evolution_greedy(self, initial_state, init_step=0):
        current_state = np.array([initial_state]).reshape(
            (self.num_state_vars,))

        state_trajectory = [current_state]
        ctrl_trajectory = []

        for step in range(init_step, self.timesteps - 1):

            (opt_ctrl_idx,
             opt_q_factor,
             next_opt_state,
             next_opt_state_idx) = self.calculate_optimal_step(step, current_state, policy='greedy')

            ctrl = self.get_ctrl_from_idx(opt_ctrl_idx)

            state_trajectory.append(next_opt_state)
            ctrl_trajectory.append(ctrl)

            current_state = next_opt_state

        system_traj = SystemTrajectory()
        system_traj.set_ctrl_trajectory(np.array(ctrl_trajectory))
        system_traj.set_state_trajectory(np.array(state_trajectory))
        system_traj.calculate_cost(self.lagrangian, self.end_cost)

        return system_traj