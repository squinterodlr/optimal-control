import numpy as np
from ._helpers import is_array

def satisfies_state_constraints(prog, step, state):
    '''Checks whether the state satisfies the state constraints.
    Takes the following arguments:
    step: The current step of the trajectory.
    state: The current state of the system.
    '''
    # Check whether constraints is a tuple or a dictionary
    if isinstance(prog.state_constraints, dict):
        state_constraints = [prog.state_constraints]
    else:
        state_constraints = prog.state_constraints

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

def get_allowed_constraints_bool(prog, step, state, next_states=None):
    '''Returns a boolean array of the allowed controls. Takes the following arguments:
    step: The current timestep.
    state: The current state.
    next_states: The next states. If not provided, they are calculated.
    Returns:
    A boolean array of the allowed controls.
    '''
    if prog.num_state_vars == 1:

        if not is_array(state):
            x = np.array([state])
        else:
            x = state

    else:
        x = state

    state_shape = tuple([prog.num_state_vars]) + \
        tuple([1]*prog.num_ctrl_vars)

    if next_states is None:
        next_states = prog.evolution_fun(step,
                                            # *([1]*self.num_ctrl_vars) will unpack as 1,1,1,1...,1 to reshape array
                                            x.reshape(state_shape),
                                            np.array(prog.ctrl_mesh))

    # bounds check
    lower_bds, upper_bds = np.array(prog.state_bounds).T

    allowed_ctrls_bool = (((next_states >= lower_bds.reshape(state_shape))
                            & (next_states <= upper_bds.reshape(state_shape)))
                            .all(axis=0))
    # check ctrl constraints
    if isinstance(prog.ctrl_constraints, dict):
        ctrl_constraints = [prog.ctrl_constraints]
    else:
        ctrl_constraints = prog.ctrl_constraints

    for constraint in ctrl_constraints:

        constr_eval = constraint['fun'](step, np.array(prog.ctrl_mesh))

        if constraint['type'] == 'ineq':
            constr_bool = (constr_eval >= 0)
        else:
            try:
                constr_bool = np.isclose(constr_eval, np.zeros_like(
                    constr_eval), atol=constraint['tol'])
            except KeyError:
                constr_bool = np.isclose(
                    constr_eval, np.zeros_like(constr_eval))

        if constr_bool.ndim > prog.num_ctrl_vars:
            constr_bool = constr_bool.all(axis=0)
        allowed_ctrls_bool = (allowed_ctrls_bool & constr_bool)

    # check state constraints
    if isinstance(prog.state_constraints, dict):
        state_constraints = [prog.state_constraints]
    else:
        state_constraints = prog.state_constraints

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