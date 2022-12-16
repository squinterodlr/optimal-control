import numpy as np

class SystemTrajectory:
    '''Holds a trajectory of state and control variables. 
    '''

    def __init__(self):
        '''Initializes the trajectory.
        '''
        self.state_traj = []
        self.ctrl_traj = []
        self.length = 0
        self.t_init = 0

    def set_state_trajectory(self, traj):
        self.state_traj = traj[:]
        self.set_length(len(traj))

    def set_ctrl_trajectory(self, traj):
        self.ctrl_traj = traj[:]
        self.set_length(len(traj))

    def set_length(self, length):
        '''Sets the length of the trajectory.
        '''
        self.length = length

    def __len__(self):
        return self.length

    def calculate_cost(self, lagrangian, end_cost):
        '''Calculates the cost of the trajectory.
        '''

        self.cost_traj = [lagrangian(step, state, ctrl) for step, (state, ctrl)
                          in enumerate(zip(self.state_traj, self.ctrl_traj))]
        self.cost_traj[-1] += end_cost(self.state_traj[-1])

        self.cost_traj = np.array(self.cost_traj)
        self.cum_cost = (self.cost_traj).cumsum()


