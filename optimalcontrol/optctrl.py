'''Optimal control class
'''
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import minimize

class ControlSystem:
    '''Control system class
    '''

    def __init__(self, state_vars, control_vars, params=None):
        '''Initialize the control system. Takes the following arguments:
        state_vars: number of state variables
        control_vars: number of control variables
        params: parameters of the system
        '''
        self.state_vars = state_vars
        self.control_vars = control_vars
        self.params = params

        self.dynamics = None
        self.cost = None
        self.end_cost = None

        self.dx_dynamics = None
        self.dx_cost = None

    def set_dynamics(self, dynamics, dx_dynamics=None, du_dynamics=None):
        '''Set the dynamics of the system. Additionally set
        the derivative of the dynamics if available. Takes the following arguments:
        dynamics: function of the form f(t, x, u, params)
        dx_dynamics: derivative of f with respect to x. Must take the same arguments as f
                    and return a matrix of shape (state_vars, state_vars), where
                    the columns are the derivatives of the state variables with
                    respect to the state variables, i.e. grad_x(f_i)
        du_dynamics: derivative of f with respect to u. Must take the same arguments as f
                    and return a matrix of shape (control_vars, state_vars), where
                    the columns are the derivatives of the state variables with
                    respect to the control variables, i.e. grad_u(f_i)
        '''
        self.dynamics = dynamics
        self.dx_dynamics = dx_dynamics
        self.du_dynamics = du_dynamics
    
    def set_cost(self, cost, dx_cost=None, du_cost=None):
        '''Set the cost function. Additionally set the derivative of the cost
        function if available. Takes the following arguments:
        cost: function of the form f(t, x, u, params)
        dx_cost: gradient of cost with to x. Must take the same arguments as f
                and return an array of shape (state_vars,).
        du_cost: gradient of cost with to u. Must take the same arguments as f
                and return an array of shape (control_vars,).
        '''
        self.cost = cost
        self.dx_cost = dx_cost
        self.du_cost = du_cost
    
    def set_end_cost(self, end_cost):
        '''Set the end cost function
        '''
        self.end_cost = end_cost
    
    def hamiltonian(self, t, x, u, p):
        '''Compute the Hamiltonian
        '''
        return -self.cost(t, x, u, self.params) + p @ self.dynamics(t, x, u, self.params)

    def d_hamiltonian(self, t, x, u, p):
        '''Compute the gradient of the Hamiltonian
        '''
        return -self.du_cost(t, x, u, self.params) + self.du_dynamics(t, x, u, self.params) @ p

    def find_optimal_control(self, x0, t_interval,
                                xf=None,
                                method='FBS',
                                n_steps=100,
                                rate=0.05):
        '''Find the optimal control
        '''

        if method == 'shooting':
            return self.find_optimal_control_shooting(x0, t_interval, xf)

        if method == 'FBS':
            return self.find_optimal_control_FBS(x0, t_interval, n_steps=n_steps, rate=rate)

        else:
            raise ValueError('Unknown method: %s' % method)

    def integrate_dynamics(self, x0, control, t_interval, n_steps):
        '''Integrate the dynamics of the system
        '''
        t0, tf = t_interval
        dt = (tf-t0)/(n_steps-1)
        def f(t, x):
            return self.dynamics(t, x, control(t), self.params)
        sol = solve_ivp(f, t_interval, x0, t_eval=np.linspace(t0, tf, n_steps))
        return sol

    def find_optimal_control_FBS(self, x0, t_interval,
                                n_steps=100,
                                u_initial=None,
                                tol=1e-6,
                                max_iter=1,
                                rate = 0.01):
        '''Find the optimal control using forward-backward sweep
        '''
        params = self.params
        t0, tf = t_interval
        dt = (tf-t0)/(n_steps-1)
        t_eval = np.linspace(t0, tf, n_steps)
        # TODO: Implement this with fixed-endpoint?
        # TODO: Implement this with BVP solver?
        # TODO: Implement with bounds on control

        # initialize controls
        if u_initial is None:
            # Shape is consistent with scipy.integrate.solve_ivp
            u = np.zeros((self.control_vars, n_steps))
        else:
            u = u_initial
        
        old_cost = np.inf
        
        for i in range(max_iter):
            u_old = u.copy()

            #we could be smart here and interpolate the controls
            #but for now we just use a piecewise constant control

            def ctrl_fun(t):
                t_idx = int(np.floor((t-t0)/dt))
                return u_old[:, t_idx]

            # Forward sweep:
            sol = self.integrate_dynamics(x0, ctrl_fun, t_interval, n_steps) 
            x = sol.y

            cost = np.sum([self.cost(t, x[:, t_idx], u_old[:,t_idx], params) for t_idx, t in enumerate(t_eval)])

            # Backward sweep:
            # Define adjoint equation:
            def adjoint_equation(t, y):
                t_idx = int(np.floor((t-t0)/dt))
                return self.dx_cost(t, x[:, t_idx], u_old[:,t_idx], params) - self.dx_dynamics(t, x[:, t_idx], u_old[:,t_idx], params).T @ y
            
            sol_adj = solve_ivp(adjoint_equation, t_interval[::-1], np.zeros(self.state_vars), t_eval=t_eval[::-1])
            p = sol_adj.y[:,::-1]

            # Improve control using maximum principle:

            for t_idx in range(n_steps):
                t = t0 + t_idx*dt
                u_test = self._maximize_hamiltonian(t, x[:, t_idx], p[:, t_idx], u_old[:, t_idx])
                u[:, t_idx] = rate*u_test + (1-rate)*u_old[:, t_idx]

            # Check convergence in L2 norm
            error = np.linalg.norm(u-u_old, axis=0).sum()
            print('Iteration: {}, Cost: {}, Error: {}'.format(i, cost, error))
            if error < tol*np.linalg.norm(u_old, axis=0).sum():
                print('FBS converged after {} iterations'.format(i))
                print('Error: {%0.3f}'.format(error))
                return u, p
        
        # If we get here, we didn't converge
        print('Warning: FBS did not converge')
        return u, p

    def _maximize_hamiltonian(self, t, x, p, u_guess, bounds=None):
        '''Find control that maximizes the Hamiltonian at a given time and state.
        '''
        
        def objective(u):
            return -self.hamiltonian(t, x, u, p)
        
        def d_objective(u):
            return -self.d_hamiltonian(t, x, u, p)
        
        kwargs = {}

        if bounds is not None:
            kwargs['bounds'] = bounds

        if (self.du_cost is not None) and (self.du_dynamics is not None):
            kwargs['jac'] = d_objective

        res = minimize(objective, u_guess, **kwargs)
        #if not res.success:
        #    print('\tWarning: optimization failed')
        #    print('\t'+res.message)
        return res.x