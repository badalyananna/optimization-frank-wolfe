from typing import Iterable, List, Optional, Tuple, Union, Dict
import time

import numpy as np

class FW_SSC:
    """
    Implements the Frank Wolfe algorithms and its variants for the maximum click on hypergraphs problem
    """
    def __init__(
        self,
        variant: Optional[str] = "clasic",
        stepsize_strategy: Optional[str] = "armijo",
        alpha: Optional[float] = 0.1, # armijo search param
        delta: Optional[float] = 0.7, # armijo search param
        tau: Optional[float] = 1.0,
        x: Optional[np.ndarray] = None,
        tolerance: Optional[float] = 1e-4,
        max_iter: Optional[int] = 10000, 
    ):
        """ #TODO write a description

        Parameters
        ----------
        x: value of x at the current iteration
        tau: parameter of the objective function
        alpha: armijo search parameter (gamma in lecture notes)
        delta: decrease rate of the armijo line search
        """
        super().__init__()
        self.variant = variant
        self.tau = tau
        self.x = x
        # parameters related to line search
        self.stepsize_strategy = stepsize_strategy
        self.alpha = alpha
        self.delta = delta
        self.gamma_max = 1.0
        self.of_value = None

        self.tolerance = tolerance
        self.max_iter = max_iter
        self.trained: bool = False
        self.tolerance_reached: bool = False
        self.training_iter: Optional[int] = None
        self.history: Optional[Dict] = None
    
    def optimize(
        self,
        edges: List[List[int]],
        N: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        # initialize x if the value has not been provided
        if N is None:
            N = max(map(max, edges)) + 1
        if self.x is None:
            self.x = self._init_x(N, seed)
        self.of_value = self.calc_obj_function(edges)
        K = len(edges[0])
        self.tau *= 1 / (K * (K - 1))
        self.L = 0.1

        max_iter, tolerance, variant = self.max_iter, self.tolerance, self.variant
        
        self.histiry = {
            "iteration": [0],
            "cpu_time": [0],
            "of_value": [self.of_value],
            "duality_gap": []
        }
        start_time = time.time()
        for iter in range(max_iter):
            print(f'iteration {iter}')
            #self.histiry["duality_gap"].append(-duality_gap)
            #if duality_gap >= -tolerance:
            #    self.trained = True
            #    self.tolerance_reached = True
            #    self.training_iter = iter
            #    self.history = None
            #    break
            self.x, converged = self._SSC(edges)
            if converged:
                self.trained = True
                self.tolerance_reached = True
                self.training_iter = iter
                break

            iter_time = time.time()
            self.histiry["iteration"].append(iter)
            self.histiry["of_value"].append(self.of_value)
            self.histiry["cpu_time"].append(iter_time - start_time)

    def _init_x(self, N: int, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        x = rng.random(N)
        x = x / x.sum()
        return x
    
    def _SSC(self, edges):
        y = self.x
        grad = self.calc_gradient(edges)
        s = LMO(grad)
        L = 0.1
        print(f'start L {L}')
        print(f'grad {grad}')
        print(f's {s}')
        iter = 0
        while True:
            iter += 1
            print(f"SSC iter {iter}")
            print(f'y {y}')
            # Phase 1
            d_fw = s - y
            duality_gap = grad @ d_fw
            if duality_gap >= -self.tolerance:
                return y, True
            if self.variant == 'pairwise':
                d, gamma_max = self._pairwise_update(grad, s, y)
            elif self.variant == 'blended_pairwise':
                d, gamma_max = self._blended_pairwise_update(grad, s, y, duality_gap)
            else:
                raise NotImplementedError("Unknown FW variant")

            print(f'd {d}')
            # Phase 2
            beta, L = self.backtracking_step_size(edges, y, d, grad, L, duality_gap, gamma_max)
            if beta == 0:
                self.L = L
                return y, False
            print(f'inner L, beta, gamma_max {L, beta, gamma_max}')
            y += beta * d
            #print(beta, gamma_max)
            if beta < gamma_max:
                #print('beta < gamma')
                self.L = L
                return y, False
            assert iter < 1000, "The while loop in SSC exceeded 1000 iterations."

    def backtracking_step_size(self, edges, y, d, grad, L_prev, duality_gap, gamma_max, tau=1.5, nu = 1):
        def Q_t(of_value, gamma, duality_gap, M, d):
            return of_value - gamma * duality_gap + ((gamma ** 2) * M  * (np.linalg.norm(d) ** 2) )/ 2
        def Q_t_withoutM(of_value, gamma, gap):
            return of_value - gamma * gap / 2
        
        grad_d = - grad @ d
        # check if d is global
        initial_of_value = self.calc_obj_function(edges, y)
        gamma_min = 1e-10
        if self.calc_obj_function(edges, y + gamma_min * d) > Q_t_withoutM(initial_of_value, gamma_min, grad_d):
            return 0, L_prev

        print(f'grad d {grad_d}')
        M = nu * L_prev
        d_norm_squared = np.linalg.norm(d) ** 2
        gamma = grad_d / (M * d_norm_squared)
        print(f"backtracking gamma estimate {gamma}, gamma_max {gamma_max}")
        gamma = min(gamma_max, gamma)
        
        while self.calc_obj_function(edges, y + gamma * d) > Q_t(initial_of_value, gamma, grad_d, M, d):
            print(f"Backtracking gamma {gamma}, o.f. {self.calc_obj_function(edges, y + gamma * d)}, Q_t {Q_t(initial_of_value, gamma, grad_d, M, d)}")
            print(f"Initial of value {initial_of_value}, Q_t withour M {Q_t_withoutM(initial_of_value, gamma, grad_d)}")
            print(f'M {M}')
            M *= tau
            gamma_new = grad_d / (M * d_norm_squared)
            gamma = min(gamma_new, gamma_max)
        return gamma, M


    
    def _global_step(self, edges: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, float]:
        x = self.x
        grad = self.calc_gradient(edges)
        s = LMO(grad)
        d_fw = s - x
        duality_gap = grad @ d_fw
        return s, d_fw, grad, duality_gap   
        
    def _pairwise_update(
        self,
        grad: np.ndarray,
        s: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        active_set = np.where(y > 0)[0].tolist()
        v, v_index = LMO(grad, active_set=active_set, task='maximize')
        d_pw = s - v
        return d_pw, y[v_index]

    def _blended_pairwise_update(
        self,
        grad: np.ndarray,
        s: np.ndarray,
        y: np.ndarray,
        duality_gap: float,
    ) -> Tuple[np.ndarray, float]:
        active_set = np.where(y > 0)[0].tolist()
        a, a_index = LMO(grad, active_set=active_set, task='maximize') # away step
        w, w_index = LMO(grad, active_set=active_set, task='minimize') # local FW step
        local_gap = grad @ (a - w)
        if local_gap >= - duality_gap:
            # optimize localy over the active set
            d = w - a
            return d, y[a_index]
        else:
            print('global step')
            d = s - y
            return d, 1
    
    def calc_obj_function(self, edges: List[List[int]], x: Optional[np.ndarray] = None) -> np.ndarray:
        tau = self.tau
        if x is None:
            x = self.x
        k = len(edges[0])
        LG = x[edges].prod(axis=1).sum()
        return LG + tau * (x ** k).sum()

    def calc_gradient(self, edges: List[List[int]]) -> np.ndarray:
        x, tau = self.x, self.tau
        N = len(x)

        hg_matrix = x[edges]
        hg_indices = np.array(edges)
        e, k = hg_matrix.shape
        remaining_products = np.zeros(hg_matrix.shape)
        for i in range(k):
            new_matrix = hg_matrix.copy()
            new_matrix[:,i] = np.ones(e)
            remaining_products[:,i] = new_matrix.prod(axis=1)
        grad = np.zeros(N)
        for i in range(N):
            grad[i] = remaining_products[hg_indices == i].sum()
        return grad + tau * k * (x ** (k - 1))
    
    def armijo(self,
            d: np.ndarray, 
            duality_gap: float,
            edges: List[List[int]]) -> Tuple[np.ndarray, float, float]:
        """
        Implements Armijo line search

        Paremeters
        ----------
        d: the descent direction of the algorithm
        duality_gap: the duality gap at the current iteration of the algorithm
        edges: list of lists containing the id of the nodes belinging to edges

        Returns
        -------
        x_new: the new value x
        of_new: the new value of the o.f. at x_new
        """
        x_old, alpha, delta, of_old, gamma_max = self.x, self.alpha, self.delta, self.of_value, self.gamma_max

        gamma = gamma_max
        x_new = x_old + gamma * d
        of_new = self.calc_obj_function(edges, x_new)
        m = 0
        while of_new > of_old + alpha * gamma * duality_gap:
            m += 1
            gamma = delta * gamma
            x_new = x_old + gamma * d
            of_new = self.calc_obj_function(edges, x_new)
            assert m < 10000, "Armijo made 10 000 iteration, something must be wrong"
        return x_new, of_new, gamma

class FW:
    """
    Implements the Frank Wolfe algorithms and its variants for the maximum click on hypergraphs problem
    """
    def __init__(
        self,
        variant: Optional[str] = "clasic",
        stepsize_strategy: Optional[str] = "armijo",
        alpha: Optional[float] = 0.1, # armijo search param
        delta: Optional[float] = 0.7, # armijo search param
        tau: Optional[float] = 1.0,
        x: Optional[np.ndarray] = None,
        tolerance: Optional[float] = 1e-4,
        max_iter: Optional[int] = 10000, 
    ):
        """ #TODO write a description

        Parameters
        ----------
        x: value of x at the current iteration
        tau: parameter of the objective function
        alpha: armijo search parameter (gamma in lecture notes)
        delta: decrease rate of the armijo line search
        """
        super().__init__()
        self.variant = variant
        self.tau = tau
        self.x = x
        # parameters related to line search
        self.stepsize_strategy = stepsize_strategy
        self.alpha = alpha
        self.delta = delta
        self.gamma_max = 1.0
        self.of_value = None

        self.tolerance = tolerance
        self.max_iter = max_iter
        self.trained: bool = False
        self.tolerance_reached: bool = False
        self.training_iter: Optional[int] = None
        self.history: Optional[Dict] = None
    
    def optimize(
        self,
        edges: List[List[int]],
        N: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        # initialize x if the value has not been provided
        if N is None:
            N = max(map(max, edges)) + 1
        if self.x is None:
            self.x = self._init_x(N, seed)
        self.of_value = self.calc_obj_function(edges)
        K = len(edges[0])
        self.tau *= 1 / (K * (K - 1))
        self.L = 0.1

        max_iter, tolerance, variant = self.max_iter, self.tolerance, self.variant
        
        self.history = {
            "iteration": [0],
            "cpu_time": [0],
            "of_value": [self.of_value],
            "duality_gap": []
        }
        start_time = time.time()
        for iter in range(max_iter):
            self.training_iter = iter
            s, d_fw, grad, duality_gap = self._global_step(edges)
            self.history["duality_gap"].append(-duality_gap)
            if duality_gap >= -tolerance:
                self.trained = True
                self.tolerance_reached = True
                break
            
            self.x, self.of_value = self._FW_step(edges, grad, d_fw, s, duality_gap)

            iter_time = time.time()
            self.history["iteration"].append(iter)
            self.history["of_value"].append(self.of_value)
            self.history["cpu_time"].append(iter_time - start_time)

    def _init_x(self, N: int, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        x = rng.random(N)
        x = x / x.sum()
        return x
    
    def _global_step(self, edges: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, float]:
        x = self.x
        grad = self.calc_gradient(edges)
        s = LMO(grad)
        d_fw = s - x
        duality_gap = grad @ d_fw
        return s, d_fw, grad, duality_gap   

    def _FW_step(
        self, 
        edges: List[List[int]],
        grad: np.ndarray,
        d_fw: np.ndarray,
        s: np.ndarray,
        duality_gap: float,
    ) -> Tuple[np.ndarray, float]:
        if self.variant == 'FW':
            d, gap, gamma_max = (d_fw, duality_gap, 1)
        elif self.variant == 'AFW':
            d, gap, gamma_max = self._away_step(grad, d_fw, duality_gap) 
        elif self.variant == 'PFW':
            d, gap, gamma_max = self._pairwise_step(grad, s)
        elif self.variant == 'BPFW':
            d, gap, gamma_max = self._blended_pairwise_step(grad, s, duality_gap)
        else:
            raise NotImplementedError("Currently ony classic, pairwise and blended_pairwise FW variants are available")
        return self._choose_stepsize(edges, d, gap, gamma_max)
        
    def _away_step(
        self,
        grad: np.ndarray,
        d_fw: np.ndarray,
        duality_gap: float,
    ) -> Tuple[np.ndarray, float]:
        x = self.x
        active_set = np.where(x > 0)[0].tolist()
        v, v_index = LMO(grad, active_set=active_set, task='maximize')
        d_a = x - v
        away_gap = grad @ d_a
        if - duality_gap >= - away_gap:
            d = d_fw
            gamma_max = 1
            gap = duality_gap
        else:
            d = d_a
            alpha_v = x[v_index]
            gamma_max = alpha_v / (1 - alpha_v)
            gap = away_gap
        return d, gap, gamma_max

    def _pairwise_step(
        self,
        grad: np.ndarray,
        s: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        x = self.x
        active_set = np.where(x > 0)[0].tolist()
        v, v_index = LMO(grad, active_set=active_set, task='maximize')
        d_pw = s - v
        gamma_max = x[v_index]
        gap = grad @ d_pw
        return d_pw, gap, gamma_max

    def _blended_pairwise_step(
        self,
        grad: np.ndarray,
        s: np.ndarray,
        duality_gap: float,
    ) -> Tuple[np.ndarray, float]:
        x = self.x
        active_set = np.where(x > 0)[0].tolist()
        a, a_index = LMO(grad, active_set=active_set, task='maximize') # away step
        w, w_index = LMO(grad, active_set=active_set, task='minimize') # local FW step
        local_gap = grad @ (a - w)
        if local_gap >= - duality_gap:
            # optimize localy over the active set
            d = w - a
            gamma_max = x[a_index]
            gap = grad @ d
        else:
            d = s - x
            gamma_max = 1
            gap = duality_gap
        return d, gap, gamma_max
    
    def _choose_stepsize(self,
            edges: List[List[int]],
            d: np.ndarray, 
            gap: float,
            gamma_max: float,
            ) -> Tuple[np.ndarray, float, float]:
        """Function to choose the step size according to the selected strategy."""
        stepsize_strategy = self.stepsize_strategy
        if stepsize_strategy == 'armijo':
            self.gamma_max = gamma_max
            return self.armijo(edges, d, gap)
        elif stepsize_strategy == 'armijo decreasing':
            self.gamma_max = min(self.gamma_max, gamma_max)
            return self.armijo(edges, d, gap)
        elif stepsize_strategy == 'backtracking':
            x = self.x
            self.gamma_max = gamma_max
            gamma = self.backtracking_classic(edges, d, gap)
            x += gamma * d
            return x, self.calc_obj_function(edges, x)
        elif stepsize_strategy == 'decreasing':
            assert self.variant == 'FW', 'Decreasing step size is possible only for the classic FW'
            x, iter = self.x, self.training_iter
            gamma = 2 / (3 + iter)
            x = x + gamma * d
            of_new = self.calc_obj_function(edges, x)
            return x, of_new
        else:
            raise NotImplementedError('The stepsize strategy is not implemented.')
    
    def calc_obj_function(self, edges: List[List[int]], x: Optional[np.ndarray] = None) -> np.ndarray:
        tau = self.tau
        if x is None:
            x = self.x
        k = len(edges[0])
        LG = x[edges].prod(axis=1).sum()
        return LG + tau * (x ** k).sum()

    def calc_gradient(self, edges: List[List[int]]) -> np.ndarray:
        x, tau = self.x, self.tau
        N = len(x)

        hg_matrix = x[edges]
        hg_indices = np.array(edges)
        e, k = hg_matrix.shape
        remaining_products = np.zeros(hg_matrix.shape)
        for i in range(k):
            new_matrix = hg_matrix.copy()
            new_matrix[:,i] = np.ones(e)
            remaining_products[:,i] = new_matrix.prod(axis=1)
        grad = np.zeros(N)
        for i in range(N):
            grad[i] = remaining_products[hg_indices == i].sum()
        return grad + tau * k * (x ** (k - 1))
    
    def backtracking_classic(self, 
            edges: List[List[int]], 
            d: np.ndarray, 
            gap: float, 
            tau: Optional[float]=1.5, 
            nu: Optional[float] = 1.0) -> Tuple[np.ndarray, float, float]:
        
        def Q_t(of_value, gamma, gap, M, d_norm_squared):
            return of_value + gamma * gap + ((gamma ** 2) * M  * d_norm_squared)/ 2
        
        x, L_prev, gamma_max = self.x, self.L, self.gamma_max
        L = nu * L_prev
        d_norm_squared = np.linalg.norm(d) ** 2
        gamma = - gap / (L * d_norm_squared)
        gamma = min(gamma_max, gamma)
        initial_of_value = self.calc_obj_function(edges, x)
        while self.calc_obj_function(edges, x + gamma * d) > Q_t(initial_of_value, gamma, gap, L, d_norm_squared):
            L *= tau
            gamma_new = - gap / (L * d_norm_squared)
            gamma = min(gamma_new, gamma_max)
        self.L = L
        return gamma
    
    def armijo(self,
            edges: List[List[int]],
            d: np.ndarray, 
            gap: float,) -> Tuple[np.ndarray, float, float]:
        """
        Implements Armijo line search

        Paremeters
        ----------
        d: the descent direction of the algorithm
        duality_gap: the duality gap at the current iteration of the algorithm
        edges: list of lists containing the id of the nodes belinging to edges

        Returns
        -------
        x_new: the new value x
        of_new: the new value of the o.f. at x_new
        """
        x_old, alpha, delta, of_old, gamma_max = self.x, self.alpha, self.delta, self.of_value, self.gamma_max

        gamma = gamma_max
        x_new = x_old + gamma * d
        of_new = self.calc_obj_function(edges, x_new)
        m = 0
        while of_new > of_old + alpha * gamma * gap:
            m += 1
            gamma = delta * gamma
            x_new = x_old + gamma * d
            of_new = self.calc_obj_function(edges, x_new)
            assert m < 10000, "Armijo made 10 000 iteration, something must be wrong"
        self.gamma_max = gamma
        return x_new, of_new
    
def LMO(grad: np.ndarray, active_set: Optional[list] = None, task: Optional[str] = 'minimize') -> np.ndarray:
    """
    Takes the gradient and returns a vector of the same dimentions 
    in the unit simplex that minimizes the product with the gradient.
    """
    s = np.zeros(grad.shape)
    if active_set is None:
        grad_argmin = grad.argmin()
        s[grad_argmin] = 1
        return s
    else:
        grad_active_set = grad[active_set]
        if task == 'maximize':
            grad_index_active = grad_active_set.argmax()
        elif task == 'minimize':
            grad_index_active = grad_active_set.argmin()
        else:
            raise NotImplementedError('The LMO can either maximize or minimize over the active set.')
        s_index = active_set[grad_index_active]
        s[s_index] = 1
        return s, s_index