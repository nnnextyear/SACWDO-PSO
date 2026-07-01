import math


class AdaptiveParams:
    """Adaptive parameter controller for inertia weight and acceleration coefficients.

    Uses tanh for inertia weight decay and cosh for coefficient smoothing.
    Adaptive mu based on the ratio between personal best and global best costs.
    """

    def __init__(self, w_max=0.9, w_min=0.2,
                 c1_max=2.5, c1_min=0.5,
                 c2_max=2.5, c2_min=0.5,
                 beta=5.0, gamma=4.0):
        self.w_max = w_max
        self.w_min = w_min
        self.c1_max = c1_max
        self.c1_min = c1_min
        self.c2_max = c2_max
        self.c2_min = c2_min
        self.beta = beta
        self.gamma = gamma

    def compute(self, pbest_cost, gbest_cost, iteration, max_iterations):
        if gbest_cost < 1e-10:
            mu = 1.0
        else:
            mu = (pbest_cost - gbest_cost) / max(pbest_cost, 1e-10)
            mu = max(mu, 0.01)

        t = iteration / max_iterations

        delta = self.w_max - (self.w_max - self.w_min) * t
        w = mu * math.tanh(self.beta * delta)
        w = max(0.1, min(0.95, w))

        psi_c1 = self.c1_max - (self.c1_max - self.c1_min) * t
        psi_c2 = self.c2_min + (self.c2_max - self.c2_min) * t

        c1 = mu * math.cosh(self.gamma * psi_c1)
        c1 = max(0.3, min(3.0, c1))

        c2 = mu * math.cosh(self.gamma * psi_c2)
        c2 = max(0.3, min(3.0, c2))

        return w, c1, c2
