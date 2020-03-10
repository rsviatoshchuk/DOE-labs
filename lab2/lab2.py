from math import sqrt
from random import randint
import numpy as np


class TwoFactorExperiment():
    def __init__(self, y_min, y_max, m, p=0.99):
        self.number_of_exp = 3
        self.m = m
        self.p = p
        self.normalized_matrix = np.array([[-1, -1],
                                           [-1, +1],
                                           [+1, -1]])
        self.feedback_func_matrix = np.array([[randint(y_min, y_max) for _ in range(m)] for _ in range(self.number_of_exp)])

        self.mean_feedback_func_vector = self.feedback_func_matrix.mean(axis=1)
        self.variances = self.feedback_func_matrix.var(axis=1)
        self.major_deviation = sqrt(abs((2*(2*m-2))/(m*(m-4))))

        self.F_uv = self.get_F_uv()
        self.theta_uv = self.get_theta_uv()
        self.romanovsky_criterion = self.get_romanovsky_criterion()
        self.critical_romanovsky_criterion = self.get_critical_romanovsky_criterion()

    def get_F_uv(self):
        F_uv = []
        for i in range(self.number_of_exp):
            for j in range(self.number_of_exp):
                if i != j and j < i:
                    if self.variances[i] >= self.variances[j]:
                        F_uv.append(self.variances[i]/self.variances[j])
                    else:
                        F_uv.append(self.variances[i] / self.variances[j])
        return F_uv

    def get_theta_uv(self):
        return [(1-2/self.m)*F_uvi for F_uvi in self.F_uv]

    def get_romanovsky_criterion(self):
        return [abs(theta-1)/self.major_deviation for theta in self.theta_uv]

    def get_critical_romanovsky_criterion(self):
        rom_crit = {0.99: [1.72, 2.16, 2.43, 2.62, 2.75, 2.90, 3.08],
                    0.98: [1.72, 2.13, 2.37, 2.54, 2.66, 2.80, 2.96],
                    0.95: [1.71, 2.10, 2.27, 2.41, 2.52, 2.64, 2.78],
                    0.90: [1.69, 2.00, 2.17, 2.29, 2.39, 2.49, 2.62]}
        m_rom = {2: 0,
                 6: 1,
                 8: 2,
                 10: 3,
                 12: 4,
                 15: 5,
                 20: 6}
        return rom_crit[self.p][m_rom[self.m]]


if __name__ == '__main__':
    y_max = 70
    y_min = -30
    m = 6
    p = 0.99
    x1_min = -20
    x1_max = 0
    x2_min = -25
    x2_max = 10

    t = TwoFactorExperiment(y_min, y_max, m)
    print(t.feedback_func_matrix)
    print(t.mean_feedback_func_vector)
    print(t.variances)
    print(t.major_deviation)
    print(t.F_uv)
    print(t.theta_uv)
    print(t.romanovsky_criterion)
    print(t.critical_romanovsky_criterion)
