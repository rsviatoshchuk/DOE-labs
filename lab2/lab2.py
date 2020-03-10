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

        self.norm_coef = self.get_coef()

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

    def start_check(self):
        for r in self.romanovsky_criterion:
            if r>self.critical_romanovsky_criterion:
                print("не підтверджується")

    def get_coef(self):
        my = sum(self.mean_feedback_func_vector) / len(self.mean_feedback_func_vector)
        mx1 = sum(self.normalized_matrix[:, 0]) / len(self.normalized_matrix[:, 0])
        mx2 = sum(self.normalized_matrix[:, 1]) / len(self.normalized_matrix[:, 1])
        a1 = sum(self.normalized_matrix[:, 0])**2 / len(self.normalized_matrix[:, 0])
        a2 = sum(self.normalized_matrix[:, 0]*self.normalized_matrix[:, 1]) / len(self.normalized_matrix[:, 1])
        a3 = sum(self.normalized_matrix[:, 1])**2 / len(self.normalized_matrix[:, 1])
        a11 = sum(self.normalized_matrix[:, 0]*self.mean_feedback_func_vector) / len(self.normalized_matrix[:, 1])
        a22 = sum(self.normalized_matrix[:, 1]*self.mean_feedback_func_vector) / len(self.normalized_matrix[:, 1])

        print("my = {}, mx1 = {}, mx2 = {}\na1 = {}, a2 = {}, a3 = {}, a11 = {}, a22 = {}".format(my, mx1, mx2, a1, a2, a3, a11, a22))

        det = np.linalg.det(np.array([[1, mx1, mx2],
                                      [mx1, a1, a2],
                                      [mx2, a2, a3]]))
        det_b0 = np.linalg.det(np.array([[my, mx1, mx2],
                                         [a11, a1, a2],
                                         [a22, a2, a3]]))
        det_b1 = np.linalg.det(np.array([[1, my, mx2],
                                         [mx1, a11, a2],
                                         [mx2, a22, a3]]))
        det_b2 = np.linalg.det(np.array([[1, mx1, my],
                                         [mx1, a1, a11],
                                         [mx2, a2, a22]]))
        b0 = det_b0 / det
        b1 = det_b1 / det
        b2 = det_b2 / det
        return [b0, b1, b2]

    def check_coef(self):
        for i in range(self.number_of_exp):
            y_exp = self.norm_coef[0] + self.norm_coef[1]*self.normalized_matrix[i][0] + self.norm_coef[2]*self.normalized_matrix[i][1]
            y_th = self.mean_feedback_func_vector[i]
            print(y_exp, y_th)


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
    print("Матриця функцій відгуку:\n{}".format(t.feedback_func_matrix))
    print("Сер. ар. функцій відгуку:\n{}".format(t.mean_feedback_func_vector))
    print("Дисперсії:\n{}".format(t.variances))
    print("Основне відхилення: {}".format(t.major_deviation))
    print("F_uv:\n{}".format(t.F_uv))
    print("Theta uv:\n{}".format(t.theta_uv))
    print("Критерії Романовсткого: \n{}".format(t.romanovsky_criterion))
    print("Критичне значення критерія Романовського: {}".format(t.critical_romanovsky_criterion))
    t.start_check()
    print("Норм. коеф {}".format(t.norm_coef))
    t.check_coef()
