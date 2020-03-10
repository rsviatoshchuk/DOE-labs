from math import sqrt
from random import randint
import numpy as np


class TwoFactorExperiment:
    def __init__(self, y_min, y_max, m, x1min, x1max, x2min, x2max, p=0.99):
        self.x1min = x1min
        self.x1max = x1max
        self.x2min = x2min
        self.x2max = x2max

        self.number_of_exp = 3
        self.m = m
        self.p = p
        self.normalized_matrix = np.array([[-1, -1],
                                           [-1, +1],
                                           [+1, -1]])
        self.naturalized_matrix = [[self.x1min, self.x2min],
                                    [self.x1min, self.x2max],
                                    [self.x1max, self.x2min]]

        while True:
            self.feedback_func_matrix = np.array(
                [[randint(y_min, y_max) for i in range(self.m)] for _ in range(self.number_of_exp)])
            # self.feedback_func_matrix = np.array([[9, 10, 11, 15, 9],
            #                                     [15, 14, 10, 12, 14],
            #                                     [20, 18, 12, 10, 16]])

            self.mean_feedback_func_vector = self.feedback_func_matrix.mean(axis=1)

            self.variances = self.feedback_func_matrix.var(axis=1)
            self.major_deviation = sqrt(abs((2*(2*m-2))/(m*(m-4))))
            self.F_uv = self.get_F_uv()
            self.theta_uv = self.get_theta_uv()
            self.romanovsky_criterion = self.get_romanovsky_criterion()
            self.critical_romanovsky_criterion = self.get_critical_romanovsky_criterion()
            if self.start_check():
                self.norm_coef = self.get_coef()
                self.nat_coef = self.naturalize_coef()
                break
            else:
                self.m += 2
                print("Нове m = {}".format(self.m))

    def get_variances(self):
        variances = []
        for row in range(self.number_of_exp):
            variance = 0
            for y in range(self.m):
                variance += (self.feedback_func_matrix[row][y] - self.mean_feedback_func_vector[row])**2
                print(variance)
            variances.append(variance/self.m)
        return variances

    def get_F_uv(self):
        F_uv = []
        for i in range(self.number_of_exp):
            for j in range(self.number_of_exp):
                if i != j and j < i:
                    if self.variances[i] >= self.variances[j]:
                        F_uv.append(self.variances[i]/self.variances[j])
                    else:
                        F_uv.append(self.variances[j] / self.variances[i])
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
            if r > self.critical_romanovsky_criterion:
                print("Не підтверджується, додаємо до m = m + 2(обумовлено таблицею)")
                return False
        return True


    def naturalize_coef(self):
        delt_x1 = abs(self.x1max - self.x1min)/2
        delt_x2 = abs(self.x2max - self.x2min) / 2
        x10 = (self.x1max + self.x1min) / 2
        x20 = (self.x2max + self.x2min) / 2
        a0 = self.norm_coef[0] - self.norm_coef[1]*x10/delt_x1 - self.norm_coef[2]*x20/delt_x2
        a1 = self.norm_coef[1]/delt_x1
        a2 = self.norm_coef[2]/delt_x2
        return [a0.round(5), a1.round(5), a2.round(5)]

    def get_coef(self):
        my = self.mean_feedback_func_vector.mean()
        mx1 = self.normalized_matrix[:, 0].mean()
        mx2 = self.normalized_matrix[:, 1].mean()
        a1 = ((self.normalized_matrix[:, 0])**2).mean()
        a2 = (self.normalized_matrix[:, 0]*self.normalized_matrix[:, 1]).mean()
        a3 = ((self.normalized_matrix[:, 1])**2).mean()
        a11 = (self.normalized_matrix[:, 0]*self.mean_feedback_func_vector).mean()
        a22 = (self.normalized_matrix[:, 1]*self.mean_feedback_func_vector).mean()

        # print("my = {}, mx1 = {}, mx2 = {}\na1 = {}, a2 = {}, a3 = {}, a11 = {}, a22 = {}".format(my, mx1, mx2, a1, a2, a3, a11, a22))

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
        return [b0.round(5), b1.round(5), b2.round(5)]

    def check_norm_coef(self):
        print("b exp   b th")
        for i in range(self.number_of_exp):
            y_exp = self.norm_coef[0] + self.norm_coef[1]*self.normalized_matrix[i][0] + self.norm_coef[2]*self.normalized_matrix[i][1]
            y_th = self.mean_feedback_func_vector[i]
            print(y_exp.round(4), "  ", y_th.round(4))

            if y_exp.round(4) != y_th.round(4):
                print("Невідповідність")

    def check_nat_coef(self):
        print("a exp      a th")
        for i in range(self.number_of_exp):
            y_exp = self.nat_coef[0] + self.nat_coef[1]*self.naturalized_matrix[i][0] + self.nat_coef[2]*self.naturalized_matrix[i][1]
            y_th = self.mean_feedback_func_vector[i]
            print(y_exp.round(4), "  ", y_th.round(4))

            if y_exp.round(3) != y_th.round(3):
                print("Невідповідність")


if __name__ == '__main__':
    y_max = 70
    y_min = -30

    x1min = -30
    x1max = 0
    x2min = -25
    x2max = 10

    m = 6
    p = 0.99
    example = TwoFactorExperiment(y_min, y_max, m, x1min, x1max, x2min, x2max)

    print("Матриця функцій відгуку:\n{}".format(example.feedback_func_matrix))
    print("Сер. ар. функцій відгуку: {}".format(example.mean_feedback_func_vector))
    print("Дисперсії: {}".format(example.variances))
    print("Основне відхилення: {}".format(example.major_deviation))
    print("F_uv: {}".format(example.F_uv))
    print("Theta uv: {}".format(example.theta_uv))
    print("Критерії Романовського: {}".format(example.romanovsky_criterion))
    print("Критичне значення критерія Романовського: {}".format(example.critical_romanovsky_criterion))
    print("Норм. коеф {}".format(example.norm_coef))
    example.check_norm_coef()
    print("Натуралізовані. коеф {}".format(example.nat_coef))
    example.check_nat_coef()
