from math import sqrt
from random import randint
import numpy as np
from scipy.stats import t, f


class ThreeFactorFractionalExperiment:
    def __init__(self, y_min, y_max, m, x1min, x1max, x2min, x2max, x3min, x3max, p=0.99):
        self.x1min = x1min
        self.x1max = x1max
        self.x2min = x2min
        self.x2max = x2max
        self.x3min = x3min
        self.x3max = x3max

        self.number_of_exp = 4
        self.m = m
        self.p = p
        self.normalized_matrix = np.array([[+1, -1, -1, -1],
                                           [+1, -1, +1, +1],
                                           [+1, +1, -1, +1],
                                           [+1, +1, +1, -1]])
        self.naturalized_matrix = np.array([[self.x1min, self.x2min, self.x3min],
                                            [self.x1min, self.x2max, self.x3max],
                                            [self.x1max, self.x2min, self.x3max],
                                            [self.x1max, self.x2max, self.x3min]])

        while True:
            self.feedback_func_matrix = np.array(
                [[randint(y_min, y_max) for i in range(self.m)] for _ in range(self.number_of_exp)])

            self.mean_feedback_func_vector = self.feedback_func_matrix.mean(axis=1)

            self.nat_coef = self.get_coef()
            self.check_nat_coef()
            break

    def get_f_critical(p, f1, f2):
        return scipy.stats.f.ppf(p, f1, f2)

    def get_t_critical(p, df):
        return scipy.stats.t.ppf(p, df)

    def get_c_critical(p, f1, f2):
        return 1 / (1 + (f2 - 1) / scipy.stats.f.ppf(1 - (1 - prob) / f2, f1, (f2 - 1) * f1))

    def get_coef(self):
        my = self.mean_feedback_func_vector.mean()

        mx1 = self.naturalized_matrix[:, 0].mean()
        mx2 = self.naturalized_matrix[:, 1].mean()
        mx3 = self.naturalized_matrix[:, 2].mean()

        a1 = (self.naturalized_matrix[:, 0] * self.mean_feedback_func_vector).mean()
        a2 = (self.naturalized_matrix[:, 1] * self.mean_feedback_func_vector).mean()
        a3 = (self.naturalized_matrix[:, 2] * self.mean_feedback_func_vector).mean()

        a11 = ((self.normalized_matrix[:, 0]) ** 2).mean()
        a22 = ((self.normalized_matrix[:, 1]) ** 2).mean()
        a33 = ((self.normalized_matrix[:, 2]) ** 2).mean()

        a12 = (self.naturalized_matrix[:, 0] * self.naturalized_matrix[:, 1]).mean()
        a21 = (self.naturalized_matrix[:, 0] * self.naturalized_matrix[:, 1]).mean()

        a13 = (self.naturalized_matrix[:, 0] * self.naturalized_matrix[:, 2]).mean()
        a31 = (self.naturalized_matrix[:, 0] * self.naturalized_matrix[:, 2]).mean()

        a23 = (self.naturalized_matrix[:, 1] * self.naturalized_matrix[:, 2]).mean()
        a32 = (self.naturalized_matrix[:, 1] * self.naturalized_matrix[:, 2]).mean()

        det = np.linalg.det(np.array([[1, mx1, mx2, mx3],
                                      [mx1, a11, a12, a13],
                                      [mx2, a12, a22, a32],
                                      [mx3, a13, a23, a33]]))

        det_b0 = np.linalg.det(np.array([[my, mx1, mx2, mx3],
                                         [a1, a11, a12, a13],
                                         [a2, a12, a22, a32],
                                         [a3, a13, a23, a33]]))

        det_b1 = np.linalg.det(np.array([[1, my, mx2, mx3],
                                         [mx1, a1, a12, a13],
                                         [mx2, a2, a22, a32],
                                         [mx3, a3, a23, a33]]))

        det_b2 = np.linalg.det(np.array([[1, mx1, my, mx3],
                                         [mx1, a11, a1, a13],
                                         [mx2, a12, a2, a32],
                                         [mx3, a13, a3, a33]]))

        det_b3 = np.linalg.det(np.array([[1, mx1, mx2, my],
                                         [mx1, a11, a12, a1],
                                         [mx2, a12, a22, a2],
                                         [mx3, a13, a23, a3]]))

        b0 = det_b0 / det
        b1 = det_b1 / det
        b2 = det_b2 / det
        b3 = det_b3 / det
        return [b0.round(5), b1.round(5), b2.round(5), b3.round(5)]

    def check_nat_coef(self):
        print("a exp      a th")
        for i in range(self.number_of_exp):
            y_exp = self.nat_coef[0] + self.nat_coef[1]*self.naturalized_matrix[i][0] + self.nat_coef[2]*self.naturalized_matrix[i][1] + self.nat_coef[3]*self.naturalized_matrix[i][2]
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
    x3min = -25
    x3max = -5

    m = 3
    p = 0.95
    example = ThreeFactorFractionalExperiment(y_min, y_max, m, x1min, x1max, x2min, x2max, x3min, x3max, p)

    print("Матриця функцій відгуку:\n{}".format(example.feedback_func_matrix))
    print("Сер. ар. функцій відгуку: {}".format(example.mean_feedback_func_vector))
    print("Натуралізовані. коеф {}".format(example.nat_coef))
