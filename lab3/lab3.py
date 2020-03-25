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
        # Критерій Кохрена
        while True:
            self.feedback_func_matrix = np.array(
                [[randint(y_min, y_max) for i in range(self.m)] for _ in range(self.number_of_exp)])

            self.mean_feedback_func_vector = self.feedback_func_matrix.mean(axis=1)

            self.nat_coef = self.get_coef()
            print("Нат. коеф:", self.nat_coef)
            self.check_nat_coef()
            if self.cochran_check():
                print("Дисперсії однорідні")
                break
            else:
                print("Згідно критерія Кохрена дисперсії неоднорідні. Отже потрібно збільшити m")
                self.m += 1

        self.student_check()

        self.fisher_check()

    def get_f_critical(self, p, f3, f4):
        return f.ppf(p, f3, f4)

    def get_t_critical(self, p, df):
        return t.ppf(p, df)

    def get_cochran_critical(self, p, f1, f2):
        return 1 / (1 + (f2 - 1) / f.ppf(1 - (1 - p) / f2, f1, (f2 - 1) * f1))

    def get_coef(self):
        my = self.mean_feedback_func_vector.mean()

        mx1 = self.naturalized_matrix[:, 0].mean()
        mx2 = self.naturalized_matrix[:, 1].mean()
        mx3 = self.naturalized_matrix[:, 2].mean()

        a1 = (self.naturalized_matrix[:, 0] * self.mean_feedback_func_vector).mean()
        a2 = (self.naturalized_matrix[:, 1] * self.mean_feedback_func_vector).mean()
        a3 = (self.naturalized_matrix[:, 2] * self.mean_feedback_func_vector).mean()

        a11 = ((self.naturalized_matrix[:, 0]) ** 2).mean()
        a22 = ((self.naturalized_matrix[:, 1]) ** 2).mean()
        a33 = ((self.naturalized_matrix[:, 2]) ** 2).mean()

        a12 = (self.naturalized_matrix[:, 0] * self.naturalized_matrix[:, 1]).mean()
        a21 = (self.naturalized_matrix[:, 0] * self.naturalized_matrix[:, 1]).mean()

        a13 = (self.naturalized_matrix[:, 0] * self.naturalized_matrix[:, 2]).mean()
        a31 = (self.naturalized_matrix[:, 0] * self.naturalized_matrix[:, 2]).mean()

        a23 = (self.naturalized_matrix[:, 1] * self.naturalized_matrix[:, 2]).mean()
        a32 = (self.naturalized_matrix[:, 1] * self.naturalized_matrix[:, 2]).mean()

        # print(a1, a2, a3)
        # print(a11, a22, a33)
        # print(a12, a13, a32)

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
        print("\nПеревірка знайдених коефіцієнтів:")
        print("a exp      a th")
        for i in range(self.number_of_exp):
            y_exp = self.nat_coef[0] + self.nat_coef[1]*self.naturalized_matrix[i][0] + self.nat_coef[2]*self.naturalized_matrix[i][1] + self.nat_coef[3]*self.naturalized_matrix[i][2]
            y_th = self.mean_feedback_func_vector[i]
            print(y_exp.round(4), "  ", y_th.round(4))

            if y_exp.round(2) != y_th.round(2):
                print("Невідповідність")

    def get_variances(self):
        return self.feedback_func_matrix.var(axis=1)

    def cochran_check(self):
        variances = self.get_variances()
        g = variances.max()/variances.sum()

        f1 = self.m + 1
        f2 = self.number_of_exp
        g_critical = self.get_cochran_critical(self.p, f1, f2)
        print("\nG critical = ", g_critical)
        print("G = ", g)
        return g <= g_critical

    def student_check(self):
        variances = self.get_variances()
        average_variance = variances.mean()
        beta_variance = average_variance/(self.number_of_exp * self.m)
        t = [abs(coef)/beta_variance for coef in self.nat_coef]

        f1 = self.m + 1
        f2 = self.number_of_exp
        f3 = f1 * f2

        t_critical = self.get_t_critical(self.p, f3)

        self.significant_coeffs = 0
        for i in range(len(t)):
            if t[i] <= t_critical:
                self.nat_coef[i] = 0
            else:
                self.significant_coeffs += 1

        print(self.nat_coef)
        print("Кількість значущих коефіцієнтів: ", self.significant_coeffs)

        self.y = [self.nat_coef[0] + self.nat_coef[1] * self.naturalized_matrix[i, 0] + self.nat_coef[2] * self.naturalized_matrix[i, 1] + self.nat_coef[3] * self.naturalized_matrix[i, 2] for i in range(self.number_of_exp)]
        print("Обчислені y:", self.y)

    def fisher_check(self):
        s_ad = (self.m/(self.number_of_exp - self.significant_coeffs)) * sum([(self.y[i] - self.mean_feedback_func_vector[i]) ** 2 for i in range(self.number_of_exp)])

        variances = self.get_variances()
        average_variance = variances.mean()

        f = s_ad / average_variance

        f1 = self.m + 1
        f2 = self.number_of_exp
        f3 = f1 * f2
        f4 = self.number_of_exp - self.significant_coeffs

        f_critical = self.get_f_critical(self.p, f3, f4)

        print("\nF critical = ", f_critical)
        print("F = ", f)

        if f <= f_critical:
            print("Рівняння лінійної регресії адекватне")
        else:
            print("Рівняння лінійної регресії неадекватне")


if __name__ == '__main__':
    y_max = 202
    y_min = 173

    x1min = -30
    x1max = 0
    x2min = -25
    x2max = 10
    x3min = -25
    x3max = -5

    m = 3
    p = 0.95
    example = ThreeFactorFractionalExperiment(y_min, y_max, m, x1min, x1max, x2min, x2max, x3min, x3max, p)

    print("\nМатриця факторів:\n{}".format(example.naturalized_matrix))
    print("\nМатриця функцій відгуку:\n{}".format(example.feedback_func_matrix))
    print("\nСер. ар. функцій відгуку: {}".format(example.mean_feedback_func_vector))
