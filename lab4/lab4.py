from math import sqrt
from random import randint
import numpy as np
from scipy.stats import t, f


class ThreeFactorExperiment:
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
        self.normalized_matrix = np.array([[+1, -1, -1, -1, +1, +1, +1, -1],
                                           [+1, -1, +1, +1, +1, -1, -1, +1],
                                           [+1, +1, -1, +1, -1, +1, -1, +1],
                                           [+1, +1, +1, -1, -1, -1, +1, -1],
                                           [+1, +1, -1, -1, -1, -1, +1, +1],
                                           [+1, +1, -1, +1, -1, +1, -1, -1],
                                           [+1, +1, +1, -1, +1, -1, -1, -1],
                                           [+1, +1, +1, +1, +1, +1, +1, +1]])
        self.naturalized_matrix = np.array([[self.x1min, self.x2min, self.x3min],
                                            [self.x1min, self.x2max, self.x3max],
                                            [self.x1max, self.x2min, self.x3max],
                                            [self.x1max, self.x2max, self.x3min],
                                            [self.x1max, self.x2min, self.x3min],
                                            [self.x1max, self.x2min, self.x3max],
                                            [self.x1max, self.x2max, self.x3min],
                                            [self.x1max, self.x2max, self.x3max]
                                            ])
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
        pass

    def check_nat_coef(self):
        pass

    def get_variances(self):
        return self.feedback_func_matrix.var(axis=1)

    def cochran_check(self):
        pass

    def student_check(self):
        pass

    def fisher_check(self):
        pass


if __name__ == '__main__':
    y_max = 232
    y_min = 205

    x1min = -5
    x1max = 15
    x2min = 10
    x2max = 60
    x3min = 10
    x3max = 20

    m = 3
    p = 0.95
    example = ThreeFactorExperiment(y_min, y_max, m, x1min, x1max, x2min, x2max, x3min, x3max, p)

    print("\nМатриця факторів:\n{}".format(example.naturalized_matrix))
    print("\nМатриця функцій відгуку:\n{}".format(example.feedback_func_matrix))
    print("\nСер. ар. функцій відгуку: {}".format(example.mean_feedback_func_vector))
