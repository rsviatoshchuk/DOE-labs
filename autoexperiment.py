import numpy
from math import sqrt
from scipy.stats import t, f
from itertools import combinations


class Experiment:
    def __init__(self):
        # Характеристики експерименту
        self.factors = None
        self.experiments = None
        self.fractionality = None           # 0 for full factorial experiment
        self.l = 2
        self.factors_ranges = None
        self.resp_var_range = None

        self.interaction_combinations = None
        self.interaction_ranges = None

        # Флаги
        self.five_level_flag = None
        self.interaction_flag = None
        self.quadr_flag = None

        # Нормалізовані частини
        self.main_part = None
        self.interaction_part = None
        self.quadr_part = None
        self.norm_matrix = None

        # Натуралізовані частини
        self.nat_matrix = None
        self.nat_interaction_part = None
        self.nat_quadr_part = None

        # Функції відгуку
        self.resp_var_matrix = None

    def set_experiment(self, num_of_factors, factors_ranges, response_var_range,
                       fractionality=0, interaction=False, quadratic=False, fivelevel=False):

        self.factors = num_of_factors
        self.factors_ranges = numpy.array(factors_ranges)
        self.resp_var_range = response_var_range

        self.five_level_flag = fivelevel
        self.fractionality = fractionality
        self.interaction_flag = interaction
        self.quadr_flag = quadratic

    def set_main_part(self, matrix):
        """func for manual norm matrix"""
        self.main_part = matrix

    def set_response_variable_matrix(self, matrix):
        self.resp_var_matrix = matrix

    def gen_random_response_var(self, m):
        self.resp_var_matrix = numpy.random.uniform(low=self.resp_var_range[0],
                                                    high=self.resp_var_range[1],
                                                    size=(len(self.main_part), m))

    def gen_main_part(self):
        """func generate normalized matrix"""
        def conv(str_el):
            if str_el == "0":
                str_el = "-1"
            return int(str_el)
        vconv = numpy.vectorize(conv)

        form = "{{0:0{}b}}".format(self.factors)  # formatting for getting 0001 instead 1
        str_array = numpy.array([list(form.format(i)[::-1]) for i in range(2 ** self.factors)])
        num_array = vconv(str_array)[:2**(self.factors-self.fractionality), :]

        # code for 5-level part
        if self.five_level_flag:
            five_level_part = []

            for col in range(self.factors):
                row = [0] * self.factors

                row[col] = -self.l
                five_level_part.append(row[:])
                row[col] = self.l
                five_level_part.append(row[:])

            num_array = numpy.append(num_array, numpy.array(five_level_part), axis=0)

        self.main_part = num_array

    def gen_interaction_part(self):
        """func generate interaction part(only for 2 and 3 factors)"""
        # Отримуємо всі можливі комбінації взаємодії факторів
        comb = []
        for n in range(2, self.factors + 1):
            comb.extend(list(map(tuple, combinations(range(1, self.factors + 1), n))))
        self.interaction_combinations = numpy.array(comb)

        # Отримуємо межі факторів взіємодії
        inter_ranges = []
        for interaction in self.interaction_combinations:
            min_inter = numpy.prod([self.factors_ranges[i - 1][0] for i in interaction])
            max_inter = numpy.prod([self.factors_ranges[i - 1][1] for i in interaction])
            inter_ranges.append((min_inter, max_inter))
        self.interaction_ranges = numpy.array(inter_ranges)

        # Отримуємо матрицю взаємодії факторів
        matrix = []
        for raw in self.main_part:
            inter_raw = []
            for interaction in self.interaction_combinations:
                inter_raw.append(numpy.prod([raw[i - 1] for i in interaction]))
            matrix.append(inter_raw)
        self.interaction_part = numpy.array(matrix)

    def gen_quadratic_part(self):
        self.quadr_part = self.main_part**2

    def naturalize(self):
        average_x = self.factors_ranges.mean(axis=1)
        delta_x = average_x - self.factors_ranges[:, 0]

        self.nat_matrix = self.main_part * delta_x + average_x

        # if self.interaction_flag:
        #     self.nat_interaction_part =

        if self.quadr_flag:
            self.nat_quadr_part = self.nat_matrix ** 2

    def gen_norm_matrix(self):

        if self.interaction_flag:
            if self.main_part.shape[0] == self.interaction_part.shape[0]:
                matrix = numpy.append(self.main_part, self.interaction_part, axis=1)
            else:
                inter_matrix = numpy.append(self.interaction_part,
                                            numpy.zeros((self.main_part.shape[0] - self.interaction_part.shape[0],
                                                         self.interaction_part.shape[1])),
                                            axis=0)
                matrix = numpy.append(self.main_part, inter_matrix, axis=1)
            if self.quadr_flag:
                self.norm_matrix = numpy.append(matrix, self.quadr_part, axis=1)
                return
            else:
                self.norm_matrix = matrix
                return
        if self.quadr_flag:
            self.norm_matrix = numpy.append(self.main_part, self.quadr_part, axis=1)

    def get_l_central(self, k, p):
        return sqrt(sqrt((2 ** (k - p - 2)) * (2 ** (k - p) + 2 * k + 1)) - 2 ** (k - p - 1))

    def get_l_rototable(self, k):
        return sqrt(k)


a = Experiment()
a.set_experiment(3, [(-5, 15), (10, 60), (10, 20)], (205, 231.666), interaction=True, quadratic=True, fivelevel=True)
a.gen_main_part()
a.gen_interaction_part()
a.gen_quadratic_part()
a.gen_norm_matrix()
a.gen_random_response_var(3)
a.naturalize()

print("\nМежі факторів:")
print(a.factors_ranges)

print("\nНормалізована матриця планування:")
print(a.main_part)

print("\nКомбінації взаємодії:")
print(a.interaction_combinations)

print("\nМежі комбінацій взаємодії:")
print(a.interaction_ranges)

print("\nВзаємодія:")
print(numpy.array(a.interaction_part))

print("\nКвадратична частина:")
print(a.quadr_part)

print("\nГотова матриця планування:")
print(a.norm_matrix)

print("Матриця функції відгуку")
print(a.resp_var_matrix)

print("\nНатуралізована матриця планування:")
print(a.nat_matrix)

print("\nНатуралізована заємодія:")
print(a.nat_interaction_part)

print("\nНатуралізована квадратична частина:")
print(a.nat_quadr_part)
