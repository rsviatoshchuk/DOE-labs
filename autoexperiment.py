import numpy
from math import sqrt
from scipy.stats import t, f


class Experiment:
    def __init__(self):
        # Характеристики експерименту
        self.factors = None
        self.experiments = None
        self.fractionality = None           # 0 for full factorial experiment
        self.l = 2
        self.factors_range = None

        # Флаги
        self.five_level_flag = None
        self.interaction_flag = None
        self.quadr_flag = None

        # Нормалізовані частини
        self.norm_matrix = None
        self.interaction_part = None
        self.quadr_part = None

        # Натуралізовані частини
        self.nat_matrix = None
        self.nat_interaction_part = None
        self.nat_quadr_part = None

        # Функції відгуку
        self.resp_var_matrix = None

    def set_experiment(self, num_of_factors, fractionality=0, interaction=False, quadratic=False, fivelevel=False):
        self.factors = num_of_factors
        self.five_level_flag = fivelevel
        self.fractionality = fractionality
        self.interaction_flag = interaction
        self.quadr_flag = quadratic

    def set_norm_matr(self, matrix):
        """func for manual norm matrix"""
        self.norm_matrix = matrix

    def set_response_variable_matrix(self, matrix):
        self.resp_var_matrix = matrix

    def gen_norm_matr(self):
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

        self.norm_matrix = num_array

    def gen_interaction_part(self):
        """func generate interaction part(only for 2 and 3 factors)"""
        if self.factors == 2:
            matrix = numpy.array([[+1], [+1], [-1], [-1]])
        elif self.factors == 3:
            matrix = numpy.array([[+1, +1, +1, -1],
                                  [+1, -1, -1, +1],
                                  [-1, +1, -1, +1],
                                  [-1, -1, +1, -1],
                                  [-1, -1, +1, +1],
                                  [-1, +1, -1, -1],
                                  [+1, -1, -1, -1],
                                  [+1, +1, +1, +1]])
        else:
            raise ValueError

        self.interaction_part = matrix

    def gen_quadratic_part(self):
        self.quadr_part = self.norm_matrix**2

    def get_l_central(self, k, p):
        return sqrt(sqrt((2 ** (k - p - 2)) * (2 ** (k - p) + 2 * k + 1)) - 2 ** (k - p - 1))

    def get_l_rototable(self, k):
        return sqrt(k)


a = Experiment()
a.set_experiment(3, fivelevel=True)
a.gen_norm_matr()
a.gen_interaction_part()
a.gen_quadratic_part()

print("\nНормалізована матриця планування:")
print(a.norm_matrix)

print("\nВзаємодія:")
print(a.interaction_part)

print("\nКвадратична частина:")
print(a.quadr_part)

# print("\nНормалізована матриця планування(5 рівнів):")
# print(a.get_norm_matrix(3, 1.41))
#
# print("\nНормалізована матриця планування  з ефектом взаємодії:")
# print(a.get_norm_matrix_inter(3, 1.41))
#
# print("\nНормалізована матриця планування  з квадратичними членами:")
# print(a.get_norm_matrix_quad(3, 1.41))
#
# print("\nНормалізована матриця планування  з ефектом взаємодії та квадратичними членами:")
# print(a.get_norm_matrix_inter_quad(3, 2))
