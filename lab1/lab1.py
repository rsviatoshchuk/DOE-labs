import numpy as np


class Experiment:
    def __init__(self, low_limit, up_limit, number_of_experiments, number_of_factors, regression_coefficients):
        self.low_limit = low_limit
        self.up_limit = up_limit
        self.number_of_experiments = number_of_experiments
        self.number_of_factors = number_of_factors
        self.regression_coefficients = regression_coefficients

        self.matrix = None
        self.normalized_matrix = None
        self.zero_factor_level_vector = None
        self.factor_change_interval_vector = None
        self.feedback_function_vector = None
        self.y_et = None

    def start_random_experiment(self):
        self.random_fill()
        self.get_zero_factor_level_vector()
        self.get_factor_change_interval_vector()
        self.get_feedback_function_vector()
        self.get_normalized_matrix()

    def random_fill(self):
        self.matrix = np.random.randint(self.low_limit, self.up_limit + 1,
                                        (self.number_of_experiments, self.number_of_factors))

    def get_zero_factor_level_vector(self):
        self.zero_factor_level_vector = np.array(
            [self.get_zero_factor_level(self.matrix[:, i]) for i in range(self.number_of_factors)], float)

    def get_factor_change_interval_vector(self):
        self.factor_change_interval_vector = np.array(
            [self.get_factor_change_interval(self.zero_factor_level_vector[i], self.matrix[:, i].min())
             for i in range(self.number_of_factors)], float)

    def get_feedback_function_vector(self):
        self.feedback_function_vector = np.array(
            [self.get_feedback_function(self.matrix[i]) for i in range(self.number_of_experiments)], float)

    def get_normalized_matrix(self):
        self.normalized_matrix = np.empty((self.number_of_experiments, self.number_of_factors))
        for i in range(self.number_of_factors):
            for j in range(self.number_of_experiments):
                self.normalized_matrix[j, i] = self.normalize_factor(self.matrix[j, i],
                                                                     self.zero_factor_level_vector[i],
                                                                     self.factor_change_interval_vector[i])

    def get_y_et(self):
        self.y_et = self.get_feedback_function(self.zero_factor_level_vector)

    def get_zero_factor_level(self, factor_values):
        return (factor_values.max() + factor_values.min())/2

    def get_factor_change_interval(self, zero_factor_level, minimal_factor_value):
        return zero_factor_level - minimal_factor_value

    def normalize_factor(self, factor_value, zero_factor_level, factor_change_interval):
        return round((factor_value - zero_factor_level)/factor_change_interval, 2)

    def get_feedback_function(self, factors_values):
        y = self.regression_coefficients[0]
        for i in range(self.number_of_factors):
            y += self.regression_coefficients[i+1] * factors_values[i]
        return y


if __name__ == '__main__':
    low_lim = 0
    up_lim = 20
    num_of_ex = 8
    num_of_fact = 3
    reg_coef = [2, 3, 4, 5]
    ex = Experiment(low_lim, up_lim, num_of_ex, num_of_fact, reg_coef)
    ex.start_random_experiment()
    print("Коефіцієнти регресії:", ex.regression_coefficients)
    print("Матриця факторів:\n", ex.matrix)
    print("Нульовий рівень х0:\n", ex.zero_factor_level_vector)
    print("dx:\n", ex.factor_change_interval_vector)
    print("Значення функції відгуку:\n", ex.feedback_function_vector)
    print("Нормалізована матриця факторів:\n", ex.normalized_matrix)
    print("Yет =", ex.y_et)
    print("Критерій вибору: min(Y)")
    min_y = ex.feedback_function_vector.min()
    print("Y =", min_y)
    factors = ex.matrix[np.where(ex.feedback_function_vector == min_y)]
    print("Значення факторів в точці плану:\n", factors)
    # print("Y = a0 + a1*{0} + a2*{1} + a3*{2}".format([factor for factor in factors[0, :]]))
