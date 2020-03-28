import numpy
from math import sqrt
from scipy.stats import t, f


class Experiment:
    def __init__(self):
        self.normalized_matrix = None

    def get_2level_normalized_matrix(self, num_of_factors):
        def conv(str_el):
            if str_el == "0":
                str_el = "-1"
            return int(str_el)
        vconv = numpy.vectorize(conv)

        num_of_levels = 2  # for future
        form = "{{0:0{}b}}".format(num_of_factors)  # formatting for getting 0001 instead 1
        str_array = numpy.array([list(form.format(i)[::-1]) for i in range(num_of_levels ** num_of_factors)])
        return vconv(str_array)

    def get_5level_part(self, num_of_factors, l):
        star_dots_matrix = []

        for col in range(num_of_factors):
            row = [0]*num_of_factors

            row[col] = -l
            star_dots_matrix.append(row[:])
            row[col] = l
            star_dots_matrix.append(row[:])

        return numpy.array(star_dots_matrix)

    def get_interaction_part(self, num_of_factors):
        pass

    def get_quadratic_part(self, num_of_factors, l):
        quadratic_matrix = numpy.append(self.get_2level_normalized_matrix(num_of_factors),
                                        self.get_5level_part(num_of_factors, l), axis=0)

        return quadratic_matrix**2

    def get_l_central(self, k, p):
        return sqrt(sqrt((2**(k-p-2))*(2**(k-p)+2*k+1))-2**(k-p-1))
    
    def get_l_rototable(self, k):
        return sqrt(k)


a = Experiment()
#print(a.get_2level_normalized_matrix(3))
# print(a.get_5level_part(5, 1.44))
print(a.get_quadratic_part(3, 1.73))
print(a.get_l_central(3, 0))
