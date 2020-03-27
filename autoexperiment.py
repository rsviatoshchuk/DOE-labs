import numpy
from scipy.stats import t, f


class Experiment:
    def __init__(self):
        self.normalized_matrix = None

    def get_normalized_matrix(self, num_of_factors):
        def conv(str_el):
            if str_el == "0":
                str_el = "-1"
            return int(str_el)
        vconv = numpy.vectorize(conv)

        num_of_levels = 2  # for future
        form = "{{0:0{}b}}".format(num_of_factors)  # formatting for getting 0001 instead 1
        str_array = numpy.array([list(form.format(i)[::-1]) for i in range(num_of_levels ** num_of_factors)])
        return vconv(str_array)


a = Experiment()
print(a.get_normalized_matrix(3))
