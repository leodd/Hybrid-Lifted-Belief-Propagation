from itertools import product


class BP:
    # loopy belief propagation

    def __init__(self, g=None):
        self.g = g
        self.message = None
        self.points = dict()

    def init_points(self):
        points = dict()
        for rv in self.g.rvs:
            if rv.value is None:
                points[rv] = rv.domain.values
            else:
                points[rv] = (rv.value,)
        return points

    def init_message(self):
        self.message = dict()

        for f in self.g.factors:
            for rv in f.nb:
                self.message[(f, rv)] = {k: 1 for k in self.points[rv]}

        for rv in self.g.rvs:
            for f in rv.nb:
                self.message[(rv, f)] = dict()

    def message_rv_to_f(self, rv, f):
        for val in self.points[rv]:
            temp = 1
            for nb in rv.nb:
                if nb != f:
                    temp = temp * self.message[(nb, rv)][val]
            self.message[(rv, f)][val] = temp

    def message_f_to_rv(self, f, rv):
        for val in self.points[rv]:
            temp = 0
            param = []
            for nb in f.nb:
                if nb == rv:
                    param.append((val,))
                else:
                    param.append(self.points[nb])
            for x in product(*param):
                income_message = 1
                for i in range(len(f.nb)):
                    if f.nb[i] != rv:
                        income_message = income_message * self.message[(f.nb[i], f)][x[i]]
                temp = temp + f.potential.get(x) * income_message
            self.message[(f, rv)][val] = temp

    def run(self, iteration=10):
        self.points = self.init_points()
        self.init_message()
        for i in range(iteration):
            for rv in self.g.rvs:
                for f in rv.nb:
                    self.message_rv_to_f(rv, f)

            for f in self.g.factors:
                for rv in f.nb:
                    self.message_f_to_rv(f, rv)

    def belief(self, rv):
        belief = dict()
        for val in self.points[rv]:
            temp = 1
            for nb in rv.nb:
                temp = temp * self.message[(nb, rv)][val]
            belief[val] = temp
        self.normalize_message(belief)
        return belief

    @staticmethod
    def normalize_message(message):
        z = 0
        for k, v in message.items():
            z = z + v
        for k, v in message.items():
            message[k] = v / z
