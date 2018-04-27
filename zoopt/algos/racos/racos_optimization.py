from zoopt.algos.racos.ssracos import SSRacos
from zoopt.algos.racos.sracos_re_eval import SRacosReEval
from zoopt.algos.racos.sracos import SRacos
from zoopt.algos.racos.racos import Racos

"""
The class RacosOptimization will contains best_solution and optimization algorithm(Racos or SRacos)

Author:
    Yuren Liu
"""


class RacosOptimization:

    def __init__(self, objective, parameter, strategy='WR'):
        self.__best_solution = None
        self.__algorithm = None
        ub = parameter.get_uncertain_bits()

        self.objective = objective
        self.parameter = parameter
        self.strategy = strategy
        if ub is None:
            ub = self.set_ub(objective)
        self.ub = ub

        if parameter.get_sequential():
            if not parameter.get_suppressioin():
                self.__algorithm = SRacos(self.objective, self.parameter, self.strategy, self.ub)
            elif parameter.use_re_eval:
                self.__algorithm = SRacosReEval(self.objective, self.parameter, self.strategy, self.ub)
            else:
                self.__algorithm = SSRacos(self.objective, self.parameter, self.strategy, self.ub)
        else:
            self.__algorithm = Racos(self.objective, self.parameter, self.ub)

    def get_algorithm(self):
        return self.__algorithm


    # General optimization function, it will choose optimization algorithm according to parameter.get_sequential()
    # Default replace strategy is 'WR'
    # If user hasn't define uncertain_bits in parameter, set_ub() will set uncertain_bits automatically according to dim
    # in objective
    def opt(self):
        self.__best_solution = self.__algorithm.opt()
        return self.__best_solution


    def get_best_sol(self): 
        return self.__best_solution

    @staticmethod
    # Set uncertain_bits
    def set_ub(objective):
        dim = objective.get_dim()
        dim_size = dim.get_size()
        is_discrete = dim.is_discrete()
        if is_discrete is False:
            if dim_size <= 100:
                ub = 1
            elif dim_size <= 1000:
                ub = 2
            else:
                ub = 3
        else:
            if dim_size <= 10:
                ub = 1
            elif dim_size <= 50:
                ub = 2
            elif dim_size <= 100:
                ub = 3
            elif dim_size <= 1000:
                ub = 4
            else:
                ub = 5
        return ub
