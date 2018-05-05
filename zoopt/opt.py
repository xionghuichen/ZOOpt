
from zoopt.algos.paretoopt.ParetoOptimization import ParetoOptimization
from zoopt.algos.racos.racos_optimization import RacosOptimization
from zoopt.utils.zoo_global import gl
from zoopt.utils.tool_function import ToolFunction

"""
The class Opt is the main entrance of using zoopt: Opt.min(objective, parameter)

Author:
    Yuren Liu
"""


class Opt:

    def __init__(self, objective, parameter, strategy='WR'):
        Opt.set_global(parameter)
        constraint = objective.get_constraint()
        algorithm = parameter.get_algorithm()
        self.objective = objective
        self.parameter = parameter
        if algorithm:
            algorithm = algorithm.lower()
        result = None
        if constraint is not None and ((algorithm is None) or (algorithm == "poss")):
            self.optimizer = ParetoOptimization()
        elif constraint is None and ((algorithm is None) or (algorithm == "racos") or (algorithm == "sracos")) or (
                algorithm == "ssracos"):
            self.optimizer = RacosOptimization(self.objective, self.parameter, strategy)
        else:
            ToolFunction.log(
                "opt.py: No proper algorithm found for %s" % algorithm)

    def min(self):
        return self.optimizer.opt()


    @staticmethod
    def set_global(parameter):
        precision = parameter.get_precision()
        if precision:
            gl.set_precision(precision)
