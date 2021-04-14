
import copy
import sys
from zoopt.utils.tool_function import ToolFunction
import numpy as np
"""
The class RacosCommon contains common attributes and methods between Racos and SRacos.
Class Racos and SRacos both inherit from RacosCommon.

Author:
    Yuren Liu
"""


class RacosCommon:

    def __init__(self, objective, parameter, ub):
        self._parameter = parameter
        self._objective = objective
        self.ub = ub
        # Solution set
        # Random sampled solutions construct self._data
        self._data = []
        # self._positive_data are best-positive_size solutions set
        self._positive_data = []
        # self._negative_data are the other solutions
        self._negative_data = []
        # Solution
        self._best_solution = None
        self._possible_solution_list = []
        return

    # Clear RacosCommon
    def clear(self):
        # Solution
        self._data = []
        self._positive_data = []
        self._negative_data = []
        # value
        self._best_solution = None

    # Construct self._data, self._positive_data, self._negative_data
    def init_attribute(self):
        # check if the initial solutions have been set
        data_temp = self._parameter.get_init_samples()
        i = 0
        if data_temp != None and self._best_solution == None:
            for j in range(len(data_temp)):
                x = self._objective.construct_solution(data_temp[j])
                self._objective.eval(x)
                self._data.append(x)
                ToolFunction.log(" init solution %s, eval %s" % (i, x.get_value()))
                # if j == 0:
                    # ret = self._objective.tester.check_and_test(self._objective.agent, always=True)
                    # ToolFunction.log(" baseline ret %s" % (ret))
                    # x.set_value(-1 * ret)
                i += 1
        # otherwise generate random solutions
        iteration_num = self._parameter.get_train_size()
        while i < iteration_num:
            # distinct_flag: True means sample is distinct(can be use),
            # False means sample is distinct, you should sample again.
            x, distinct_flag = self.distinct_sample_from_set(self._objective.get_dim(), self._data,
                                                             data_num=iteration_num)
            # panic stop
            if x is None:
                break
            if distinct_flag:
                self._objective.eval(x)
                self._data.append(x)
                ToolFunction.log(" init solution. random %s, eval %s" % (i, x.get_value()))
                i += 1
        self.selection(self._data)
        return

    # Sort self._data
    # Choose first-train_size solutions as the new self._data
    # Choose first-positive_size solutions as self._positive_data
    # Choose [positive_size, train_size) (Include the begin, not include the
    # end) solutions as self._negative_data
    def selection(self, data):
        new_data = sorted(data, key=lambda x: x.get_value())
        self._data = new_data[0:self._parameter.get_train_size()]
        self._positive_data = new_data[0: self._parameter.get_positive_size()]
        self._negative_data = new_data[
            self._parameter.get_positive_size():self._parameter.get_train_size()]
        self._best_solution = self._positive_data[0]
        # set origin solution
        self._objective.origin_solution = np.array(new_data[0].get_x())
        return

    # Distinct sample from dim, return a solution
    def distinct_sample(self, dim, check_distinct=True, data_num=0):
        objective = self._objective
        x = objective.construct_solution(dim.rand_sample())
        times = 1
        distinct_flag = True
        if check_distinct is True:
            while self.is_distinct(self._positive_data, x) is False or \
                    self.is_distinct(self._negative_data, x) is False:
                x = objective.construct_solution(dim.rand_sample())
                times += 1
                if times % 10 == 0:
                    limited, number = dim.limited_space()
                    if limited is True:
                        if number <= data_num:
                            ToolFunction.log(
                                'racos_common.py: WARNING -- sample space has been fully enumerated. Stop early')
                            return None, None
                    if times > 100:
                        distinct_flag = False
                        break
        return x, distinct_flag

    def distinct_sample_from_set(self, dim, set, check_distinct=True, data_num=0):
        objective = self._objective
        x = objective.construct_solution(dim.rand_sample())
        times = 1
        distinct_flag = True
        if check_distinct is True:
            while self.is_distinct(set, x) is False:
                x = objective.construct_solution(dim.rand_sample())
                times += 1
                if times % 10 == 0:
                    limited, number = dim.limited_space()
                    if limited is True:
                        if number <= data_num:
                            ToolFunction.log(
                                'racos_common.py: WARNING -- sample space has been fully enumerated. Stop early')
                            return None, None
                    if times > 100:
                        distinct_flag = False
                        break
        return x, distinct_flag

    # Distinct sample from a classifier, return a solution
    # if check_distinct is False, you don't need to sample distinctly
    def distinct_sample_classifier(self, classifier, check_distinct=True, data_num=0):
        x = classifier.rand_sample()
        ins = self._objective.construct_solution(x)
        times = 1
        distinct_flag = True
        if check_distinct is True:
            while self.is_distinct(self._positive_data, ins) is False or \
                    self.is_distinct(self._negative_data, ins) is False:
                x = classifier.rand_sample()
                ins = self._objective.construct_solution(x)
                times += 1
                if times % 10 == 0:
                    if times == 10:
                        space = classifier.get_sample_space()
                    limited, number = space.limited_space()
                    if limited is True:
                        if number <= data_num:
                            ToolFunction.log(
                                'racos_common: WARNING -- sample space has been fully enumerated. Stop early')
                            return None, None
                    if times > 100:
                        distinct_flag = False
                        break
        return ins, distinct_flag

    # Append setb to seta, deepcopy
    @staticmethod
    def extend(seta, setb):
        result = copy.deepcopy(seta)
        for x in setb:
            result.append(copy.deepcopy(x))
        return result

    # Check if x is distinct from each solution in seta
    # return False if there exists a solution the same as x,
    # otherwise return True
    @staticmethod
    def is_distinct(seta, x):
        for ins in seta:
            if x.is_equal(ins):
                return False
        return True

    def set_parameters(self, parameter):
        self._parameter = parameter
        return

    def get_parameters(self):
        return self._parameter

    def set_objective(self, objective):
        self._objective = objective
        return

    def get_objective(self):
        return self._objective

    # For debugging
    def print_positive_data(self):
        ToolFunction.log('------print positive_data------')
        ToolFunction.log('the size of positive_data is: %d' %
                         (len(self._positive_data)))
        for x in self._positive_data:
            x.print_solution()

    def print_negative_data(self):
        ToolFunction.log('------print negative_data------')
        ToolFunction.log('the size of negative_data is: %d' %
                         (len(self._negative_data)))
        for x in self._negative_data:
            x.print_solution()

    def print_data(self):
        ToolFunction.log('------print b------')
        ToolFunction.log('the size of b is: %d' % (len(self._data)))
        for x in self._data:
            x.print_solution()
