
import numpy as np
import time
import numpy
from zoopt.solution import Solution
from zoopt.algos.racos.racos_classification import RacosClassification
from zoopt.algos.racos.racos_common import RacosCommon
from zoopt.utils.zoo_global import gl
from zoopt.utils.tool_function import ToolFunction
from random import shuffle
"""
The class SRacos represents SRacos algorithm. It's inherited from RacosCommon.

Author:
    Yuren Liu
"""

class SRacos(RacosCommon):

    def __init__(self, objective, parameter, strategy='WR', ub=1):
        self.strategy = strategy
        RacosCommon.__init__(self, objective, parameter, ub)
        return

    # SRacos's optimization function
    # Default strategy is WR(worst replace)
    # Default uncertain_bits is 1, but actually ub will be set either by user
    # or by RacosOptimization automatically.
    def opt(self):
        self.clear()
        self.init_attribute()
        i = 0
        iteration_num = self._parameter.get_budget() - self._parameter.get_train_size()
        time_log1 = time.time()
        max_distinct_repeat_times = 100
        current_not_distinct_times = 0
        last_best = None
        max_stay_times = self._parameter.get_max_stay()
        current_stay_times = 0
        while i < iteration_num:
            if gl.rand.random() < self._parameter.get_probability():
                classifier = RacosClassification(
                    self._objective.get_dim(), self._positive_data, self._negative_data, ub)
                classifier.mixed_classification()
                solution, distinct_flag = self.distinct_sample_classifier(
                    classifier, True, self._parameter.get_train_size())
            else:
                solution, distinct_flag = self.distinct_sample(
                    self._objective.get_dim())
            # panic stop
            if solution is None:
                ToolFunction.log(" [break loop] because solution is None")
                return self._best_solution
            if distinct_flag is False:
                current_not_distinct_times += 1
                if current_not_distinct_times >= max_distinct_repeat_times:
                    ToolFunction.log(
                        "[break loop] because distinct_flag is false too much times")
                    return self._best_solution
                else:
                    continue
            # evaluate the solution
            self._objective.eval(solution)
            bad_ele = self.replace(self._positive_data, solution, 'pos')
            self.replace(self._negative_data, bad_ele, 'neg', self.strategy)
            self._best_solution = self._positive_data[0]

            # if best_solution stay longer than max_stay_times, break loop
            if last_best is not None and last_best - self._best_solution.get_value() < parameter.get_max_stay_precision():
                current_stay_times += 1
                if current_stay_times >= max_stay_times:
                    ToolFunction.log(
                        "[break loop] because stay longer than max_stay_times, break loop")
                    return self._best_solution
            else:
                current_stay_times = 0
            last_best = self._best_solution.get_value()
            if i == 4:
                time_log2 = time.time()
                expected_time = (self._parameter.get_budget() - self._parameter.get_train_size()) * \
                                (time_log2 - time_log1) / 5
                if self._parameter.get_time_budget() is not None:
                    expected_time = min(
                        expected_time, self._parameter.get_time_budget())
                if expected_time > 5:
                    m, s = divmod(expected_time, 60)
                    h, m = divmod(m, 60)
                    ToolFunction.log(
                        'expected remaining running time: %02d:%02d:%02d' % (h, m, s))
            # time budget check
            if self._parameter.get_time_budget() is not None:
                if (time.time() - time_log1) >= self._parameter.get_time_budget():
                    ToolFunction.log('time_budget runs out')
                    return self._best_solution
            # terminal_value check
            if self._parameter.get_terminal_value() is not None:
                if self._best_solution.get_value() <= self._parameter.get_terminal_value():
                    ToolFunction.log('terminal function value reached')
                    return self._best_solution
            i += 1
        return self._best_solution

    def replace(self, iset, x, iset_type, strategy='WR'):

        if self.get_parameters().distance_replace and strategy != 'none':
            from baselines import logger
            _, dis = self.get_parameters().replace_func(x)
            avg_dis_norm = []
            found = False
            update = False
            random_index = list(range(len(iset)))
            shuffle(random_index)
            for index in random_index:
                replace, norm = self.get_parameters().replace_func(iset[index])
                avg_dis_norm.append(norm)
                ToolFunction.log('[distance_replace %s][%s] dis %.5f, norm %.5f. value %.5f, set_value %.5f' %(iset_type, index,
                                                                                                       dis, norm,
                                                                                                       x.get_value(),
                                                                                                       iset[index].get_value()))
                logger.record_tabular('distance', norm)
                logger.dump_tabular()
                if replace and dis < norm:
                    # if iset[index].get_value() < x.get_value() and iset_type == 'pos':
                    #     update = True
                    if index == 0 and iset_type == 'pos':
                        self.last_best = self._positive_data[1]
                    sol = iset[index]
                    iset[index] = x
                    found = True
                    ToolFunction.log("[replace success]")
                    break
            logger.record_tabular('distance/rep_freq', np.mean(self._parameter.replace_frequent))
            if found:
                self._parameter.replace_frequent.append(1)
                return sol
            else:
                self._parameter.replace_frequent.append(0)
                ToolFunction.log('distance solution not found')
        if strategy == 'WR':
            return self.strategy_wr(iset, x, iset_type)
        elif strategy == 'RR':
            return self.strategy_rr(iset, x)
        elif strategy == 'LM':
            best_sol = min(iset, key=lambda x: x.get_value())
            return self.strategy_lm(iset, best_sol, x)
        elif strategy == 'RS':
            self.replace_n(iset, x)
        elif strategy == 'none':
            ToolFunction.log('dont replace')
        else:
            raise NotImplementedError

    def replace_n(self, iset, x):
        len_iset = len(iset)
        replace_index = gl.rand.randint(0, len_iset - 1)
        replace_ele = iset[replace_index]
        data = self._positive_data + self._negative_data
        x, distinct_flag = self.distinct_sample_from_set(self._objective.get_dim(), data,
                                                         data_num=1)
        x.set_value(999999)
        iset[replace_index] = x
        return replace_ele

    # Find first element larger than x
    def binary_search(self, iset, x, begin, end):
        x_value = x.get_value()
        if x_value <= iset[begin].get_value():
            return begin
        if x_value >= iset[end].get_value():
            return end + 1
        if end == begin + 1:
            return end
        mid = (begin + end) // 2
        if x_value <= iset[mid].get_value():
            return self.binary_search(iset, x, begin, mid)
        else:
            return self.binary_search(iset, x, mid, end)

    # Worst replace
    def strategy_wr(self, iset, x, iset_type):
        if iset_type == 'pos':
            index = self.binary_search(iset, x, 0, len(iset) - 1)
            iset.insert(index, x)
            worst_ele = iset.pop()
        else:
            worst_ele, worst_index = Solution.find_maximum(iset)
            if worst_ele.get_value() > x.get_value():
                if len(iset) < self._parameter.get_max_neg_size():
                    iset.append(x)
                else:
                    iset[worst_index] = x
            else:
                if len(iset) < self._parameter.get_max_neg_size():
                    iset.append(x)
                worst_ele = x
        return worst_ele

    # Random replace
    def strategy_rr(self, iset, x):
        len_iset = len(iset)
        replace_index = gl.rand.randint(0, len_iset - 1)
        replace_ele = iset[replace_index]
        iset[replace_index] = x
        return replace_ele

    # replace the farthest solution from best_sol
    def strategy_lm(self, iset, best_sol, x):
        farthest_dis = 0
        farthest_index = 0
        for i in range(len(iset)):
            dis = self.distance(iset[i].get_x(), best_sol.get_x())
            if dis > farthest_dis:
                farthest_dis = dis
                farthest_index = i
        farthest_ele = iset[farthest_index]
        iset[farthest_index] = x
        return farthest_ele

    @staticmethod
    def distance(x, y):
        dis = 0
        for i in range(len(x)):
            dis += (x[i] - y[i])**2
        return numpy.sqrt(dis)
