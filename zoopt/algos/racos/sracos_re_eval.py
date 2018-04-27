
#!/usr/bin/env python
# coding=utf-8

"""
The class SSRacos represents SSRacos algorithm. It's inherited from SRacos.


Author:
    Xionghui Chen
"""

import time
import numpy
from zoopt.solution import Solution
from zoopt.algos.racos.racos_classification import RacosClassification
from zoopt.algos.racos.racos_common import RacosCommon
from zoopt.utils.zoo_global import gl
from zoopt.utils.tool_function import ToolFunction
from zoopt.algos.racos.sracos import SRacos


class SRacosReEval(SRacos):

    def __init__(self, objective, parameter, strategy='WR', ub=1):
        self.strategy = strategy
        SRacos.__init__(self, objective, parameter, ub)
        return

    # SRacos's optimization function
    # Default strategy is WR(worst replace)
    # Default uncertain_bits is 1, but actually ub will be set either by user
    # or by RacosOptimization automatically.
    def opt(self):
        self.clear()
        self.init_attribute()
        self.i = 0
        iteration_num = self._parameter.get_budget() - self._parameter.get_train_size()
        iteration_num = iteration_num
        time_log1 = time.time()
        max_distinct_repeat_times = 100
        current_not_distinct_times = 0
        dont_early_stop = False
        non_update_allowed = self._parameter.get_non_update_allowed()
        update_precision = self._parameter.get_max_stay_precision()
        last_best = None
        non_update_times = 0

        while self.i < iteration_num:
            if gl.rand.random() < self._parameter.get_probability():
                classifier = RacosClassification(
                    self._objective.get_dim(), self._positive_data, self._negative_data, self.ub)
                classifier.mixed_classification()
                solution, distinct_flag = self.distinct_sample_classifier(
                    classifier, True, self._parameter.get_train_size())
            else:
                solution, distinct_flag = self.distinct_sample(
                    self._objective.get_dim())
            # panic stop
            if solution is None:
                ToolFunction.log(" [break loop] solution is None")
                return self.get_best_solution()
            if distinct_flag is False:
                current_not_distinct_times += 1
                if current_not_distinct_times >= max_distinct_repeat_times:
                    ToolFunction.log(
                        "[break loop] distinct_flag is false too much times")
                    return self.get_best_solution()
                else:
                    continue
            # evaluate the solution
            self._objective.eval(solution)
            bad_ele = self.replace(self._positive_data, solution, 'pos')
            self.replace(self._negative_data, bad_ele, 'neg', self.strategy)
            self._best_solution = self._positive_data[0]

            if last_best is not None and last_best - self._best_solution.get_value() < update_precision:
                non_update_times += 1
                if non_update_times >= non_update_allowed:
                    ToolFunction.log(
                        "[break loop] because stay longer than max_stay_times, break loop")
                    return self._best_solution
            else:
                non_update_times = 0
            last_best = self._best_solution.get_value()

            if self.i == 4:
                time_log2 = time.time()
                expected_time = (self._parameter.get_budget(
                ) - self._parameter.get_train_size()) * (time_log2 - time_log1) / 5
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
                    return self.get_best_solution()
            # early stop
            if self._parameter.early_stop is not None and not dont_early_stop:
                if solution.get_value() < self._objective.return_before * 0.9:
                    dont_early_stop = True
                elif self.i > self._parameter.early_stop:
                    ToolFunction.log(
                        '[break loop] early stop for too low value.')
                    return self.get_best_solution()
                ToolFunction.log('[early stop warning ]: current iter %s , target %s. current value %s. target value %s'% (
                    self.i, self._parameter.early_stop, solution.get_value(), self._objective.return_before * 0.9))

            # terminal_value check
            if self._parameter.get_terminal_value() is not None:
                solution = self.get_best_solution()
                if solution is not None and solution.get_value() <= self._parameter.get_terminal_value():
                    ToolFunction.log('terminal function value reached')
                    return self.get_best_solution()

            if self.i % self._parameter.update_q_frequent == 0:
                self._objective.update_q_func()

            if self.i % self._parameter.re_eval_frequent == 0:
                self._objective.copy_q_value_func()
                for solu in self._positive_data:
                    self._objective.re_eval_func(solu)
                for solu in self._negative_data:
                    self._objective.re_eval_func(solu)
                data = self._positive_data + self._negative_data
                self.selection(data)
                for solu in self._positive_data:
                    self._objective.test_explore_actor_func(solu)

            ToolFunction.log('[iter log] i %s, non_update_times %s, non_update_allowed %s ' %(
                self.i, non_update_times, non_update_allowed))
            self.i += 1
        return self.get_best_solution()

    def get_best_solution(self):
        return self._positive_data[0]

    def sort_solution_list(self, solution_list, key=lambda x: x.get_value()):
        return sorted(solution_list, key=key)
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


    def _is_worest(self, solution):
        return self._positive_data[-1].get_value() <= solution.get_value()

    def add_custom_solution(self, solution):
        if solution is not None:
            bad_ele = self.replace(self._positive_data, solution, 'pos')
            self.replace(self._negative_data, bad_ele, 'neg', self.strategy)
            self._best_solution = self._positive_data[0]