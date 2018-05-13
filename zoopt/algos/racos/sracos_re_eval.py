
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
        SRacos.__init__(self, objective, parameter, strategy, ub)
        self.solution_counter = 0
        self.init_data = self._parameter.get_init_samples()
        self.current_not_distinct_times = 0
        self.current_solution = None
        self.non_update_times = 0
        self.last_best = None
        self.in_re_eval_mode = False
        self.dont_early_stop = False
        self.need_restart = False
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
        non_update_times = 0
        last_best = None
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
            self.non_update_times = non_update_times
            if self.last_best  is not None and self.last_best .get_value() - self._best_solution.get_value() <= update_precision and (self._parameter.get_terminal_value() is None or self._best_solution.get_value() > self._parameter.get_terminal_value()):
                non_update_times += 1
                if non_update_times >= non_update_allowed:
                    ToolFunction.log(
                        "[break loop] because stay longer than max_stay_times, break loop")
                    return self._best_solution
            else:
                non_update_times = int(non_update_times / 2)
            self.last_best = self._best_solution
            self.update_ub()
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
            # if self._parameter.get_terminal_value() is not None:
            #     solution = self.get_best_solution()
            #     if solution is not None and solution.get_value() <= self._parameter.get_terminal_value():
            #         ToolFunction.log('terminal function value reached')
            #         return self.get_best_solution()

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

    def get_next_re_eval_solution(self):
        from SLBDAO.explore_agent import ExploreModule
        while self.re_eval_index < len(self._positive_data):
            solution = self._positive_data[self.re_eval_index]
            attach = solution.get_post_attach()
            if ExploreModule.VAR_CONTENT in attach:
                ToolFunction.log("[construct explore actor] use re-eval solution")
                self.re_eval_index += 1
                return solution
            else:
                ToolFunction.log(
                    "[WARNING][construct explore actor] solution %s, without var content. select next." % solution.get_x())
                self.re_eval_index += 1
        self.in_re_eval_mode = False
        solution = self.generate_solution()
        return solution
        # algorithm.end_re_eval_solution() # TODO 这里的sort只能改成每次进行了

        # return solution, False

    def generate_solution(self):
        if self.solution_counter < len(self.init_data):
            ToolFunction.log(" [generate_solution] init_data")
            x = self._objective.construct_solution(self.init_data[self.solution_counter])
        elif self.solution_counter < self._parameter.get_train_size():
            if self._parameter.evaluate_negative_data:
                x, distinct_flag = self.distinct_sample_from_set(self._objective.get_dim(), self._data,
                                                                 data_num=self._parameter.get_train_size())
                if x is None:
                    self.solution_counter = self._parameter.get_train_size()
                    return self.generate_solution()
            else:
                if self.solution_counter < self._parameter.get_positive_size():
                    ToolFunction.log(" [generate_solution] set positive data")
                    x, distinct_flag = self.distinct_sample_from_set(self._objective.get_dim(), self._data,
                                                                     data_num=self._parameter.get_train_size())
                    if x is None:
                        self.solution_counter = self._parameter.get_train_size()
                        return self.generate_solution()
                else:
                    ToolFunction.log(" [generate_solution] set negative data")
                    for i in range(self.get_parameters().get_negative_size()):
                        x, distinct_flag = self.distinct_sample_from_set(self._objective.get_dim(), self._data,
                                                                         data_num=self._parameter.get_train_size())
                        x.set_value(999999)
                        self._data.append(x)
                        self.solution_counter += 1
                    self.selection(self._data)
                    self.print_all_solution()
                    return self.generate_solution()
        else:
            ToolFunction.log(" [generate_solution] generate by classfication")
            if gl.rand.random() < self._parameter.get_probability():
                classifier = RacosClassification(
                    self._objective.get_dim(), self._positive_data, self._negative_data, self.ub)
                classifier.mixed_classification()
                x, distinct_flag = self.distinct_sample_classifier(
                    classifier, True, self._parameter.get_train_size())
            else:
                x, distinct_flag = self.distinct_sample(
                    self._objective.get_dim())
            # panic stop
            if x is None:
                ToolFunction.log(" [break loop] solution is None")
                return self.get_best_solution()
            if distinct_flag is False:
                self.current_not_distinct_times += 1
                if self.current_not_distinct_times >= 100:
                    ToolFunction.log(
                        "[break loop] distinct_flag is false too much times")
                    return self.get_best_solution()
                else:
                    return self.generate_solution()
        # self.current_solution = x
        return x

    def update_racos_stats(self, solution_eval, attach, feed_solution):
        assert isinstance(feed_solution, Solution)
        feed_solution.set_value(solution_eval)
        feed_solution.set_post_attach(attach)
        self.solution_counter += 1

        if self.solution_counter < self._parameter.get_train_size():
            # do nothing.
            self._data.append(feed_solution)
        elif self.solution_counter == self._parameter.get_train_size():
            self._data.append(feed_solution)
            self.selection(self._data)
        else:
            bad_ele = self.replace(self._positive_data, feed_solution, 'pos')
            self.replace(self._negative_data, bad_ele, 'neg', self.strategy)
            self._best_solution = self._positive_data[0]
            self.update_ub()
            if self.last_best is not None and self.last_best.get_value() - self._best_solution.get_value() <= self._parameter.get_max_stay_precision()\
                    and (self._parameter.get_terminal_value() is None or self._best_solution.get_value() > self._parameter.get_terminal_value()):
                self.non_update_times += 1
                if self.non_update_times >= self._parameter.get_non_update_allowed():
                    ToolFunction.log(
                        "[break loop] because stay longer than max_stay_times, break loop")
                    self.need_restart = True
                    return self._best_solution
            else:
                self.non_update_times = int(self.non_update_times / 2)
                self.last_best = self._best_solution

            # early stop
            if self._parameter.early_stop is not None and not self.dont_early_stop:
                if feed_solution.get_value() < self._objective.return_before * 0.9:
                    self.dont_early_stop = True
                elif self.solution_counter - self.get_parameters().get_train_size() > self._parameter.early_stop:
                    ToolFunction.log(
                        '[break loop] early stop for too low value.')
                    self.need_restart = True
                    return self.get_best_solution()
                ToolFunction.log(
                    '[early stop warning ]: current iter %s , target %s. current value %s. target value %s' % (
                        self.solution_counter - self.get_parameters().get_train_size(),
                        self._parameter.early_stop, feed_solution.get_value(), self._objective.return_before * 0.9))
            ToolFunction.log('[iter log] i %s, non_update_times %s, non_update_allowed %s ' % (
                self.solution_counter - self.get_parameters().get_train_size(),
                self.non_update_times, self._parameter.get_non_update_allowed()))
        return None

    def _is_worest(self, solution):
        return self._positive_data[-1].get_value() <= solution.get_value()

    def add_custom_solution(self, solution):
        found = False
        if solution is not None:
            if self.solution_counter < self._parameter.get_train_size():
                ToolFunction.log("[add_custom_solution] init not complete, just do nothing.")
                return
            for index, sol in enumerate(self._positive_data):
                if sol.is_the_same(solution):
                    ToolFunction.log("[add_custom_solution] solution in positive data, value %s" % sol.get_value())
                    self._positive_data[index] = solution
                    self._positive_data = sorted(self._positive_data, key=lambda x: x.get_value())
                    self._best_solution = self._positive_data[0]
                    found = True
                    break
            for index, sol in enumerate(self._negative_data):
                if sol.is_the_same(solution):
                    ToolFunction.log("[add_custom_solution] solution in negative data, value %s" % sol.get_value())
                    bad_ele = self.replace(self._positive_data, solution, 'pos')
                    self._negative_data[index] = bad_ele
                    found = True
                    break
            if not found:
                ToolFunction.log("[add_custom_solution] new solution. value is %s" % solution.get_value())
                bad_ele = self.replace(self._positive_data, solution, 'pos')
                self.replace(self._negative_data, bad_ele, 'neg', self.strategy)
                ToolFunction.log("[replace solution]")
                bad_ele.print_solution()
            self._positive_data = sorted(self._positive_data, key=lambda x: x.get_value())
            self._best_solution = self._positive_data[0]
            if self.solution_counter % 5 == 0:
                self.print_all_solution()

    def set_re_eval_solution(self, solution):
        self.add_custom_solution(solution)
        tester = self.get_objective().tester
        tester.add_custom_record('re-eval-point', x=tester.time_step_holder.get_time(),
                                 y=abs(solution.get_value()),
                                 x_name='time step', y_name='re-eval-point')

    def end_re_eval_solution(self):
        self._positive_data = sorted(self._positive_data, key=lambda x: x.get_value())
        ToolFunction.log("---print positive solution----")
        for i in range(len(self._positive_data)):
            ToolFunction.log("i : %s, value %s " % (i, self._positive_data[i].get_value()))
        ToolFunction.log("----end----")

    def print_all_solution(self):
        ToolFunction.log("----print positive solution----")
        for index, sol in enumerate(self._positive_data):
            ToolFunction.log("[%s]" % str(index))
            sol.print_solution()
        ToolFunction.log("----print negative solution----")
        for index, sol in enumerate(self._negative_data):
            ToolFunction.log("[%s]" % str(index))
            sol.print_solution()

    def re_eval_positive_solution(self):
        for solu in self._positive_data:
            ToolFunction.log("solution info: eval %s" %solu.get_value())
            self._objective.eval(solu)
            tester = self.get_objective().tester
            tester.add_custom_record('re-eval-point',x=tester.time_step_holder.get_time(),
                                          y=solu.get_value(),
                                          x_name='time step', y_name='re-eval-point')
        self._positive_data = sorted(self._positive_data, key=lambda x: x.get_value())
        self._best_solution = self._positive_data[0]
        # for i in range(5):
        #     ToolFunction.log("random sample solution.")
        #     solution, distinct_flag = self.distinct_sample(self._objective.get_dim())
        #     if distinct_flag:
        #         self._objective.eval(solution)
        #         bad_ele = self.replace(self._positive_data, solution, 'pos')
        #         self.replace(self._negative_data, bad_ele, 'neg', 'RR')
        ToolFunction.log("---print positive solution----")
        for i in range(len(self._positive_data)):
            ToolFunction.log("i : %s, value %s " %(i, self._positive_data[i].get_value()))
        # ToolFunction.log("---print negative solution----")
        # for i in range(len(self._negative_data)):
        #     ToolFunction.log("i : %s, value %s " %(i, self._negative_data[i].get_value()))
        ToolFunction.log("----end----")

    def update_ub(self):
        if not self.get_parameters().update_uncertain_bit:
            ToolFunction.log("[update ub] don't update: %s" % self.ub)
            return
        if self.non_update_times is None:
            return
        else:
            self.ub = int(self.non_update_times / self._parameter.get_non_update_allowed() * 5) + 1
        ToolFunction.log("[update ub] : %s" % self.ub)

    def re_test_solution(self, test_func):
        diff = []
        for solu in self._positive_data:
            value_before = solu.get_value()
            value_after = test_func(solu)
            if value_after is None:
                continue
            else:
                value_after = value_after * -1
            ToolFunction.log("[re_test_solution] before %s, after %s. " % (value_before, value_after))
            diff.append(abs(value_after - value_before))

        for solu in self._negative_data:
            value_before = solu.get_value()
            value_after = test_func(solu)
            if value_after is None:
                continue
            else:
                value_after = value_after * -1
            ToolFunction.log("[re_test_solution] before %s, after %s. " % (value_before, value_after))
            diff.append(abs(value_after - value_before))
        import numpy as np
        diff_mean = np.array(diff).mean()
        tester = self.get_objective().tester
        tester.add_custom_record('avg_diff_solution', x=tester.time_step_holder.get_time(),
                                      y=diff_mean,
                                      x_name='time step', y_name='avg_diff_solution')
