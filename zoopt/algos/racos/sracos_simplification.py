
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
from collections import deque
import numpy as np
from RLA.easy_log import logger


class SRacosSimple(SRacos):

    def __init__(self, objective, parameter, strategy='WR', ub=1):
        self.strategy = strategy
        SRacos.__init__(self, objective, parameter, strategy, ub)
        self.init_ub = ub
        self.solution_counter = 0
        self.init_data = self._parameter.get_init_samples()
        self.current_not_distinct_times = 0
        self.current_solution = None
        self.non_update_times = 0
        self.non_update_times_cumulative = 0
        self.last_times_update_success = False
        self.last_best = None
        self.in_re_eval_mode = False
        self.dont_early_stop = False
        self.need_restart = False
        self.finish_init = False
        self.must_select_index = self._parameter.must_select_index
        self.history_best_value = deque(maxlen=50)
        self.generate_failed = False
        self.baseline_data = []
        return

    def get_best_solution(self):
        self.print_all_solution(record=True, name='best_solution')
        return self._positive_data[0]

    def sort_solution_list(self, solution_list, key=lambda x: x.get_value()):
        return sorted(solution_list, key=key)
    # Find first element larger than x

    def binary_search(self, iset, x, begin, end):
        if len(iset) == 0:
            return 0
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
        from SLBDAO.explore_agent import RacosHolder
        while self.re_eval_index < len(self.re_eval_solution_list):
            solution = self.re_eval_solution_list[self.re_eval_index]
            attach = solution.get_post_attach()
            if RacosHolder.VAR_CONTENT in attach: # this condition will skip baseline solution
                ToolFunction.log("[construct explore actor] use re-eval solution")
                self.re_eval_index += 1
                idx = self.solution_counter
                return solution, idx
            else:
                ToolFunction.log(
                    "[WARNING][construct explore actor] solution %s, without var content. select next." % solution.get_x())
                self.re_eval_index += 1
        self.in_re_eval_mode = False
        # 这个mode只是标记RACOS是否继续产生re-eval的样本，而不是记录re-eval是否已经结束了，因为re-eval的结果还没传过来
        self.end_re_eval_solution()
        solution, idx = self.generate_solution()
        return solution, idx

    def generate_solution(self):
        assert not self.need_restart

        if self.solution_counter < len(self.init_data):
            ToolFunction.log(" [generate_solution] init_data . counter %s" % self.solution_counter)
            x = self._objective.construct_solution(self.init_data[self.solution_counter])
        elif self.solution_counter <= self._parameter.get_train_size():
            if self._parameter.evaluate_negative_data:
                ToolFunction.log(" [generate_solution] init_data random sampling. counter %s" % self.solution_counter)
                x, distinct_flag = self.distinct_sample_from_set(self._objective.get_dim(), self._data,
                                                                 data_num=self._parameter.get_train_size())
                if x is None:
                    self.solution_counter = self._parameter.get_train_size()
                    return self.generate_solution()
            else:
                assert False
                # if self.solution_counter < self._parameter.get_positive_size():
                #     ToolFunction.log(" [generate_solution] set positive data")
                #     x, distinct_flag = self.distinct_sample_from_set(self._objective.get_dim(), self._data,
                #                                                      data_num=self._parameter.get_train_size())
                #     if x is None:
                #         self.solution_counter = self._parameter.get_train_size()
                #         return self.generate_solution()
                # else:
                #     ToolFunction.log(" [generate_solution] set negative data")
                #     for i in range(self.get_parameters().get_negative_size()):
                #         x, distinct_flag = self.distinct_sample_from_set(self._objective.get_dim(), self._data,
                #                                                          data_num=self._parameter.get_train_size())
                #         x.set_value(999999)
                #         self._data.append(x)
                #         self.solution_counter += 1
                #     self.selection(self._data)
                #     self.print_all_solution()
                #     self.finish_init = True
                #     return self.generate_solution()
        else:
            if not self.finish_init:
                ToolFunction.log(" [generate_solution][WARNING] didn't finish init yet: %s. " % self.solution_counter)
                return None, None
            # in_re_eval_mode should only call once a time.
            # if not self.in_re_eval_mode and self.get_parameters().re_eval_solution and (self.non_update_times + 1) % int(
            #         self._parameter.get_non_update_allowed() / 1.5) == 0:
            #     self.non_update_times += 1
            #     ToolFunction.log("[construct_next_explore_actor] do re-eval")
            #     self.in_re_eval_mode = True
            #     self.re_eval_index = 0
            #
            #     if self._parameter.drop_re_eval:
            #         for sol in self._positive_data:
            #             sol.set_value(self._objective.return_before * 0.9 if self._objective.return_before < 0 else self._objective.return_before * 1.1)
            #         self.in_re_eval_mode = False
            #         self._positive_data = sorted(self._positive_data, key=lambda x: x.get_value())
            #         self._best_solution = self._positive_data[0]
            #         ToolFunction.log("[construct_next_explore_actor] end re-eval")
            #         self.end_re_eval_solution()
            #         return self.generate_solution()
            #         # self.re_eval_solution_list = self._positive_data.copy()
            #     else:
            #         self.re_eval_solution_list = self._positive_data.copy()
            #         # self.re_eval_solution_list = self._positive_data
            #         return self.get_next_re_eval_solution()
            # elif self.in_re_eval_mode:
            #     ToolFunction.log("[construct_next_explore_actor] in re-eval. get next solution")
            #     return self.get_next_re_eval_solution()
            ToolFunction.log(" [generate_solution] generate by classfication. counter %s" % self.solution_counter)
            if gl.rand.random() < self._parameter.get_probability():
                # if self._positive_data[0].get
                if len(self.baseline_data) > 0 and self.baseline_data[0].get_value() < self._positive_data[0].get_value():
                    pd = self._positive_data + self.baseline_data
                else:
                    pd = self._positive_data
                classifier = RacosClassification(
                    self._objective.get_dim(), pd,
                    self._negative_data, self.must_select_index, self.ub)
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
                    self.generate_failed = True
                    self.solution_counter += 1
                    return self.get_best_solution(), self.solution_counter  # self.get_best_solution()
                else:
                    return self.generate_solution()
        # self.current_solution = x
        self.solution_counter += 1
        post_index = self.solution_counter

        return x, post_index

    def update_baseline_solution(self, feed_solution):
        self.baseline_data = [feed_solution]

    def update_racos_stats(self, solution_eval, attach, feed_solution, idx):
        if self.need_restart:
            logger.warn("[WARN] SRACOS need restart")
            return None
        assert isinstance(feed_solution, Solution)
        feed_solution.set_value(solution_eval)
        feed_solution.set_post_attach(attach)
        ToolFunction.log("[update racos] idx %s. solution info:" % (idx))
        feed_solution.print_solution(self._parameter, record=True)
        if self.generate_failed:
            ToolFunction.log("[WARNING] generate failed")
            return self.get_best_solution()
        if idx < self._parameter.get_train_size():
            # do nothing.
            self._data.append(feed_solution)
        elif idx == self._parameter.get_train_size():
            self._data.append(feed_solution)
            self.selection(self._data)
            self.finish_init = True
        else:
            self.selection(self._positive_data + self._negative_data)
            bad_ele = self.replace(self._positive_data, feed_solution, 'pos')
            ToolFunction.log("[update racos] replace solution :")
            bad_ele.print_solution(self._parameter, record=False)
            # ToolFunction.log("replace solution value: %s, x:%s" % (bad_ele.get_value(), bad_ele.get_x()))
            self.replace(self._negative_data, bad_ele, 'neg', self.strategy)
            self._positive_data = sorted(self._positive_data, key=lambda x: x.get_value())
            self._best_solution = self._positive_data[0]
            self.update_ub()
            self.last_best = self._best_solution

            # early stop
            if self._parameter.early_stop > 0 and not self.dont_early_stop:
                # 当前的solution value 应该小于某一个阈值,这个阈值应该比上一次的opt得到的return更容易达到.所以我们需要让return_before向大的方向移动.
                if feed_solution.get_value() < self._objective.return_before * 0.99 \
                        if self._objective.return_before < 0 else self._objective.return_before * 1.01:
                    self.dont_early_stop = True
                elif idx - self.get_parameters().get_train_size() > self._parameter.early_stop:
                    ToolFunction.log('[break loop] early stop for too low value.')
                    self.need_restart = True
                    return self.get_best_solution()
                ToolFunction.log(
                    '[early stop warning ]: current solution idx %s - %s, target %s. current value %s. target value %s' % (
                        idx, self.get_parameters().get_train_size(),
                        self._parameter.early_stop, feed_solution.get_value(), self._objective.return_before * 0.9))
            ToolFunction.log('[iter log] idx %s - %s, counter %s, non_update_times %s, non_update_allowed %s ' % (
                idx, self.get_parameters().get_train_size(), self.solution_counter,
                self.non_update_times, self._parameter.get_non_update_allowed()))
            # if self.solution_counter % 3 == 0:
            #     need_restartd_list = []
            #     d_list = []
            #     de_list = []
            #     for pos in self._positive_data:
            #         x_p = np.array(pos.get_x())[:-1]
            #         de_list.append(x_p)
            #     logger.record_tabular('racos/positive_data_std', np.std(de_list))
            #     if self.get_parameters().low_bound_distance > 0 and np.std(de_list) < self.get_parameters().low_bound_distance:
            #         logger.record_tabular('racos/de_stop', 1)
            #         self.need_restart = True
            #         return self.get_best_solution()
            #     else:
            #         logger.record_tabular('racos/de_stop', 0)
            #     logger.dump_tabular()

            if self.solution_counter % 100 <= 2:
                self.print_all_solution(record=False)
        return None

    def _is_worest(self, solution):
        return self._positive_data[-1].get_value() <= solution.get_value()

    def add_custom_solution(self, solution):
        """
        目前使用场景记录
        1. 在baseline solution 增加的时候使用
        2. 在设置re-eval 的value的时候使用
        :param solution:
        :return:
        """
        found = False
        if solution is not None:
            if not self.finish_init:
                ToolFunction.log("[add_custom_solution] init not complete, just do nothing.")
                return
            for index, sol in enumerate(self._positive_data):
                if sol.is_the_same(solution):
                    ToolFunction.log("[add_custom_solution] solution in positive data, value %s" % sol.get_value())
                    sol.set_value(solution.get_value())
                    # self._positive_data[index] = solution
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
                bad_ele.print_solution(self._parameter)
            self._positive_data = sorted(self._positive_data, key=lambda x: x.get_value())
            self._best_solution = self._positive_data[0]
            self.last_best = self._best_solution
            # self.last_best = self._best_solution
            # if self.solution_counter % 5 == 0:
            #     self.print_all_solution()

    def set_re_eval_solution(self, solution):
        ToolFunction.log("[set_re_eval_solution]")
        self.add_custom_solution(solution)
        tester = self.get_objective().tester
        tester.add_custom_record('re-eval-point', x=tester.time_step_holder.get_time(),
                                 y=solution.get_value() * -1,
                                 x_name='time step', y_name='re-eval-point')
        self._positive_data = sorted(self._positive_data, key=lambda x: x.get_value())
        self._best_solution = self._positive_data[0]
        # self.last_best = self._positive_data[0]

    def end_re_eval_solution(self):
        ToolFunction.log("---print positive solution----")
        for i in range(len(self._positive_data)):
            ToolFunction.log("i : %s, value %s " % (i, self._positive_data[i].get_value()))
        ToolFunction.log("----end----")

    def print_all_solution(self, record=False, name='solution'):
        ToolFunction.log("----print positive solution----")
        for index, sol in enumerate(self._positive_data):
            ToolFunction.log("[%s]" % str(index))
            sol.print_solution(self._parameter, record, name=name)
        if self.strategy == 'none' and self.solution_counter % 100 == 0:
            ToolFunction.log("----print negative solution----")
            for index, sol in enumerate(self._negative_data):
                ToolFunction.log("[%s]" % str(index))
                sol.print_solution(self._parameter, False, name=name)
        elif self.strategy != 'none':
            ToolFunction.log("----print negative solution----")
            for index, sol in enumerate(self._negative_data):
                ToolFunction.log("[%s]" % str(index))
                sol.print_solution(self._parameter, record, name=name)

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
