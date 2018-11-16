
import random

import numpy as np

from . import StaticScheduler
from .utils import compute_b_level_duration, compute_independent_tasks, get_duration_estimate, \
    get_size_estimate, max_cpus_worker
from ..simulator import TaskAssignment


class Camp2Scheduler(StaticScheduler):

    def __init__(self, iterations=2000):
        super().__init__()
        self.iterations = iterations

    def init(self, simulator):
        super().init(simulator)

    def static_schedule(self):
        independencies = compute_independent_tasks(self.simulator.task_graph)
        workers = self.simulator.workers
        cpu_factor = sum([w.cpus for w in workers]) / len(workers) / 8

        self.repulse_score = {}

        for task, indeps in independencies.items():
            lst = []
            self.repulse_score[task] = lst
            if not indeps:
                continue
            task_value = get_duration_estimate(task) / len(indeps) * task.cpus / cpu_factor
            for t in indeps:
                score = task_value + get_duration_estimate(t) / len(independencies[t]) \
                        * t.cpus / cpu_factor
                lst.append((t.id, score))

        tasks = self.simulator.task_graph.tasks
        placement = np.empty(self.simulator.task_graph.task_count,
                             dtype=np.int32)
        placement[:] = workers.index(max_cpus_worker(workers))
        # score_cache = np.empty_like(placement, dtype=np.float)

        # for t in tasks:
        #    score_cache[t.id] = self.compute_task_score(placement, t)

        for i in range(self.iterations):
            t = random.randint(0, len(tasks) - 1)
            old_w = placement[t]
            new_w = random.randint(0, len(workers) - 2)
            if new_w >= old_w:
                new_w += 1
            if workers[new_w].cpus < tasks[t].cpus:
                continue
            old_score = self.compute_task_score(placement, tasks[t])
            placement[t] = new_w
            new_score = self.compute_task_score(placement, tasks[t])
            # and np.random.random() > (i / limit) / 100:
            if new_score > old_score:
                placement[t] = old_w

        b_level = compute_b_level_duration(self.simulator.task_graph)

        r = [TaskAssignment(workers[w], task, b_level[task])
             for task, w in zip(tasks, placement)]
        return r

    def compute_input_score(self, placement, task):
        old_worker = placement[task.id]
        score = 0
        for inp in task.inputs:
            size = get_size_estimate(self.simulator, inp)
            if size > score and placement[inp.id] != old_worker:
                score = size
        return score

    def compute_task_score(self, placement, task):
        score = self.compute_input_score(placement, task)
        for t in task.consumers():
            score += self.compute_input_score(placement, t)
        score /= self.simulator.netmodel.bandwidth
        p = placement[task.id]
        for t_id, v in self.repulse_score[task]:
            if placement[t_id] == p:
                score += v
        """
        tids, scores = self.repulse_score[task]
        score += (scores * (placement[tids] == p)).sum()
        """
        return score

    """
    def placement_cost(self, placement):
        s = 0
        bandwidth = self.simulator.netmodel.bandwidth

        for t in self.simulator.task_graph.tasks:
            p = placement[t.id]
            m = 0
            for inp in t.inputs:
                size = get_expected_size(self.simulator, inp)
                if p != placement[inp.id] and size > m:
                    m = size
            s += m

        s /= bandwidth

        a = placement[self.tab[:, 0]]
        b = placement[self.tab[:, 1]]
        return (self.costs * (a == b)).sum() + s
    """
