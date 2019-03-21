
import random

import numpy as np

from . import StaticScheduler
from .utils import compute_b_level_duration, compute_independent_tasks, get_duration_estimate, \
    max_cpus_worker


class CampCore:

    def __init__(self, task_graph, workers, network_bandwidth, default_size):
        independencies = compute_independent_tasks(task_graph)
        self.independecies = independencies
        self.workers = workers

        placement = np.empty(len(task_graph.tasks),
                             dtype=np.int32)
        placement[:] = workers.index(max_cpus_worker(workers))
        self.placement = placement
        self.b_level = compute_b_level_duration(task_graph)
        self.network_bandwidth = network_bandwidth
        self.default_size = default_size

    def compute(self, iterations):

        placement = self.placement
        workers = self.workers
        independencies = self.independecies
        cpu_factor = sum([w.cpus for w in workers]) / len(workers)

        # Repulse score
        repulse_score = {}
        tasks = []
        self.tasks = tasks

        for task, indeps in independencies.items():
            if not task.is_waiting:
                continue
            tasks.append(task)
            lst = []
            repulse_score[task] = lst
            if not indeps:
                continue
            task_value = get_duration_estimate(task) / len(indeps) * task.cpus / cpu_factor
            for t in indeps:
                score = task_value + get_duration_estimate(t) / len(independencies[t]) \
                        * t.cpus / cpu_factor
                lst.append((t.id, score))

        if not tasks:
            return

        for _ in range(iterations):
            t = random.randint(0, len(tasks) - 1)
            task = tasks[t]
            old_w = placement[task.id]
            new_w = random.randint(0, len(workers) - 2)
            if new_w >= old_w:
                new_w += 1
            if workers[new_w].cpus < tasks[t].cpus:
                continue
            old_score = self.compute_task_score(repulse_score, placement, task)
            placement[task.id] = new_w
            new_score = self.compute_task_score(repulse_score, placement, task)
            # and np.random.random() > (i / limit) / 100:
            if new_score > old_score:
                placement[task.id] = old_w

    def compute_input_score(self, placement, task):
        old_worker = placement[task.id]
        score = 0
        default_size = self.default_size
        for inp in task.inputs:
            size = inp.expected_size or default_size
            if size > score and placement[inp.parent.id] != old_worker:
                score = size
        return score

    def compute_task_score(self, repulse_score, placement, task):
        score = self.compute_input_score(placement, task)
        for t in task.consumers():
            score += self.compute_input_score(placement, t)
        score /= self.network_bandwidth
        p = placement[task.id]
        for t_id, v in repulse_score[task]:
            if placement[t_id] == p:
                score += v
        return score

    def make_assignments(self, builder):
        workers = self.workers
        placement = self.placement
        b_level = self.b_level

        for task in self.tasks:
            builder(workers[placement[task.id]], task, b_level[task])


class Camp2Scheduler(StaticScheduler):

    def __init__(self, iterations=2000):
        super().__init__("camp", "0")
        self.iterations = iterations

    def static_schedule(self):
        core = CampCore(self.task_graph,
                        [w for w in self.workers.values()],
                        self.network_bandwidth,
                        5)
        core.compute(self.iterations)
        core.make_assignments(self.assign)
