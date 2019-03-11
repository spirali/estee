import random

import numpy as np
import collections

from .scheduler import SchedulerBase, TaskState
from .utils import compute_b_level_duration, compute_t_level_duration
from ..simulator import TaskAssignment


class QueueScheduler(SchedulerBase):

    def __init__(self, name, version):
        super().__init__(name, version)
        self.queue = collections.deque()
        self.ready = []
        self.w_assignments = {}
        self.free_cpus = None

    def make_queue(self):
        raise NotImplementedError()

    def choose_worker(self, workers, task):
        raise NotImplementedError()

    def schedule(self, new_ready, new_finished, graph_changed, cluster_changed):
        self.ready += new_ready

        if cluster_changed:
            free_cpus = {w: w.cpus for w in self.workers.values()}
            for task in self.task_graph.tasks.values():
                if task.state != TaskState.Finished and task.worker:
                    free_cpus[task.worker] -= task.cpus
            self.free_cpus = free_cpus
        else:
            free_cpus = self.free_cpus

        if graph_changed:
            self.queue = self.make_queue()

        for task in new_finished:
            free_cpus[task.scheduled_worker] += task.cpus

        aws = set(self.workers.values())
        for t in list(self.queue):
            if t in self.ready:
                ws = [w for w in aws
                      if free_cpus[w] >= t.cpus]
                if not ws:
                    aws = [w for w in aws if w.cpus < t.cpus]
                    if aws:
                        continue
                    else:
                        break
                self.ready.remove(t)
                self.queue.remove(t)
                w = self.choose_worker(ws, t)
                free_cpus[w] -= t.cpus
                self.assign(w, t)


class RandomScheduler(QueueScheduler):

    def __init__(self):
        super().__init__("random-q", "0")

    def make_queue(self):
        tasks = [t for t in self.task_graph.tasks.values() if t.state == TaskState.Waiting]
        random.shuffle(tasks)
        return tasks

    def choose_worker(self, workers, task):
        return random.choice(workers)


class GreedyTransferQueueScheduler(QueueScheduler):

    def choose_worker(self, workers, task):
        costs = np.zeros(len(workers))
        for i in range(len(workers)):
            w = workers[i]
            for inp in task.inputs:
                if w in inp.availability or w in inp.placing:
                    continue
                if w in inp.scheduled:
                    costs[i] += 0.10 * inp.size
                else:
                    costs[i] += inp.size
        return workers[np.random.choice(np.flatnonzero(costs == costs.min()))]


class RandomGtScheduler(GreedyTransferQueueScheduler):

    def __init__(self):
        super().__init__("random-gt", "0")

    def make_queue(self):
        tasks = list(self.task_graph.tasks.values())
        random.shuffle(tasks)
        return tasks


class BlevelGtScheduler(GreedyTransferQueueScheduler):

    def __init__(self):
        super().__init__("blevel-gt", "0")

    def make_queue(self):
        b_level = compute_b_level_duration(self.task_graph)
        tasks = list(self.task_graph.tasks.values())
        random.shuffle(tasks)  # To randomize keys with the same level
        tasks.sort(key=lambda n: b_level[n], reverse=True)
        return tasks


class TlevelGtScheduler(GreedyTransferQueueScheduler):

    def __init__(self):
        super().__init__("tlevel-gt", "0")

    def make_queue(self):
        t_level = compute_t_level_duration(self.task_graph)
        tasks = list(self.task_graph.tasks.values())
        random.shuffle(tasks)  # To randomize keys with the same level
        tasks.sort(key=lambda n: t_level[n])
        return tasks
