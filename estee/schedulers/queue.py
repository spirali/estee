import random

import numpy as np

from .scheduler import SchedulerBase
from .utils import compute_b_level_duration, compute_t_level_duration
from ..simulator import TaskAssignment


class QueueScheduler(SchedulerBase):

    def __init__(self):
        self.ready = []
        self.queue = None

    def init(self, simulator):
        super().init(simulator)
        self.queue = self.make_queue()

    def make_queue(self):
        raise NotImplementedError()

    def choose_worker(self, workers, task):
        raise NotImplementedError()

    def schedule(self, new_ready, new_finished):
        self.ready += new_ready
        results = []
        free_cpus = np.zeros(len(self.simulator.workers))
        workers = self.simulator.workers
        for i, worker in enumerate(workers):
            free_cpus[i] = worker.cpus
            for a in worker.assignments:
                free_cpus[i] -= a.task.cpus
        aws = list(range(len(workers)))
        for t in self.queue[:]:
            if t in self.ready:
                ws = [i for i in aws
                      if free_cpus[i] >= t.cpus]
                if not ws:
                    aws = [i for i in aws if workers[i].cpus < t.cpus]
                    if aws:
                        continue
                    else:
                        break
                self.ready.remove(t)
                self.queue.remove(t)
                idx = self.choose_worker([workers[i] for i in ws], t)
                idx = ws[idx]
                free_cpus[idx] -= t.cpus
                results.append(TaskAssignment(workers[idx], t))
        return results


class RandomScheduler(QueueScheduler):

    def make_queue(self):
        tasks = self.simulator.task_graph.tasks[:]
        random.shuffle(tasks)
        return tasks

    def choose_worker(self, workers, task):
        return random.randrange(len(workers))


class GreedyTransferQueueScheduler(QueueScheduler):

    def choose_worker(self, workers, task):
        costs = np.zeros(len(workers))
        runtime_state = self.simulator.runtime_state
        for i in range(len(workers)):
            w = workers[i]
            for inp in task.inputs:
                if w not in runtime_state.output_info(inp).placing:
                    costs[i] += inp.size

        return np.random.choice(np.flatnonzero(costs == costs.min()))


class RandomGtScheduler(GreedyTransferQueueScheduler):

    def make_queue(self):
        tasks = self.simulator.task_graph.tasks[:]
        random.shuffle(tasks)
        return tasks


class BlevelGtScheduler(GreedyTransferQueueScheduler):

    def make_queue(self):
        b_level = compute_b_level_duration(self.simulator.task_graph)
        tasks = self.simulator.task_graph.tasks[:]
        random.shuffle(tasks)  # To randomize keys with the same level
        tasks.sort(key=lambda n: b_level[n], reverse=True)
        return tasks


class TlevelGtScheduler(GreedyTransferQueueScheduler):

    def make_queue(self):
        t_level = compute_t_level_duration(self.simulator.task_graph)
        tasks = self.simulator.task_graph.tasks[:]
        random.shuffle(tasks)  # To randomize keys with the same level
        tasks.sort(key=lambda n: t_level[n])
        return tasks
