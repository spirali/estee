
from .scheduler import SchedulerBase, StaticScheduler
from ..simulator import TaskAssignment
from .utils import max_cpus_worker, compute_b_level
import random


class DoNothingScheduler(SchedulerBase):
    pass


class RandomAssignScheduler(StaticScheduler):

    def init(self, simulator):
        self.scheduled = False

    def static_schedule(self):
        tasks = list(self.simulator.task_graph.tasks)
        random.shuffle(tasks)

        workers = self.simulator.workers

        results = []
        for t in tasks:
            w = random.choice(workers)
            while w.cpus < t.cpus:
                w = random.choice(workers)
            results.append(TaskAssignment(w, t))
        return results


class AllOnOneScheduler(SchedulerBase):

    def init(self, simulator):
        super().init(simulator)
        self.worker = max_cpus_worker(self.simulator.workers)

    def schedule(self, new_ready, new_finished):
        worker = self.worker
        b_level = compute_b_level(self.simulator.task_graph,
                                  lambda t: t.duration)
        return [TaskAssignment(worker, task, b_level[task])
                for task in new_ready]

