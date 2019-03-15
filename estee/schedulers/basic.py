
from .scheduler import SchedulerBase, StaticScheduler, TaskState
from .utils import max_cpus_worker, compute_b_level_duration
from ..simulator import TaskAssignment
import numpy as np


class DoNothingScheduler(SchedulerBase):

    def __init__(self):
        super().__init__("do-nothing", "0")

    def schedule(self, update):
        pass


class RandomAssignScheduler(StaticScheduler):

    def __init__(self):
        super().__init__("random-s", "0")

    def static_schedule(self):
        tasks = [t for t in self.task_graph.tasks.values() if t.state == TaskState.Waiting]
        np.random.shuffle(tasks)

        workers = list(self.workers.values())

        p = np.array([w.cpus for w in workers], dtype=np.float)
        p /= p.sum()

        for t in tasks:
            w = np.random.choice(workers, p=p)
            while w.cpus < t.cpus:
                w = np.random.choice(workers, p=p)
            self.assign(w, t)


class AllOnOneScheduler(SchedulerBase):

    def __init__(self):
        super().__init__("single", "0")
        self.worker = None
        self.b_level = None

    def schedule(self, update):
        if update.cluster_changed:
            worker = max_cpus_worker(self.workers.values())
            self.worker = worker
        else:
            worker = self.worker

        if update.graph_changed:
            b_level = compute_b_level_duration(self.task_graph)
            self.b_level = b_level
        else:
            b_level = self.b_level

        for task in update.new_ready_tasks:
            self.assign(worker, task, b_level[task])