
import random

import numpy as np

from . import SchedulerBase, StaticScheduler
from .utils import compute_b_level_duration, compute_independent_tasks, get_duration_estimate, \
    max_cpus_worker
from ..simulator import TaskAssignment


class CampCore:

    def __init__(self, simulator):
        self.simulator = simulator

        independencies = compute_independent_tasks(simulator.task_graph)
        self.independecies = independencies

        workers = self.simulator.workers

        placement = np.empty(len(simulator.task_graph.tasks),
                             dtype=np.int32)
        placement[:] = workers.index(max_cpus_worker(workers))
        self.placement = placement
        self.b_level = compute_b_level_duration(simulator.task_graph)

    def compute(self, iterations):

        placement = self.placement
        workers = self.simulator.workers
        independencies = self.independecies
        cpu_factor = sum([w.cpus for w in workers]) / len(workers)

        # Repulse score
        repulse_score = {}
        tasks = []
        self.tasks = tasks
        task_info = self.simulator.runtime_state.task_info

        for task, indeps in independencies.items():
            if not task_info(task).is_waiting:
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

        for i in range(iterations):
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
        for inp in task.inputs:
            size = inp.expected_size
            if size > score and placement[inp.parent.id] != old_worker:
                score = size
        return score

    def compute_task_score(self, repulse_score, placement, task):
        score = self.compute_input_score(placement, task)
        for t in task.consumers():
            score += self.compute_input_score(placement, t)
        score /= self.simulator.netmodel.bandwidth
        p = placement[task.id]
        for t_id, v in repulse_score[task]:
            if placement[t_id] == p:
                score += v
        return score

    def make_assignments(self):
        workers = self.simulator.workers
        placement = self.placement
        b_level = self.b_level

        return [TaskAssignment(workers[placement[task.id]], task, b_level[task])
                for task in self.tasks]


class Camp2Scheduler(StaticScheduler):

    def __init__(self, iterations=2000):
        super().__init__()
        self.iterations = iterations

    def init(self, simulator):
        super().init(simulator)

    def static_schedule(self):
        core = CampCore(self.simulator)
        core.compute(self.iterations)
        return core.make_assignments()


class TwoLayerScheduler(SchedulerBase):

    def compute_schedule(self):
        raise NotImplementedError()

    def schedule(self, new_ready, new_finished):
        assignments = self.compute_schedule()
        simulator = self.simulator
        result = []

        task_info = self.simulator.task_info

        free_cpus = {
            worker: worker.cpus - sum(t.cpus for t in worker.assigned_tasks)
            for worker in simulator.workers
        }

        assignments.sort(key=lambda a: a.priority, reverse=True)

        for assignment in assignments:
            # info = simulator.task_info(assignment.task)
            task = assignment.task
            # or any(task_info(o.parent).is_finished for o in task.inputs):
            if free_cpus[assignment.worker] > 0 and task_info(task).is_ready:
                result.append(assignment)
                free_cpus[assignment.worker] -= task.cpus

        """
        result = []
        for worker in self.simulator.workers:
            #free_cpus = 2 * worker.cpus - sum(t.cpus for t in worker.assigned_tasks if t.is_ready)
            #tasks = [self._is_task_prepared(t) for t in worker.s_info]
            result += [assignment for assignment in worker.s_info
                       if assignment.task.info.is_ready or (assignment.task.info.is_waiting and
                          any(t.info.is_ready for t in assignment.task.inputs))]
        """
        return result


class Camp3Scheduler(TwoLayerScheduler):

    def __init__(self, iterations=2000):
        super().__init__()
        self.iterations = iterations

    def init(self, simulator):
        super().init(simulator)
        self.core = CampCore(simulator)
        self.core.compute(4000)

    def compute_schedule(self):
        self.core.compute(self.iterations)
        return self.core.make_assignments()
