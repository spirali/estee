from .scheduler import SchedulerBase, TaskState
from .utils import compute_b_level_duration, compute_t_level_duration
from ..simulator import TaskAssignment
import numpy as np

class WorkStealingScheduler(SchedulerBase):

    def __init__(self):
        super().__init__("ws", "0", reassigning=True)

    def schedule(self, update):

        for w in update.new_workers:
            w.free_cpus = w.cpus
            w.tasks = set()

        if update.graph_changed:
            self.b_level = compute_b_level_duration(self.task_graph)

        for task in update.reassign_failed:
            for worker in self.workers.values():
                if task in worker.tasks:
                    worker.tasks.remove(task)
                    worker.free_cpus += task.cpus
            worker = task.scheduled_worker
            worker.tasks.add(task)
            worker.free_cpus -= task.cpus

        for task in update.new_finished_tasks:
            task.scheduled_worker.free_cpus += task.cpus
            task.scheduled_worker.tasks.remove(task)

        plan = {}
        if update.new_ready_tasks:
            workers = list(self.workers.values())
            for task in update.new_ready_tasks:
                worker = self.choose_worker([w for w in workers if w.cpus >= task.cpus], task)
                plan[task] = worker
                worker.tasks.add(task)
                worker.free_cpus -= task.cpus

        for worker in self.workers.values():
            if worker.free_cpus > 0:
                self.process_work_stealing(worker, plan)

        for task, worker in plan.items():
            level = self.b_level[task]
            self.assign(worker, task, level, level - task.expected_duration)

    def process_work_stealing(self, worker, plan):
        tasks = []
        for w in self.workers.values():
            if w.free_cpus >= 0:
                continue
            tasks.extend(w.tasks)

        # TODO: Try random sort for benchmark
        def sort_key(task):
            return self.task_worker_cost(worker, task) / task.expected_duration
        tasks.sort(key=sort_key, reverse=False)

        for task in tasks:
            cpus = task.cpus
            if cpus > worker.cpus:
                continue
            w = plan.get(task, task.scheduled_worker)
            if w.free_cpus - cpus >= worker.free_cpus or w.free_cpus + cpus > 0:
                continue
            #print("STEALING {} : {}->{}".format(task.id, w.worker_id, worker.worker_id))

            w.free_cpus += cpus
            w.tasks.remove(task)
            worker.free_cpus -= cpus
            worker.tasks.add(task)
            plan[task] = worker

    def task_worker_cost(self, worker, task):
        cost = 0
        for inp in task.inputs:
            if worker in inp.availability or worker in inp.placing:
                continue
            if worker in inp.scheduled:
                cost += 0.10 * inp.size
            else:
                cost += inp.size
        return cost

    def choose_worker(self, workers, task):
        costs = np.zeros(len(workers))
        for i in range(len(workers)):
            costs[i] = self.task_worker_cost(workers[i], task)
        return workers[np.random.choice(np.flatnonzero(costs == costs.min()))]
