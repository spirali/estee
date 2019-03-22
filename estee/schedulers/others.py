import itertools
import random

from estee.schedulers.queue import GreedyTransferQueueScheduler
from .scheduler import SchedulerBase, Update
from .utils import compute_alap, compute_b_level_duration, compute_t_level_duration, \
    get_size_estimate, schedule_all, transfer_cost_parallel, \
    worker_estimate_earliest_time, update_worker_occupancy


def apply_schedule(scheduler, schedules):
    for assignment in schedules:
        scheduler.assign(assignment.worker, assignment.task)


class DLSScheduler(SchedulerBase):
    """
    Implementation of the dynamic level scheduler (DLS) from
    A Compile-Time Scheduling Heuristic for Interconnection-Constrained Heterogeneous Processor
    Architectures (1993)

    The scheduler calculates t.b_level -
    (minimum time when all dependencies of task t are available on worker w)
    for all task-worker pairs (t, w) and selects the maximum.
    """
    def __init__(self):
        super().__init__("DLS", "0", task_start_notification=True, only_in_simulator=True)
        self.b_level = {}

    def schedule(self, update: Update):
        if update.graph_changed:
            self.b_level = compute_b_level_duration(self.task_graph)
        update_worker_occupancy(self.workers, update)

        workers = list(self.workers.values())
        apply_schedule(self, schedule_all(workers, update.new_ready_tasks,
                                          lambda w, t, a: self.find_assignment(w, t, a)))

    def find_assignment(self, workers, tasks, worker_assignments):
        return max(itertools.product(workers, tasks),
                   key=lambda item: self.calculate_cost(item[0], item[1],
                                                        worker_assignments.get(item[0], [])))

    def calculate_cost(self, worker, task, worker_assignments):
        if task.cpus > worker.cpus:
            return -10e10

        earliest_transfer = (transfer_cost_parallel(self.task_graph, worker, task) /
                             self.network_bandwidth)

        earliest_computation = worker_estimate_earliest_time(worker, task,
                                                             self._simulator.env.now,
                                                             worker_assignments)

        return self.b_level[task] - max(earliest_transfer, earliest_computation)


class MCPScheduler(SchedulerBase):
    """
    Implementation of the MCP (Modified Critical Path) scheduler from
    Hypertool: A Programming Aid for Message-Passing Systems (1990)

    The scheduler prioritizes tasks by their latest possible start times
    (ALAP).
    """
    def __init__(self):
        super().__init__("MCP", "0", task_start_notification=True, only_in_simulator=True)
        self.alap = {}

    def schedule(self, update):
        bandwidth = self.network_bandwidth
        update_worker_occupancy(self.workers, update)

        if update.graph_changed:
            self.alap = compute_alap(self.task_graph, get_size_estimate, bandwidth)

        tasks = sorted(update.new_ready_tasks,
                       key=lambda t: sorted([self.alap[t]] + [self.alap[c] for c in t.consumers()],
                                            reverse=True))

        worker_assignments = {}

        def cost(w, t):
            if t.cpus > w.cpus:
                return 10e10
            transfer = transfer_cost_parallel(self.task_graph, w, t) / bandwidth
            computation = worker_estimate_earliest_time(w, task, self._simulator.env.now,
                                                        worker_assignments.get(w, []))
            return max(transfer, computation)

        for task in tasks:
            worker = min(self.workers.values(), key=lambda w: cost(w, task))
            self.assign(worker, task)
            worker_assignments.setdefault(worker, []).append(task)


class MCPGTScheduler(GreedyTransferQueueScheduler):
    def make_queue(self):
        bandwidth = self.network_bandwidth
        alap = compute_alap(self.task_graph, get_size_estimate, bandwidth)
        tasks = list(self.task_graph.tasks.values())
        random.shuffle(tasks)  # To randomize keys with the same level
        tasks.sort(key=lambda t: sorted([alap[t]] + [alap[c] for c in t.consumers()],
                                        reverse=True))
        return tasks


class ETFScheduler(SchedulerBase):
    """
    Implementation of the ETF (Earliest Time First) scheduler from
    Scheduling Precedence Graphs in Systems with Interprocessor Communication
    Times (1989)

    The scheduler prioritizes (worker, task) pairs with the earliest possible
    start time. Ties are broken with static B-level.
    """
    def __init__(self):
        super().__init__("ETF", "0", only_in_simulator=True)
        self.b_level = {}

    def schedule(self, update):
        if update.graph_changed:
            self.b_level = compute_b_level_duration(self.task_graph)
        update_worker_occupancy(self.workers, update)

        apply_schedule(self, schedule_all(self.workers.values(), update.new_ready_tasks,
                                          lambda w, t, a: self.find_assignment(w, t, a)))

    def find_assignment(self, workers, tasks, worker_assignments):
        return min(itertools.product(workers, tasks),
                   key=lambda item: (self.calculate_cost(item[0],
                                                         item[1],
                                                         worker_assignments.get(item[0], [])),
                                     -self.b_level[item[1]]))

    def calculate_cost(self, worker, task, worker_assignments):
        if task.cpus > worker.cpus:
            return 10e10

        bandwidth = self.network_bandwidth
        transfer = transfer_cost_parallel(self.task_graph, worker, task) / bandwidth
        computation = worker_estimate_earliest_time(worker, task, self._simulator.env.now,
                                                    worker_assignments)
        return max(computation, transfer)


class StaticSortScheduler(SchedulerBase):
    def sort_tasks(self, tasks):
        raise NotImplementedError()

    def schedule(self, update):
        if update.graph_changed:
            self.recalculate()

        for assignment in schedule_all(self.workers, update.new_ready_tasks,
                                       lambda w, t, a: self.find_assignment(w, t, a)):
            self.assign(assignment.worker, assignment.task)

    def find_assignment(self, workers, tasks, worker_assignments):
        tasks = self.sort_tasks(tasks)
        task = tasks[0]
        return (min(workers, key=lambda w: self.calculate_cost(w, task,
                                                               worker_assignments.get(w, []))),
                task)

    def calculate_cost(self, worker, task, worker_assignments):
        if task.cpus > worker.cpus:
            return 10e10

        earliest_transfer = (transfer_cost_parallel(self.task_graph, worker, task) /
                             self.network_bandwidth)

        earliest_computation = worker_estimate_earliest_time(
            worker, task, self._simulator.env.now, worker_assignments)

        return max(earliest_transfer, earliest_computation)

    def recalculate(self):
        raise NotImplementedError()


class BlevelScheduler(StaticSortScheduler):
    def __init__(self):
        super().__init__("B level", "0")
        self.b_level = {}

    def recalculate(self):
        self.b_level = compute_b_level_duration(self.task_graph)

    def sort_tasks(self, tasks):
        return sorted(tasks, key=lambda t: self.b_level[t], reverse=True)


class TlevelScheduler(StaticSortScheduler):
    def __init__(self):
        super().__init__("T level", "0")
        self.t_level = {}

    def recalculate(self):
        self.t_level = compute_t_level_duration(self.task_graph)

    def sort_tasks(self, tasks):
        return sorted(tasks, key=lambda t: self.t_level[t])
