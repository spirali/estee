import itertools
import random

from schedsim.schedulers.queue import GreedyTransferQueueScheduler
from .scheduler import SchedulerBase
from .utils import compute_alap, compute_b_level_duration, compute_t_level_duration, \
    get_duration_estimate, get_size_estimate, schedule_all, transfer_cost_parallel, \
    worker_estimate_earliest_time
from ..simulator import TaskAssignment


class K1hScheduler(SchedulerBase):
    def schedule(self, new_ready, new_finished):
        return schedule_all(self.simulator.workers, new_ready,
                            lambda w, t, a: self.find_assignment(w, t))

    def find_assignment(self, workers, tasks):
        return min(itertools.product(workers, tasks),
                   key=lambda item: self.calculate_cost(item[0], item[1]))

    def calculate_cost(self, worker, task):
        if task.cpus > worker.cpus:
            return 10e10
        transfer = self.calculate_transfer(worker, task)
        cpu = get_duration_estimate(task)
        worker_cost = self.worker_cost(worker)

        return transfer + cpu + worker_cost

    def worker_cost(self, worker):
        return sum(get_duration_estimate(t) for t in worker.assigned_tasks)

    def calculate_transfer(self, worker, task):
        bandwidth = self.simulator.netmodel.bandwidth
        cost = transfer_cost_parallel(self.simulator.runtime_state, worker, task)

        for c in task.consumers():
            for i in c.inputs:
                if (i.parent != task and
                        worker not in self.simulator.runtime_state.output_info(i).placing):
                    cost += get_size_estimate(self.simulator.runtime_state, i)

        return cost / bandwidth


class DLSScheduler(SchedulerBase):
    """
    Implementation of the dynamic level scheduler (DLS) from
    A Compile-Time Scheduling Heuristic for Interconnection-Constrained Heterogeneous Processor
    Architectures (1993)

    The scheduler calculates t.b_level -
    (minimum time when all dependencies of task t are available on worker w)
    for all task-worker pairs (t, w) and selects the maximum.
    """

    def init(self, simulator):
        super().init(simulator)
        self.b_level = compute_b_level_duration(simulator.task_graph)

    def schedule(self, new_ready, new_finished):
        return schedule_all(self.simulator.workers, new_ready,
                            lambda w, t, a: self.find_assignment(w, t, a))

    def find_assignment(self, workers, tasks, worker_assignments):
        return max(itertools.product(workers, tasks),
                   key=lambda item: self.calculate_cost(item[0], item[1],
                                                        worker_assignments.get(item[0], [])))

    def calculate_cost(self, worker, task, worker_assignments):
        if task.cpus > worker.cpus:
            return -10e10

        now = self.simulator.env.now
        earliest_transfer = (transfer_cost_parallel(self.simulator.runtime_state, worker, task) /
                             self.simulator.netmodel.bandwidth)

        earliest_computation = worker_estimate_earliest_time(
            worker, task, self.simulator.env.now, worker_assignments)

        return self.b_level[task] - (now + max(earliest_transfer, earliest_computation))


class LASTScheduler(SchedulerBase):
    """
    Implementation of the LAST scheduler from
    The LAST Algorithm: A Heuristic-Based Static Task Allocation
    Algorithm (1989)

    The scheduler tries to minimize overall communication by prioriting tasks
    with small sizes and small neighbours.
    """
    def schedule(self, new_ready, new_finished):
        bandwidth = self.simulator.netmodel.bandwidth
        workers = self.simulator.workers[:]
        runtime_state = self.simulator.runtime_state

        def edge_cost(o1, t2):
            if (runtime_state.output_info(o1).placing ==
                    runtime_state.task_info(t2).assigned_workers):
                return 0
            else:
                return 1

        d_nodes = {}
        for task in new_ready:
            if not task.inputs:
                d_nodes[task] = 1
            else:
                sizes = tuple((get_size_estimate(runtime_state, i) / bandwidth)
                              for i in task.inputs)
                input_weighted = sum([s * edge_cost(i, task)
                                      for (s, i) in zip(sizes, task.inputs)])
                input = sum(sizes)
                output = sum((get_size_estimate(runtime_state, output) / bandwidth)
                             for output in task.outputs)
                d_nodes[task] = (input_weighted + output) / max(0.001, input + output)

        def worker_cost(worker, task):
            if task.cpus > worker.cpus:
                return 10e10
            return 0

        schedules = []
        for _ in new_ready[:]:
            m = max(d_nodes, key=lambda t: d_nodes[t])
            worker_costs = [transfer_cost_parallel(runtime_state, w, m) +
                            worker_cost(w, m) for w in workers]
            worker_index = min(range(len(workers)),
                               key=lambda i: worker_costs[i])
            schedules.append(TaskAssignment(workers[worker_index], m))

            del d_nodes[m]
        return schedules


class MCPScheduler(SchedulerBase):
    """
    Implementation of the MCP (Modified Critical Path) scheduler from
    Hypertool: A Programming Aid for Message-Passing Systems (1990)

    The scheduler prioritizes tasks by their latest possible start times
    (ALAP).
    """
    def __init__(self):
        super().__init__()
        self.alap = {}

    def init(self, simulator):
        super().init(simulator)
        bandwidth = simulator.netmodel.bandwidth
        self.alap = compute_alap(self.simulator.runtime_state, self.simulator.task_graph, bandwidth)

    def schedule(self, new_ready, new_finished):
        tasks = sorted(new_ready,
                       key=lambda t: sorted([self.alap[t]] + [self.alap[c] for c in t.consumers()],
                                            reverse=True))
        bandwidth = self.simulator.netmodel.bandwidth

        worker_assignments = {}

        def cost(w, t):
            if t.cpus > w.cpus:
                return 10e10
            transfer = transfer_cost_parallel(self.simulator.runtime_state, w, t) / bandwidth
            computation = worker_estimate_earliest_time(w, task, self.simulator.env.now,
                                                        worker_assignments.get(w, []))
            return max(transfer, computation)

        schedules = []
        for task in tasks:
            worker = min(self.simulator.workers, key=lambda w: cost(w, task))
            schedules.append(TaskAssignment(worker, task))
            worker_assignments.setdefault(worker, []).append(task)

        return schedules


class MCPGTScheduler(GreedyTransferQueueScheduler):
    def make_queue(self):
        bandwidth = self.simulator.netmodel.bandwidth
        alap = compute_alap(self.simulator.runtime_state,
                            self.simulator.task_graph,
                            bandwidth)
        tasks = self.simulator.task_graph.tasks[:]
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
        super().__init__()
        self.b_level = {}

    def init(self, simulator):
        super().init(simulator)
        self.b_level = compute_b_level_duration(simulator.task_graph)

    def schedule(self, new_ready, new_finished):
        return schedule_all(self.simulator.workers, new_ready,
                            lambda w, t, a: self.find_assignment(w, t, a))

    def find_assignment(self, workers, tasks, worker_assignments):
        return min(itertools.product(workers, tasks),
                   key=lambda item: (self.calculate_cost(item[0],
                                                         item[1],
                                                         worker_assignments.get(item[0], [])),
                                     -self.b_level[item[1]]))

    def calculate_cost(self, worker, task, worker_assignments):
        if task.cpus > worker.cpus:
            return 10e10

        bandwidth = self.simulator.netmodel.bandwidth
        transfer = transfer_cost_parallel(self.simulator.runtime_state, worker, task) / bandwidth
        computation = worker_estimate_earliest_time(worker, task,
                                                    self.simulator.env.now, worker_assignments)
        return max(computation, transfer)


class StaticSortScheduler(SchedulerBase):
    def sort_tasks(self, tasks):
        raise NotImplementedError()

    def schedule(self, new_ready, new_finished):
        return schedule_all(self.simulator.workers, new_ready,
                            lambda w, t, a: self.find_assignment(w, t, a))

    def find_assignment(self, workers, tasks, worker_assignments):
        tasks = self.sort_tasks(tasks)
        task = tasks[0]
        return (min(workers, key=lambda w: self.calculate_cost(w, task,
                                                               worker_assignments.get(w, []))),
                task)

    def calculate_cost(self, worker, task, worker_assignments):
        if task.cpus > worker.cpus:
            return 10e10

        earliest_transfer = (transfer_cost_parallel(self.simulator.runtime_state, worker, task) /
                             self.simulator.netmodel.bandwidth)

        earliest_computation = worker_estimate_earliest_time(
            worker, task, self.simulator.env.now, worker_assignments)

        return max(earliest_transfer, earliest_computation)


class BlevelScheduler(StaticSortScheduler):
    def init(self, simulator):
        super().init(simulator)
        self.b_level = compute_b_level_duration(simulator.task_graph)

    def sort_tasks(self, tasks):
        return sorted(tasks, key=lambda t: self.b_level[t], reverse=True)


class TlevelScheduler(StaticSortScheduler):
    def init(self, simulator):
        super().init(simulator)
        self.t_level = compute_t_level_duration(simulator.task_graph)

    def sort_tasks(self, tasks):
        return sorted(tasks, key=lambda t: self.t_level[t])
