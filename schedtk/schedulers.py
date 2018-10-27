import itertools
import random

import numpy as np

from .simulator import TaskAssignment


class SchedulerBase:

    def init(self, simulator):
        self.simulator = simulator

    def schedule(self, new_ready, new_finished):
        return ()


class DoNothingScheduler(SchedulerBase):
    pass


class StaticScheduler(SchedulerBase):

    def init(self, simulator):
        super().init(simulator)
        self.scheduled = False

    def schedule(self, new_ready, new_finished):
        if self.scheduled:
            return ()
        self.scheduled = True
        return self.static_schedule()

    def static_schedule(self):
        raise NotImplementedError()


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
        assign_b_level(self.simulator.task_graph, lambda t: t.duration)
        return [TaskAssignment(worker, task, task.s_info) for task in new_ready]


class QueueScheduler(SchedulerBase):

    def __init__(self):
        self.ready = []
        self.queue = None

    def init(self, simulator):
        super().init(simulator)
        self.queue = self.make_queue()

    def make_queue(self, simulator):
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

        for i in range(len(workers)):
            w = workers[i]
            for inp in task.inputs:
                if w not in inp.info.assigned_workers:
                    costs[i] += inp.size

        return np.random.choice(np.flatnonzero(costs == costs.min()))


class RandomGtScheduler(GreedyTransferQueueScheduler):

    def make_queue(self):
        tasks = self.simulator.task_graph.tasks[:]
        random.shuffle(tasks)
        return tasks


class BlevelGtScheduler(GreedyTransferQueueScheduler):

    def __init__(self, include_size=False):
        super().__init__()
        self.include_size = include_size

    def make_queue(self):
        def cost_fn1(t):
            return t.duration

        def cost_fn2(t):
            if not t.consumers:
                return t.duration
            return t.duration + t.size / bandwidth

        bandwidth = self.simulator.connector.bandwidth
        assign_b_level(self.simulator.task_graph, cost_fn2 if self.include_size else cost_fn1)
        tasks = self.simulator.task_graph.tasks[:]
        random.shuffle(tasks)  # To randomize keys with the same level
        tasks.sort(key=lambda n: n.s_info, reverse=True)
        return tasks


class Camp2Scheduler(StaticScheduler):

    def __init__(self, iterations=2000):
        super().__init__()
        self.iterations = iterations

    def init(self, simulator):
        super().init(simulator)

    def static_schedule(self):
        independencies = compute_independent_tasks(self.simulator.task_graph)
        tab = []
        costs = []
        workers = self.simulator.workers
        cpu_factor = sum([w.cpus for w in workers]) / len(workers) / 8
        for task, indeps in independencies.items():
            if not indeps:
                continue
            task_para_value = task.duration / len(indeps) * task.cpus / cpu_factor
            for t in indeps:
                #if t.id < task.id:
                #    continue
                tab.append((t.id, task.id))
                costs.append(task_para_value)
        self.tab = np.array(tab, dtype=np.int32)
        self.costs = np.array(costs)

        tasks = self.simulator.task_graph.tasks
        placement = np.empty(self.simulator.task_graph.task_count, dtype=np.int32)
        placement[:] = workers.index(max_cpus_worker(workers))
        pcost = self.placement_cost(placement)

        for i in range(self.iterations):
            t = random.randint(0, len(tasks) - 1)
            old_w = placement[t]
            new_w = random.randint(0, len(workers) - 2)
            if new_w >= old_w:
                new_w += 1
            if workers[new_w].cpus < tasks[t].cpus:
                continue
            placement[t] = new_w
            new_pcost = self.placement_cost(placement)
            if new_pcost > pcost:  # and np.random.random() > (i / limit) / 100:
                placement[t] = old_w
            else:
                pcost = new_pcost

        assign_b_level(self.simulator.task_graph, lambda t: t.duration)

        r = [TaskAssignment(workers[w], task, task.s_info)
             for task, w in zip(tasks, placement)]
        return r

    def placement_cost(self, placement):
        s = 0
        bandwidth = self.simulator.connector.bandwidth

        for t in self.simulator.task_graph.tasks:
            p = placement[t.id]
            m = 0
            for inp in t.inputs:
                if p != placement[inp.id] and inp.size > m:
                    m = inp.size
            s += m

        s /= bandwidth

        a = placement[self.tab[:, 0]]
        b = placement[self.tab[:, 1]]
        return (self.costs * (a == b)).sum() + s


class K1hScheduler(SchedulerBase):
    def schedule(self, new_ready, new_finished):
        return schedule_all(self.simulator.workers, new_ready,
                            lambda w, t: self.find_assignment(w, t))

    def find_assignment(self, workers, tasks):
        return min(itertools.product(workers, tasks),
                   key=lambda item: self.calculate_cost(item[0], item[1]))

    def calculate_cost(self, worker, task):
        if task.cpus > worker.cpus:
            return 10e10
        transfer = self.calculate_transfer(worker, task)
        cpu = task.duration
        worker_cost = self.worker_cost(worker)

        return transfer + cpu + worker_cost

    def worker_cost(self, worker):
        return sum(t.duration for t in worker.assigned_tasks)

    def calculate_transfer(self, worker, task):
        bandwidth = self.simulator.connector.bandwidth
        cost = transfer_cost_parallel(worker, task)

        for c in task.consumers:
            for i in c.inputs:
                if i != task and worker not in i.info.assigned_workers:
                    cost += i.size

        return cost / bandwidth


class DLSScheduler(SchedulerBase):
    """
    Implementation of the dynamic level scheduler (DLS) from
    A Compile-Time Scheduling Heuristic
    for Interconnection-Constrained
    Heterogeneous Processor Architectures (1993)

    The scheduler calculates t.b_Level -
    (minimum time when all dependencies of task t are available on worker w)
    for all task-worker pairs (t, w) and selects the maximum.

    :param extended_selection True if extended processor selection
    should be used
    """
    def __init__(self, extended_selection=False):
        self.extended_selection = extended_selection

    def init(self, simulator):
        super().init(simulator)
        assign_b_level(simulator.task_graph, lambda t: t.duration)

    def schedule(self, new_ready, new_finished):
        return schedule_all(self.simulator.workers, new_ready,
                            lambda w, t: self.find_assignment(w, t))

    def find_assignment(self, workers, tasks):
        return max(itertools.product(workers, tasks),
                   key=lambda item: self.calculate_cost(item[0], item[1]))

    def calculate_cost(self, worker, task):
        if task.cpus > worker.cpus:
            return -10e10

        now = self.simulator.env.now
        transfer = self.calculate_transfer(worker, task)

        if self.extended_selection:
            last_finish = now + max([t.remaining_time(now)
                                     for t in worker.running_tasks.values()],
                                    default=0)
            transfer = max(transfer, last_finish)

        return task.s_info - transfer

    def calculate_transfer(self, worker, task):
        return self.simulator.env.now + (transfer_cost_parallel(
            worker, task) / self.simulator.connector.bandwidth)


class LASTScheduler(SchedulerBase):
    """
    Implementation of the LAST scheduler from
    “The LAST Algorithm: A Heuristic-Based Static Task Allocation
    Algorithm (1989)

    The scheduler tries to minimize overall communication by prioriting tasks
    with small sizes and small neighbours.
    """
    def schedule(self, new_ready, new_finished):
        bandwidth = self.simulator.connector.bandwidth
        workers = self.simulator.workers[:]

        def edge_cost(t1, t2):
            return (0 if t1.info.assigned_workers == t2.info.assigned_workers
                    else 1)

        d_nodes = {}
        for task in new_ready:
            if not task.inputs:
                d_nodes[task] = 1
            else:
                input_weighted = sum((i.size / bandwidth) * edge_cost(i, task)
                            for i in task.inputs)
                input = sum((i.size / bandwidth) for i in task.inputs)
                output = sum((task.size / bandwidth) for _ in task.consumers)
                d_nodes[task] = (input_weighted + output) / (input + output)

        def worker_cost(worker, task):
            if task.cpus > worker.cpus:
                return 10e10
            return 0

        schedules = []
        for _ in new_ready[:]:
            m = max(d_nodes, key=lambda t: d_nodes[t])
            worker_costs = [transfer_cost_parallel(w, m) +
                            worker_cost(w, m) for w in workers]
            worker_index = min(range(len(workers)),
                               key=lambda i: worker_costs[i])
            schedules.append(TaskAssignment(workers[worker_index], m))

            del d_nodes[m]
        return schedules


def assign_b_level(task_graph, cost_fn):
    for task in task_graph.tasks:
        task.s_info = cost_fn(task)
    graph_dist_crawl(task_graph.leaf_tasks(), lambda t: t.inputs, cost_fn)


def graph_dist_crawl(initial_tasks, nexts_fn, cost_fn):
    tasks = initial_tasks
    while tasks:
        new_tasks = set()
        for task in tasks:
            dist = task.s_info
            for t in nexts_fn(task):
                new_value = max(t.s_info, dist + cost_fn(t))
                if new_value != t.s_info:
                    t.s_info = new_value
                    new_tasks.add(t)
        tasks = new_tasks


def graph_crawl(initial_tasks, nexts_fn, value_fn):
    values = {}

    def compute(state):
        v = values.get(state)
        if v is not None:
            return v
        v = value_fn(state, [compute(s) for s in nexts_fn(state)])
        values[state] = v
        return v

    for state in initial_tasks:
        compute(state)
    return values


def compute_independent_tasks(task_graph):
    def union(state, values):
        values.append(frozenset((state,)))
        return frozenset.union(*values)

    tasks = frozenset(task_graph.tasks)
    up_deps = graph_crawl(task_graph.leaf_tasks(), lambda t: t.inputs, union)
    down_deps = graph_crawl(task_graph.source_tasks(), lambda t: t.consumers, union)
    #print(down_deps)
    return {task: tasks.difference(up_deps[task] | down_deps[task])
            for task in task_graph.tasks}


def max_cpus_worker(workers):
    return max(workers, key=lambda w: w.cpus)


def transfer_cost_parallel(worker, task):
    """
    Calculates the cost of transferring inputs of `task` to `worker`.
    Assumes parallel download.
    """
    return max((i.size for i in task.inputs
                if worker not in i.info.assigned_workers),
               default=0)


def schedule_all(workers, tasks, get_assignment):
    """
    Schedules all tasks by repeatedly calling `get_assignment`.
    Tasks are removed after being scheduler, workers stay the same.
    """
    schedules = []

    for _ in tasks[:]:
        (w, task) = get_assignment(workers, tasks)
        tasks.remove(task)
        schedules.append(TaskAssignment(w, task))

    return schedules
