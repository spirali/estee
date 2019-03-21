from collections import deque
from heapq import heappop, heappush
from typing import Callable, List, Dict

from estee.common import TaskGraph, DataObject
from ..common import Task
from ..common.taskbase import DataObjectBase, TaskBase, TaskGraphBase
from ..schedulers.scheduler import Update, SchedulerWorker
from ..schedulers.tasks import SchedulerTaskGraph, SchedulerTask, SchedulerDataObject
from ..simulator import Worker, TaskAssignment
from ..simulator.runtimeinfo import TaskState


def update_worker_occupancy(workers: Dict[int, SchedulerWorker], update: Update):
    for task in update.new_started_tasks:
        worker = workers[task.computed_by.worker_id]
        worker.scheduled_tasks.remove(task)
        worker.running_tasks.add(task)

    for task in update.new_finished_tasks:
        workers[task.computed_by.worker_id].running_tasks.remove(task)


def compute_alap(task_graph: TaskGraphBase, size_resolver: Callable[[DataObjectBase], float],
                 bandwidth: float):
    """
    Calculates the As-late-as-possible metric.
    """

    def task_size(task):
        return sum(size_resolver(o) for o in task.outputs)

    t_level = compute_t_level_duration_size(task_graph, size_resolver, bandwidth)

    alap = {}

    def calc(task):
        if task in alap:
            return alap[task]

        consumers = task.consumers()
        if not consumers:
            value = t_level[task]
        else:
            value = min((calc(t) - task_size(t) / bandwidth
                         for t in consumers),
                        default=t_level[task]) - get_duration_estimate(task)
        alap[task] = value
        return value

    tasks = task_graph.leaf_tasks()
    while tasks:
        new_tasks = set()
        for task in tasks:
            calc(task)
            new_tasks |= set(task.pretasks)
        tasks = new_tasks

    return alap


def compute_b_level(task_graph: TaskGraphBase, cost_fn: Callable[[Task, Task], float]):
    """
    Calculates the B-level (taken from the HLFET algorithm).
    """
    b_level = {}
    for task in task_graph.tasks.values():
        if task.is_leaf:
            b_level[task] = cost_fn(task, task)
        else:
            b_level[task] = 0.0

    graph_dist_crawl(b_level,
                     {t: sum(len(o.consumers) for o in t.outputs)
                      for t in task_graph.tasks.values()},
                     lambda t: t.pretasks,
                     lambda task, next: max(b_level[next],
                                            b_level[task] +
                                            cost_fn(next, task)))
    return b_level


def compute_b_level_duration(task_graph: TaskGraphBase, default_value=30):
    return compute_b_level(task_graph,
                           lambda task, next: get_duration_estimate(task, default_value))


def compute_b_level_duration_size(task_graph: TaskGraphBase,
                                  size_resolver: Callable[[DataObjectBase], float],
                                  bandwidth=1):
    return compute_b_level(
        task_graph,
        lambda t, n: get_duration_estimate(t) + largest_transfer(t, n, size_resolver) / bandwidth
    )


def compute_t_level(task_graph: TaskGraphBase, cost_fn: Callable[[Task, Task], float]):
    """
    Calculates the T-level (the earliest possible time to start the task).
    """
    t_level = {}
    for task in task_graph.tasks.values():
        t_level[task] = 0.0

    graph_dist_crawl(t_level,
                     {t: len(t.inputs) for t in task_graph.tasks.values()},
                     lambda t: t.consumers(),
                     lambda task, next: max(t_level[next],
                                            t_level[task] +
                                            cost_fn(task, next)))
    return t_level


def compute_t_level_duration(task_graph: TaskGraphBase):
    return compute_t_level(task_graph, lambda task, next: get_duration_estimate(task))


def compute_t_level_duration_size(task_graph: TaskGraphBase,
                                  size_resolver: Callable[[DataObjectBase], float],
                                  bandwidth):
    return compute_t_level(
        task_graph,
        lambda t, n: get_duration_estimate(t) + largest_transfer(t, n, size_resolver) / bandwidth
    )


def graph_dist_crawl(values, backlinks, nexts_fn, aggregate):
    tasks = [t for t, v in backlinks.items() if v == 0]
    while tasks:
        new_tasks = set()
        for task in tasks:
            for next in nexts_fn(task):
                values[next] = aggregate(task, next)
                backlinks[next] -= 1
                if backlinks[next] == 0:
                    new_tasks.add(next)
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

    tasks = frozenset(task_graph.tasks.values())
    up_deps = graph_crawl(task_graph.leaf_tasks(), lambda t: t.pretasks, union)
    down_deps = graph_crawl(task_graph.source_tasks(), lambda t: t.consumers(), union)
    return {task: tasks.difference(up_deps[task] | down_deps[task])
            for task in task_graph.tasks.values()}


def max_cpus_worker(workers):
    return max(workers, key=lambda w: w.cpus)


def transfer_cost_parallel(runtime_graph: SchedulerTaskGraph, worker: Worker, task: Task):
    """
    Calculates the cost of transferring inputs of `task` to `worker`.
    Assumes parallel download.
    """
    return max((get_size_estimate_runtime(runtime_graph, i) for i in task.inputs
                if worker not in i.placement),
               default=0)


def schedule_all(workers: List[Worker], tasks: List[Task], get_assignment):
    """
    Schedules all tasks by repeatedly calling `get_assignment`.
    Tasks are removed after being scheduler, workers stay the same.
    """
    schedules = []
    worker_assignments = {}

    for _ in tasks[:]:
        (w, task) = get_assignment(workers, tasks, worker_assignments)
        tasks.remove(task)
        worker_assignments.setdefault(w, []).append(task)
        schedules.append(TaskAssignment(w, task))

    return schedules


def largest_transfer(task1: TaskBase, task2: TaskBase,
                     size_resolver: Callable[[DataObjectBase], float]):
    """
    Returns the largest transferred output from `task1` to `task2`.
    """
    return max((size_resolver(o)
                for o in set(task1.outputs).intersection(task2.inputs)),
               default=0)


def get_duration_estimate(task: Task, default=1):
    return task.expected_duration if task.expected_duration is not None else default


def get_size_estimate_runtime(runtime_graph: SchedulerTaskGraph, output, default=1):
    if runtime_graph.tasks[output.parent.id].state == TaskState.Finished:
        return output.size
    return get_size_estimate(output, default)


def get_size_estimate(output, default=1):
    return output.expected_size if output.expected_size is not None else default


def worker_estimate_earliest_time(worker: SchedulerWorker, task: SchedulerTask,
                                  now: int, worker_assignments=None):
    """
    Estimates in how many time units from `now` will `worker` be able to start executing
    the given `task`. Neglects data transfers.
    """
    assert task.cpus <= worker.cpus

    if worker_assignments is None:
        worker_assignments = []

    running_tasks = worker.running_tasks

    free_cpus = worker.cpus
    index = 0
    runqueue = []
    for t in running_tasks:
        heappush(runqueue, (t.start_time + (t.expected_duration or 1), index, t))
        index += 1
        free_cpus -= t.cpus
    assignments = deque(worker.scheduled_tasks + worker_assignments)

    clock = now
    while free_cpus < task.cpus:
        (finish_time, _, t) = heappop(runqueue)
        clock = finish_time
        free_cpus += t.cpus
        while assignments and free_cpus >= assignments[0].cpus:
            heappush(runqueue,
                     (clock + (assignments[0].expected_duration or 1), index, assignments[0]))
            index += 1
            free_cpus -= assignments[0].cpus
            assignments.popleft()
    return clock - now


def assign_expected_values(graph, duration_estimate=1, size_estimate=1):
    for t in graph.tasks:
        t.duration = t.expected_duration or duration_estimate
        for o in t.outputs:
            o.size = o.expected_size or size_estimate


def topological_sort(graph):
    visited = [False] * graph.task_count
    result = graph.source_tasks()
    next = []

    for t in result:
        visited[t.id] = True
        next += list(t.consumers())

    while len(result) < graph.task_count:
        forward = []
        for t in next:
            if not visited[t.id]:
                visited[t.id] = True
                result.append(t)
                forward += list(t.consumers())
        next = forward

    return result


def estimate_schedule(schedule: List[TaskAssignment], netmodel):
    def transfer_cost_parallel_finished(task_to_worker, worker, task):
        return max((i.size for i in task.inputs
                    if worker != task_to_worker[i.parent]),
                   default=0)

    task_to_worker = {assignment.task: assignment.worker for assignment in schedule}
    tasks = []

    for assignment in schedule:
        tasks.append(assignment.task)
        assignment.worker.scheduled_tasks.append(assignment.task)

    def task_push(time, task, type):
        nonlocal index

        if task.expected_duration == 0:
            task_end(time, task)
        else:
            heappush(events, (time, index, task, task_to_worker[task], type))
            index += 1

    def task_start(time, task, worker):
        nonlocal index
        dta = (transfer_cost_parallel_finished(task_to_worker, worker, task) /
               netmodel.bandwidth)
        rt = worker_estimate_earliest_time(worker, task, time)
        start = time + max(rt, dta)
        finish = start + task.expected_duration
        worker.scheduled_tasks.remove(task)
        task.start_time = start
        worker.running_tasks.add(task)
        task_push(finish, task, "end")

    def task_end(time, task):
        nonlocal index, end

        finished[task.id] = True
        for output in task.outputs:
            output.size = output.expected_size
        for consumer in task.consumers():
            remaining_inputs[consumer] -= 1
            if remaining_inputs[consumer] == 0:
                task_push(time, consumer, "start")
        end = max(end, time)

    events = []
    finished = [False] * len(tasks)
    remaining_inputs = {task: len(task.inputs) for task in tasks}
    index = 0
    end = 0

    for task in tasks:
        if not task.inputs:
            task_push(0, task, "start")

    while events:
        (time, _, task, worker, type) = heappop(events)
        if type == "start":
            task_start(time, task, worker)
        elif type == "end":
            worker.running_tasks.remove(task)
            task_end(time, task)

    return end


def create_scheduler_graph(graph: TaskGraph) -> SchedulerTaskGraph:
    def scheduler_object(obj: DataObject) -> SchedulerDataObject:
        return SchedulerDataObject(obj.id, obj.expected_size, obj.size)

    def scheduler_task(task: Task, objects: Dict[int, SchedulerDataObject]) -> SchedulerTask:
        return SchedulerTask(task.id,
                             [objects[o.id] for o in task.inputs],
                             [objects[o.id] for o in task.outputs],
                             task.expected_duration,
                             task.cpus)

    objects = {o.id: scheduler_object(o) for t in graph.tasks.values() for o in t.outputs}
    tasks = [scheduler_task(task, objects) for task in graph.tasks.values()]

    for task in tasks:
        for output in task.outputs:
            output.parent = task
        for input in task.inputs:
            input.consumers.add(task)

    return SchedulerTaskGraph({t.id: t for t in tasks}, objects)
