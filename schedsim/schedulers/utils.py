from collections import deque
from heapq import heappop, heappush

from ..simulator import TaskAssignment
from ..simulator.runtimeinfo import TaskState


def compute_alap(runtime_state, task_graph, bandwidth):
    """
    Calculates the As-late-as-possible metric.
    """
    def task_size(task):
        return sum(get_size_estimate(runtime_state, o) for o in task.outputs)

    t_level = compute_t_level_duration_size(runtime_state, task_graph, bandwidth)

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


def compute_b_level(task_graph, cost_fn):
    """
    Calculates the B-level (taken from the HLFET algorithm).
    """
    b_level = {}
    for task in task_graph.tasks:
        if task.is_leaf:
            b_level[task] = cost_fn(task, task)
        else:
            b_level[task] = 0

    graph_dist_crawl(b_level,
                     {t: sum(len(o.consumers) for o in t.outputs) for t in task_graph.tasks},
                     lambda t: t.pretasks,
                     lambda task, next: max(b_level[next],
                                            b_level[task] +
                                            cost_fn(next, task)))
    return b_level


def compute_b_level_duration(task_graph):
    return compute_b_level(task_graph,
                           lambda task, next: task.expected_duration or 1)


def compute_b_level_duration_size(runtime_state, task_graph, bandwidth=1):
    return compute_b_level(
        task_graph,
        lambda t, n: get_duration_estimate(t) + largest_transfer(runtime_state, t, n) / bandwidth
    )


def compute_t_level(task_graph, cost_fn):
    """
    Calculates the T-level (the earliest possible time to start the task).
    """
    t_level = {}
    for task in task_graph.tasks:
        t_level[task] = 0

    graph_dist_crawl(t_level,
                     {t: len(t.inputs) for t in task_graph.tasks},
                     lambda t: t.consumers(),
                     lambda task, next: max(t_level[next],
                                            t_level[task] +
                                            cost_fn(task, next)))
    return t_level


def compute_t_level_duration(task_graph):
    return compute_t_level(task_graph,
                           lambda task, next: get_duration_estimate(task))


def compute_t_level_duration_size(runtime_state, task_graph, bandwidth=1):
    return compute_t_level(
        task_graph,
        lambda t, n: get_duration_estimate(t) + largest_transfer(runtime_state, t, n) / bandwidth
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

    tasks = frozenset(task_graph.tasks)
    up_deps = graph_crawl(task_graph.leaf_tasks(), lambda t: t.pretasks, union)
    down_deps = graph_crawl(task_graph.source_tasks(), lambda t: t.consumers(), union)
    return {task: tasks.difference(up_deps[task] | down_deps[task])
            for task in task_graph.tasks}


def max_cpus_worker(workers):
    return max(workers, key=lambda w: w.cpus)


def transfer_cost_parallel(runtime_state, worker, task):
    """
    Calculates the cost of transferring inputs of `task` to `worker`.
    Assumes parallel download.
    """
    return max((get_size_estimate(runtime_state, i) for i in task.inputs
                if worker not in runtime_state.output_info(i).placing),
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


def largest_transfer(runtime_state, task1, task2):
    """
    Returns the largest transferred output from `task1` to `task2`.
    """
    return max((get_size_estimate(runtime_state, o)
                for o in set(task1.outputs).intersection(task2.inputs)),
               default=0)


def get_duration_estimate(task):
    return task.expected_duration if task.expected_duration is not None else 1


def get_size_estimate(runtime_state, output):
    if runtime_state is None or runtime_state.task_info(output.parent).state == TaskState.Finished:
        return output.size
    return output.expected_size if output.expected_size is not None else 1


def worker_estimate_earliest_time(worker, task, now):
    """
    Estimates in how many time units from `now` will `worker` be able to start executing
    the given `task`. Neglects data transfers.
    """
    assert task.cpus <= worker.cpus

    running_tasks = list(worker.running_tasks)

    free_cpus = worker.cpus
    index = 0
    runqueue = []
    for t in running_tasks:
        heappush(runqueue,
                 (worker.running_tasks[t].start_time + (t.expected_duration or 1), index, t))
        index += 1
        free_cpus -= t.cpus
    assignments = deque([a.task for a in worker.assignments if a.task not in running_tasks])

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
