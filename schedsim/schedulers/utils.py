from schedsim.simulator.runtimeinfo import TaskState
from ..simulator import TaskAssignment


def compute_alap(simulator, task_graph, bandwidth):
    """
    Calculates the As-late-as-possible metric.
    """
    def task_size(task):
        return sum(get_size_estimate(simulator, o) for o in task.outputs)

    t_level = compute_t_level_duration_size(simulator, task_graph, bandwidth)

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
                     task_graph.leaf_tasks(),
                     lambda t: t.pretasks,
                     lambda task, next: max(b_level[next],
                                            b_level[task] +
                                            cost_fn(next, task)))
    return b_level


def compute_b_level_duration(task_graph):
    return compute_b_level(task_graph,
                           lambda task, next: task.expected_duration or 1)


def compute_b_level_duration_size(simulator, task_graph, bandwidth=1):
    return compute_b_level(
        task_graph,
        lambda t, n: get_duration_estimate(t) + largest_transfer(simulator, t, n) / bandwidth
    )


def compute_t_level(task_graph, cost_fn):
    """
    Calculates the T-level (the earliest possible time to start the task).
    """
    t_level = {}
    for task in task_graph.tasks:
        t_level[task] = 0

    graph_dist_crawl(t_level,
                     task_graph.source_tasks(),
                     lambda t: t.consumers(),
                     lambda task, next: max(t_level[next],
                                            t_level[task] +
                                            cost_fn(task, next)))
    return t_level


def compute_t_level_duration(task_graph):
    return compute_t_level(task_graph,
                           lambda task, next: get_duration_estimate(task))


def compute_t_level_duration_size(simulator, task_graph, bandwidth=1):
    return compute_t_level(
        task_graph,
        lambda t, n: get_duration_estimate(t) + largest_transfer(simulator, t, n) / bandwidth
    )


def graph_dist_crawl(values, initial_tasks, nexts_fn, aggregate):
    tasks = initial_tasks
    while tasks:
        new_tasks = set()
        for task in tasks:
            for next in nexts_fn(task):
                new_value = aggregate(task, next)
                if new_value != values[next]:
                    values[next] = new_value
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


def transfer_cost_parallel(simulator, worker, task):
    """
    Calculates the cost of transferring inputs of `task` to `worker`.
    Assumes parallel download.
    """
    return max((get_size_estimate(simulator, i) for i in task.inputs
                if worker not in simulator.output_info(i).placing),
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


def largest_transfer(simulator, task1, task2):
    """
    Returns the largest transferred output from `task1` to `task2`.
    """
    return max((get_size_estimate(simulator, o)
                for o in set(task1.outputs).intersection(task2.inputs)),
               default=0)


def get_duration_estimate(task):
    return task.expected_duration if task.expected_duration is not None else 1


def get_size_estimate(simulator, output):
    if simulator is None or simulator.task_info(output.parent).state == TaskState.Finished:
        return output.size
    return output.expected_size if output.expected_size is not None else 1
