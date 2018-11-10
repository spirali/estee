from ..simulator import TaskAssignment


def compute_alap(task_graph, bandwidth):
    """
    Calculates the As-late-as-possible metric.
    """
    t_level = compute_t_level(task_graph,
                              lambda t: t.duration + t.size / bandwidth)

    alap = {}

    def calc(task):
        if task in alap:
            return alap[task]

        if not task.consumers:
            value = t_level[task]
        else:
            value = min((calc(t) - task.size / bandwidth
                        for t in task.consumers),
                        default=t_level[task]) - task.duration
        alap[task] = value
        return value

    tasks = task_graph.leaf_tasks()
    while tasks:
        new_tasks = set()
        for task in tasks:
            calc(task)
            new_tasks |= set(task.inputs)
        tasks = new_tasks

    return alap


def compute_b_level(task_graph, cost_fn):
    """
    Calculates the B-level (taken from the HLFET algorithm).
    """
    b_level = {}
    for task in task_graph.tasks:
        b_level[task] = cost_fn(task)

    graph_dist_crawl(b_level,
                     task_graph.leaf_tasks(),
                     lambda t: t.pretasks,
                     lambda task, next: max(b_level[next],
                                            b_level[task] + cost_fn(next)))
    return b_level


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
                                            t_level[task] + cost_fn(task)))
    return t_level


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
    #print(down_deps)
    return {task: tasks.difference(up_deps[task] | down_deps[task])
            for task in task_graph.tasks}


def max_cpus_worker(workers):
    return max(workers, key=lambda w: w.cpus)


def transfer_cost_parallel(simulator, worker, task):
    """
    Calculates the cost of transferring inputs of `task` to `worker`.
    Assumes parallel download.
    """
    return max((i.size for i in task.inputs
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
