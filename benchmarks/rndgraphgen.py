
from schedtk.taskgraph import TaskGraph

import numpy as np

CPUS = np.array([1, 2, 3, 4], dtype=np.int32)
CPUS_P = np.array([0.50, 0.24, 0.02, 0.24])


def random_cpus():
    return np.random.choice(CPUS, p=CPUS_P)


def make_task(graph, cpus=None):
    if cpus is None:
        cpus = random_cpus()
    return graph.new_task(None,
                          duration=np.random.random(), size=np.random.random(), cpus=cpus)


def make_independent_tasks(graph, cpus=None):
    return [make_task(graph, cpus) for _ in range(np.random.randint(1, 20))]


def gen_independent_tasks(graph):
    cpus = random_cpus()
    make_independent_tasks(graph, cpus)
    return graph


def gen_level(graph):
    sources = graph.leaf_tasks()
    counts = np.arange(1, len(sources) + 1)
    w = np.array([1/(i * 1.2 + 1) for i in range(len(counts))])
    p = w / w.sum()

    for task in make_independent_tasks(graph):
        n_inps = np.random.choice(counts, p=p)
        inps = np.random.choice(sources, n_inps, replace=False)
        task.add_inputs(inps)
    return graph


def gen_uniform_level(graph):
    sources = graph.leaf_tasks()
    counts = np.arange(1, len(sources) + 1)
    w = np.array([1/(i * 1.2 + 1) for i in range(len(counts))])
    p = w / w.sum()
    n_inps = np.random.choice(counts, p=p)
    duration = np.random.random()
    size = np.random.random()
    cpus = random_cpus()
    for task in range(np.random.randint(4, 30)):
        task = graph.new_task(size=size, duration=duration, cpus=cpus)
        inps = np.random.choice(sources, n_inps, replace=False)
        task.add_inputs(inps)
        add_noise(task)

    return graph


def gen_random_links(graph):
    if graph.task_count < 2:
        return graph

    for _ in range(int(np.ceil(graph.task_count / 20))):
        t1, t2 = np.random.choice(graph.tasks, 2, replace=False)
        if t1.is_predecessor_of(t2):
            continue
        t1.add_input(t2)
    return graph


def add_noise(task):
    task.size += max(0.000001, np.random.normal(0, task.size / 20))
    task.duration += max(0.000001, np.random.normal(0, task.duration / 20))


def gen_noisy_duplicate(graph):
    graph2 = graph.copy()
    for task in graph2.tasks:
        add_noise(task)
    return TaskGraph.merge([graph, graph2])


actions = [
    (gen_independent_tasks, 10),
    (gen_level, 40),
    (gen_uniform_level, 20),
    (gen_random_links, 10),
    (gen_noisy_duplicate, 5),
    (None, 5)
]


def generate_graph(steps):
    graph = TaskGraph()
    gen_independent_tasks(graph)

    gen_ops, weights = zip(*actions)
    weights = np.array(weights)
    p = weights / weights.sum()
    for i in range(steps):
        op = np.random.choice(gen_ops, p=p)
        if op is None:
            graph2 = generate_graph(i)
            graph = TaskGraph.merge([graph, graph2])
        else:
            graph = op(graph)

    gen_random_links(graph)
    graph.write_dot("/tmp/x.dot")
    return graph