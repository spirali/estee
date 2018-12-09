import numpy as np

from ..common import TaskGraph, TaskOutput


class Generator:

    CPUS = None
    CPUS_P = None
    DURATIONS = None
    SIZES = None

    def rnd_duration(self):
        return float(np.random.choice(self.DURATIONS))

    def rnd_size(self):
        return float(np.random.choice(self.SIZES))

    def random_cpus(self):
        return int(np.random.choice(self.CPUS, p=self.CPUS_P))


class SGen(Generator):
    CPUS = np.array([1, 2, 3, 4], dtype=np.int32)
    CPUS_P = np.array([0.50, 0.24, 0.02, 0.24])

    DURATIONS = [1, 6, 30, 60, 180, 320, 600]
    SIZES = [0.01, 1, 10, 20, 50, 100, 200, 320, 640, 1000, 2500, 16000, 32000]


class MGen(Generator):
    CPUS = np.array([1, 4, 16], dtype=np.int32)
    CPUS_P = np.array([0.65, 0.1, 0.25])

    DURATIONS = [1, 6, 30, 60, 180, 320, 600]
    SIZES = [0.01, 1, 10, 20, 50, 100, 200, 320, 640, 1000, 2500, 16000, 32000]


def make_task(graph, gen, cpus=None):
    if cpus is None:
        cpus = gen.random_cpus()
    d = gen.rnd_duration()

    if np.random.random() < 0.98:
        n_outputs = 1
    else:
        n_outputs = np.random.randint(1, 20)

    outputs = []
    for _ in range(n_outputs):
        s = gen.rnd_size()
        output = TaskOutput(np.random.normal(s, s / 10), s)
        outputs.append(output)

    return graph.new_task(None,
                          duration=np.random.normal(d, d / 10),
                          expected_duration=d,
                          outputs=outputs,
                          cpus=cpus)


def make_independent_tasks(graph, gen, cpus=None):
    return [make_task(graph, gen, cpus) for _ in range(np.random.randint(1, 20))]


def gen_independent_tasks(graph, gen):
    cpus = gen.random_cpus()
    make_independent_tasks(graph, gen, cpus)
    return graph


def gen_level(graph, gen):
    sources = graph.leaf_tasks()
    counts = np.arange(1, len(sources) + 1)
    w = np.array([1/(i * 1.2 + 1) for i in range(len(counts))])
    p = w / w.sum()

    for task in make_independent_tasks(graph, gen):
        n_inps = np.random.choice(counts, p=p)
        inps = np.random.choice(sources, n_inps, replace=False)
        inps = list(set(np.random.choice(t.outputs) for t in inps))
        task.add_inputs(inps)
    return graph


def gen_uniform_level(graph, gen):
    sources = graph.leaf_tasks()
    counts = np.arange(1, len(sources) + 1)
    w = np.array([1/(i * 1.2 + 1) for i in range(len(counts))])
    p = w / w.sum()
    n_inps = np.random.choice(counts, p=p)
    duration = gen.rnd_duration()
    size = gen.rnd_size()
    cpus = gen.random_cpus()
    if np.random.random() < 0.5:
        sz = np.random.randint(4, 50)
    else:
        sz = np.random.randint(0, sum(len(t.outputs) for t in sources))
    for task in range(sz):
        task = graph.new_task(duration=np.random.normal(duration, duration / 15),
                              expected_duration=duration,
                              output_size=size,
                              cpus=cpus)
        inps = np.random.choice(sources, n_inps, replace=False)
        inps = list(set(np.random.choice(t.outputs) for t in inps))
        task.add_inputs(inps)
    return graph


def gen_random_links(graph, gen):
    if graph.task_count < 2:
        return graph

    for _ in range(int(np.ceil(graph.task_count / 20))):
        t1, t2 = np.random.choice(graph.tasks, 2, replace=False)
        if t1.is_predecessor_of(t2):
            continue
        t1.add_input(np.random.choice(t2.outputs))
    return graph


def add_noise(task):
    for o in task.outputs:
        o.size += max(0.000001, np.random.normal(0, o.size / 30))
    task.duration += max(0.000001, np.random.normal(0, task.duration / 20))


def gen_noisy_duplicate(graph, gen):
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


def generate_randomized_graph(gen, steps):
    graph = TaskGraph()
    gen_independent_tasks(graph, gen)

    gen_ops, weights = zip(*actions)
    weights = np.array(weights)
    p = weights / weights.sum()
    for i in range(steps):
        op = np.random.choice(gen_ops, p=p)
        if op is None:
            graph2 = generate_randomized_graph(gen, i)
            graph = TaskGraph.merge([graph, graph2])
        else:
            graph = op(graph, gen)

    gen_random_links(graph, gen)
    graph.normalize()
    return graph
