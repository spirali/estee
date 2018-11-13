import random


def random_dependencies(count: int, edge_density: float, task_fn):
    """
    Creates a complete graph with the given edge density.
    """

    nodes = [task_fn() for i in range(count)]

    for n1 in nodes:
        for n2 in nodes:
            if (n1 == n2 or random.random() > edge_density
                    or n1.is_predecessor_of(n2)):
                continue
            n1.add_input(random.choice(n2.outputs))


def random_levels(counts, inputs, task_fn):
    """
        Counts - number of tasks in each of level, it may be an integer or
        range (min, max)
        Inputs - number of inputs for each level, it may be an integer or
        range (min, max)
    """
    prev = None
    for count, inps in zip(counts, inputs):
        if isinstance(count, tuple):
            count = random.randint(count[0], count[1])

        level = [task_fn() for _ in range(count)]
        for task in level:
            if inps:
                if isinstance(inps, tuple):
                    inps = random.randint(min(len(prev), inps[0]),
                                          min(len(prev), inps[1]))
                task.add_inputs(random.sample(prev, inps))
        prev = sum((task.outputs for task in level), ())