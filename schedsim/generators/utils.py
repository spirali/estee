import collections

import numpy as np


def normal(loc, scale):
    return max(0.0000001, np.random.normal(loc, scale))


def exponential(scale):
    return max(0.0000001, np.random.exponential(scale))


def listify(value):
    if not isinstance(value, collections.Iterable):
        return (value, )
    return value


def gen_level(tg, width, name, cost_fn, output_fn):
    return [tg.new_task(name=name.format(i),
                        duration=cost_fn(),
                        outputs=listify(output_fn())) for i in range(width)]


def join_level(parents, children):
    for (p, c) in zip(parents, children):
        c.add_input(p.outputs[0])
