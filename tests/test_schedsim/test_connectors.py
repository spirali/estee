import random

import numpy as np
import pytest
import simpy
from numpy.testing import assert_array_equal

from schedsim.communication.netmodels import compute_maxmin_flow, \
    MaxMinFlowNetModel, SimpleNetModel
from schedsim.worker import Worker


def test_maxmin_flow():

    def mm_flow(send_capacities, recv_capacities, connections):
        return compute_maxmin_flow(
            np.array(send_capacities, dtype=np.float),
            np.array(recv_capacities, dtype=np.float),
            np.array(connections, dtype=np.int32))

    assert_array_equal(np.array([[0.5], [0.5]]),
                       mm_flow([1, 1], [1], [[1], [1]]))

    assert_array_equal(np.array([[0.5, 0.5]]),
                       mm_flow([1], [1, 1], [[1, 1]]))

    assert_array_equal(np.array([[0.5, 0], [0.5, 0]]),
                       mm_flow([1, 1], [1, 1], [[1, 0], [1, 0]]))

    assert_array_equal(np.array([[0.5, 0.5], [0.5, 0]]),
                       mm_flow([1, 1], [1, 1], [[1, 1], [1, 0]]))

    assert_array_equal(np.array([[0.5, 0.25], [0.5, 0]]),
                       mm_flow([1, 1], [1, 0.25], [[1, 1], [1, 0]]))

    assert_array_equal(np.array([[0.25, 0.25, 0.25, 0.25]] * 4),
                       mm_flow([1, 1, 1, 1], [1, 1, 1, 1], [[1, 1, 1, 1]] * 4))

    assert_array_equal(np.array([[0.1, 0.1, 0.1, 0.1],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.25, 0.25, 0.25, 0.25],
                                [0.0, 0.0, 0.0, 0.4]]),
                       mm_flow([0.4, 1, 1, 1], [1, 1, 0.8, 1], [[1, 1, 1, 1]]
                               * 3 + [[0, 0, 0, 1]]))

    assert_array_equal(np.diag([0.1, 0.2, 0.2, 0.1]),
                       mm_flow([0.1, 0.2, 0.3, 0.4], [1, 0.2, 0.2, 0.1],
                               np.eye(4, dtype=np.int32)))


def create_connector(cclass=MaxMinFlowNetModel):
    env = simpy.Environment()
    workers = [Worker() for _ in range(4)]
    for i, w in enumerate(workers):
        w.id = i
    connector = cclass(100)
    connector.init(env, workers)
    return connector, env, workers


def test_maxmin_connector_simple():
    connector, env, workers = create_connector()
    d = connector.download(workers[0], workers[1], 200)
    env.run(d)
    assert env.now == pytest.approx(2.0)

    d1 = connector.download(workers[0], workers[1], 200)
    d2 = connector.download(workers[2], workers[3], 300)
    env.run(d1 & d2)
    assert env.now == pytest.approx(5.0)

    d1 = connector.download(workers[0], workers[1], 200)
    d2 = connector.download(workers[0], workers[1], 300)
    env.run(d1)
    assert env.now == pytest.approx(9.0)

    env.run(d2)
    assert env.now == pytest.approx(10.0)


def test_maxmin_connector_mix():
    connector, env, workers = create_connector()
    d1 = connector.download(workers[0], workers[1], 200)
    # d1 200
    env.run(env.timeout(1))
    assert env.now == pytest.approx(1.0)
    d2 = connector.download(workers[0], workers[2], 200)
    d3 = connector.download(workers[3], workers[1], 1000)
    # env.run(env.timeout(1))
    # d1 100; d2 200; d3 1000
    env.run(d1)
    assert env.now == pytest.approx(3.0)
    # d2 100; d3 900
    env.run(env.timeout(0.5))
    assert env.now == pytest.approx(3.5)
    d4 = connector.download(workers[0], workers[1], 100)
    d5 = connector.download(workers[0], workers[1], 100)

    env.run(d2)
    assert env.now == pytest.approx(4.5)

    env.run(d4)
    assert env.now == pytest.approx(7.5)

    env.run(d5)
    assert env.now == pytest.approx(7.5)

    env.run(d3)
    assert env.now == pytest.approx(14)


def test_maxmin_connector():

    random.seed(42)
    COUNT = 50

    for i in range(10):
        ids = list(range(4))
        pairs = [random.sample(ids, 2) for i in range(COUNT)]
        sizes = [random.random() * 200 + 0.00001 for i in range(COUNT)]
        diffs = [random.random() / 10.0 + 0.00001 for i in range(COUNT)]

        events = []
        connector, env, workers = create_connector()
        for p, s, d in zip(pairs, sizes, diffs):
            ev = connector.download(workers[p[0]], workers[p[1]], s)
            env.run(env.timeout(d))
            events.append(ev)
        env.run(env.all_of(events))
        tm1 = env.now

        events = []
        connector, env, workers = create_connector(SimpleNetModel)
        for p, s, d in zip(pairs, sizes, diffs):
            ev = connector.download(workers[p[0]], workers[p[1]], s)
            env.run(env.timeout(d))
            events.append(ev)
        env.run(env.all_of(events))
        tm2 = env.now

        assert tm1 > tm2
        assert tm1 < sum(diffs) + sum(sizes) / connector.bandwidth
