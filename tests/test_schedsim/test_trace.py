from schedsim.common import TaskGraph
from schedsim.communication import SimpleNetModel
from schedsim.simulator.trace import FetchEndTraceEvent, FetchStartTraceEvent, \
    TaskAssignTraceEvent, TaskEndTraceEvent, TaskStartTraceEvent
from .test_utils import do_sched_test, fixed_scheduler


def test_trace_task_assign(plan1):
    assignments = [(i % 2, task, 0) for i, task in enumerate(plan1.tasks)]

    simulator = do_sched_test(plan1, [1, 1], fixed_scheduler(assignments),
                              trace=True, return_simulator=True)

    assignment_events = [e for e in simulator.trace_events if isinstance(e, TaskAssignTraceEvent)]
    assert len(assignment_events) == len(plan1.tasks)
    assert (set((w, t) for (w, t, _) in assignments) ==
            set((e.worker.id, e.task) for e in assignment_events))


def test_trace_task_execution():
    tg = TaskGraph()
    a = tg.new_task(output_size=0, duration=2)
    b = tg.new_task(output_size=0, duration=3)
    b.add_input(a)
    c = tg.new_task(duration=4)
    c.add_input(b)

    simulator = do_sched_test(tg, [1, 1], fixed_scheduler([
        (0, a, 0),
        (1, b, 0),
        (0, c, 0)
    ]), trace=True, return_simulator=True)

    start_events = [e for e in simulator.trace_events if isinstance(e, TaskStartTraceEvent)]
    assert start_events == [
        TaskStartTraceEvent(0, simulator.workers[0], a),
        TaskStartTraceEvent(2, simulator.workers[1], b),
        TaskStartTraceEvent(5, simulator.workers[0], c)
    ]

    end_events = [e for e in simulator.trace_events if isinstance(e, TaskEndTraceEvent)]
    assert end_events == [
        TaskEndTraceEvent(2, simulator.workers[0], a),
        TaskEndTraceEvent(5, simulator.workers[1], b),
        TaskEndTraceEvent(9, simulator.workers[0], c)
    ]


def test_trace_task_fetch():
    tg = TaskGraph()
    a = tg.new_task(output_size=5, duration=2)
    b = tg.new_task(output_size=3, duration=3)
    b.add_input(a)
    c = tg.new_task(duration=4)
    c.add_input(b)

    simulator = do_sched_test(tg, [1, 1], fixed_scheduler([
        (0, a, 0),
        (1, b, 0),
        (0, c, 0)
    ]), netmodel=SimpleNetModel(1), trace=True, return_simulator=True)

    workers = simulator.workers
    fetch_start_events = [e for e in simulator.trace_events if isinstance(e, FetchStartTraceEvent)]
    assert fetch_start_events == [
        FetchStartTraceEvent(2, workers[1], workers[0], a.output),
        FetchStartTraceEvent(10, workers[0], workers[1], b.output),
    ]

    fetch_end_events = [e for e in simulator.trace_events if isinstance(e, FetchEndTraceEvent)]
    assert fetch_end_events == [
        FetchEndTraceEvent(7, workers[1], workers[0], a.output),
        FetchEndTraceEvent(13, workers[0], workers[1], b.output),
    ]
