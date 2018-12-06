from heapq import heappop, heappush

from ..schedulers.utils import worker_estimate_earliest_time
from ..worker.worker import RunningTask


def estimate_schedule(schedule, graph, netmodel):
    def transfer_cost_parallel_finished(task_to_worker, worker, task):
        return max((i.size for i in task.inputs
                    if worker != task_to_worker[i.parent]),
                   default=0)

    task_to_worker = {assignment.task: assignment.worker for assignment in schedule}
    finish_time = {t: 0 for t in graph.tasks}
    start_time = dict(finish_time)

    events = []
    index = 0
    for task in graph.tasks:
        if not task.inputs:
            heappush(events, (0, index, task, task_to_worker[task], "start"))
            index += 1

    finished = [False] * graph.task_count
    remaining_inputs = {task: len(task.inputs) for task in graph.tasks}

    while events:
        (time, _, task, worker, type) = heappop(events)
        if type == "start":
            dta = (transfer_cost_parallel_finished(task_to_worker, worker, task) /
                   netmodel.bandwidth)
            rt = worker_estimate_earliest_time(worker, task, time)
            start_time[task] = time + max(rt, dta)
            finish_time[task] = start_time[task] + task.expected_duration
            worker.running_tasks[task] = RunningTask(task, start_time[task])
            heappush(events, (finish_time[task], index, task, worker, "end"))
            index += 1
        elif type == "end":
            del worker.running_tasks[task]
            finished[task.id] = True
            for consumer in task.consumers():
                remaining_inputs[consumer] -= 1
                if remaining_inputs[consumer] == 0:
                    heappush(events, (time, index, consumer, task_to_worker[consumer], "start"))
                    index += 1

    return max(finish_time.values())
