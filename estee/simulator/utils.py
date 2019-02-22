from heapq import heappop, heappush

from ..schedulers.utils import worker_estimate_earliest_time
from ..worker.worker import RunningTask


def estimate_schedule(schedule, graph, netmodel):
    def transfer_cost_parallel_finished(task_to_worker, worker, task):
        return max((i.size for i in task.inputs
                    if worker != task_to_worker[i.parent]),
                   default=0)

    task_to_worker = {assignment.task: assignment.worker for assignment in schedule}

    def task_push(time, task, type):
        nonlocal index

        if task.expected_duration == 0:
            task_end(time, task)
        else:
            heappush(events, (time, index, task, task_to_worker[task], type))
            index += 1

    def task_start(time, task, worker):
        nonlocal index
        dta = (transfer_cost_parallel_finished(task_to_worker, worker, task) /
               netmodel.bandwidth)
        rt = worker_estimate_earliest_time(worker, task, time)
        start = time + max(rt, dta)
        finish = start + task.expected_duration
        worker.running_tasks[task] = RunningTask(task, start)
        task_push(finish, task, "end")

    def task_end(time, task):
        nonlocal index, end

        finished[task.id] = True
        for consumer in task.consumers():
            remaining_inputs[consumer] -= 1
            if remaining_inputs[consumer] == 0:
                task_push(time, consumer, "start")
        end = max(end, time)

    events = []
    finished = [False] * graph.task_count
    remaining_inputs = {task: len(task.inputs) for task in graph.tasks}
    index = 0
    end = 0

    for task in graph.tasks:
        if not task.inputs:
            task_push(0, task, "start")

    while events:
        (time, _, task, worker, type) = heappop(events)
        if type == "start":
            task_start(time, task, worker)
        elif type == "end":
            del worker.running_tasks[task]
            task_end(time, task)

    return end
