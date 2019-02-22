
from enum import Enum


class TaskState(Enum):
    Waiting = 1
    Assigned = 2
    Finished = 3


class TaskRuntimeInfo:

    __slots__ = ("state",
                 "assign_time",
                 "end_time",
                 "assigned_workers",
                 "unfinished_inputs")

    def __init__(self, task):
        self.state = TaskState.Waiting
        self.assign_time = None
        self.end_time = None
        self.assigned_workers = []
        self.unfinished_inputs = len(task.inputs)

    @property
    def is_running(self):
        return self.state == TaskState.Running

    @property
    def is_ready(self):
        return self.unfinished_inputs == 0

    @property
    def is_finished(self):
        return self.state == TaskState.Finished

    @property
    def is_waiting(self):
        return self.state == TaskState.Waiting


class OutputRuntimeInfo:

    __slots__ = ("placing")

    def __init__(self, output):
        self.placing = []


class RuntimeState:

    def __init__(self, task_graph):
        self.task_infos = [TaskRuntimeInfo(task)
                           for task in task_graph.tasks]
        self.output_infos = [OutputRuntimeInfo(
            task) for task in task_graph.outputs]

    def task_info(self, task):
        return self.task_infos[task.id]

    def output_info(self, output):
        return self.output_infos[output.id]
