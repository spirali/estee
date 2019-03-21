from enum import IntEnum


class TaskState(IntEnum):
    Waiting = 1
    Assigned = 2
    Finished = 3


class TaskRuntimeInfo:

    __slots__ = ("state",
                 "assign_time",
                 "end_time",
                 "assigned_workers",
                 "running_at_workers",
                 "unfinished_inputs")

    def __init__(self, task):
        self.state = TaskState.Waiting
        self.assign_time = None
        self.end_time = None
        self.assigned_workers = []
        self.running_at_workers = []
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


class ObjectRuntimeInfo:

    __slots__ = ("placing", "availability")

    def __init__(self, output):
        self.placing = []
        self.availability = []


class RuntimeState:

    def __init__(self, task_graph):
        self.task_infos = {task.id: TaskRuntimeInfo(task)
                           for task in task_graph.tasks.values()}
        self.object_infos = {obj.id: ObjectRuntimeInfo(obj)
                             for obj in task_graph.objects.values()}

    def task_info(self, task):
        return self.task_infos[task.id]

    def object_info(self, output):
        return self.object_infos[output.id]
