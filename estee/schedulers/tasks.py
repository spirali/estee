from ..common.taskbase import TaskBase, DataObjectBase, TaskGraphBase
from ..simulator.runtimeinfo import TaskState


class SchedulerTask(TaskBase):

    def __init__(self, id, inputs, outputs, expected_duration, cpus):
        super().__init__(id, inputs, outputs)
        unfinished_inputs = 0
        for o in inputs:
            if not o.placement:
                unfinished_inputs += 1
        self.unfinished_inputs = unfinished_inputs
        self.state = TaskState.Waiting
        self.cpus = cpus
        self.expected_duration = expected_duration
        self.assigned_worker = None

    @property
    def is_waiting(self):
        return self.state == TaskState.Waiting


class SchedulerDataObject(DataObjectBase):

    def __init__(self, id, expected_size, size=None):
        super().__init__(id)
        self.placement = ()
        self.availability = ()
        self.scheduled = set()
        self.expected_size = expected_size
        self.size = size


class SchedulerTaskGraph(TaskGraphBase):
    pass