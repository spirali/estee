from enum import IntEnum
from typing import Dict, List

from ..common.taskbase import TaskBase, DataObjectBase, TaskGraphBase


class TaskState(IntEnum):
    Waiting = 1
    Assigned = 2
    Finished = 3


class SchedulerTask(TaskBase):

    def __init__(self, id: int, inputs: List["SchedulerDataObject"],
                 outputs: List["SchedulerDataObject"],
                 expected_duration: float, cpus: int):
        super().__init__(id, inputs, outputs)
        unfinished_inputs = 0
        for o in inputs:
            if not o.placement:
                unfinished_inputs += 1
        self.unfinished_inputs = unfinished_inputs
        self.state = TaskState.Waiting
        self.cpus = cpus
        self.expected_duration = expected_duration
        self.scheduled_worker = None
        self.computed_by = None
        self.running = False
        self.start_time: float = None

    @property
    def is_waiting(self):
        return self.state == TaskState.Waiting

    def simple_copy(self):
        inputs = [o.simple_copy() for o in self.inputs]
        outputs = [o.simple_copy() for o in self.outputs]
        task = SchedulerTask(self.id, inputs, outputs,
                             expected_duration=self.expected_duration,
                             cpus=self.cpus)
        for o in outputs:
            o.parent = task
        for o in inputs:
            o.consumers.add(task)
        return task

    def __repr__(self):
        return "SchedulerTask(id={}, cpus={}, expected_duration={})".format(
            self.id, self.cpus, self.expected_duration)


class SchedulerDataObject(DataObjectBase):

    def __init__(self, id, expected_size, size=None):
        super().__init__(id)
        self.placement = ()
        self.availability = ()
        self.scheduled = set()
        self.expected_size = expected_size
        self.size = size

    def simple_copy(self):
        return SchedulerDataObject(self.id, self.expected_size, self.size)

    def __repr__(self):
        return "SchedulerObject(id={}, expected_size={}, size={})".format(
            self.id, self.expected_size, self.size)


class SchedulerTaskGraph(TaskGraphBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks: Dict[int, SchedulerTask] = self.tasks
        self.objects: Dict[int, SchedulerDataObject] = self.objects
