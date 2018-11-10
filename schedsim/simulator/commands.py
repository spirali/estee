
class TaskAssignment:

    __slots__ = ("worker", "task", "priority")

    def __init__(self, worker, task, priority=0):
        self.worker = worker
        self.task = task
        self.priority = priority
