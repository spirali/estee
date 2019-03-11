
class TaskAssignment:

    __slots__ = ("worker", "task", "priority", "block", "cancelled", "remaining_inputs_count")

    def __init__(self, worker, task, priority=0, block=0):
        assert block <= priority
        self.worker = worker
        self.task = task
        self.priority = priority
        self.block = block
        self.cancelled = False
        self.remaining_inputs_count = None
