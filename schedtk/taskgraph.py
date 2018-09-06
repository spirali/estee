
from .task import Task


class TaskGraph:

    def __init__(self):
        self.tasks = []
        self._id_counter = 0

    @property
    def task_count(self):
        return len(self.tasks)

    def new_task(self, name=None, duration=1):
        task = Task(name, duration)
        task.id = self._id_counter
        self._id_counter += 1
        self.tasks.append(task)
        return task

    def source_nodes(self):
        return [t for t in self.tasks if not t.inputs]

    def leaf_nodes(self):
        return [t for t in self.tasks if not t.consumers]

    @property
    def arcs(self):
        for task in self.tasks:
            for t in task.inputs:
                yield (task, t)

    def validate(self):
        for i, task in enumerate(self.tasks):
            assert task.id == i
            task.validate()

            for t in task.inputs:
                assert t in self.tasks

            for t in task.consumers:
                assert t in self.tasks