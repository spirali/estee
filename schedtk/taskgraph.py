
from .task import Task


class TaskGraph:

    def __init__(self, tasks=None):
        if tasks is None:
            self.tasks = []
        else:
            for i, task in enumerate(tasks):
                task.id = i
            self.tasks = tasks

    def _copy_tasks(self):
        tasks = [task.simple_copy() for task in self.tasks]
        for old_task, task in zip(self.tasks, tasks):
            for inp in old_task.inputs:
                task.add_input(tasks[inp.id])
        return tasks

    def copy(self):
        return TaskGraph(tasks=self._copy_tasks())

    @property
    def task_count(self):
        return len(self.tasks)

    def cleanup(self):
        for task in self.tasks:
            task.cleanup()

    def new_task(self, name=None, duration=1, size=0, cpus=1):
        task = Task(name, duration, size, cpus)
        task.id = len(self.tasks)
        self.tasks.append(task)
        return task

    def source_tasks(self):
        return [t for t in self.tasks if not t.inputs]

    def leaf_tasks(self):
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

    def to_dot(self, name, verbose=False):
        stream = ["digraph ", name, " {\n"]

        for task in self.tasks:
            label = "{}\\n{}\\n{:.2f}/{:.2f}".format(task.label, task.cpus, task.duration, task.size)
            stream.append("v{} [shape=box,label=\"{}\"]\n".format(task.id, label))
            for t in task.inputs:
                stream.append("v{} -> v{}\n".format(t.id, task.id))
        stream.append("}\n")
        return "".join(stream)

    def write_dot(self, filename):
        dot = self.to_dot("g")
        with open(filename, "w") as f:
            f.write(dot)

    def __repr__(self):
        return "<TaskGraph #t={}>".format(len(self.tasks))

    @staticmethod
    def merge(task_graphs):
        tasks = sum((tg._copy_tasks() for tg in task_graphs), [])
        return TaskGraph(tasks=tasks)