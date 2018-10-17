
from .task import Task


class TaskGraph:

    def __init__(self):
        self.tasks = []
        self._id_counter = 0

    @property
    def task_count(self):
        return len(self.tasks)

    def cleanup(self):
        for task in self.tasks:
            task.cleanup()

    def new_task(self, name=None, duration=1, size=0, cpus=1):
        task = Task(name, duration, size, cpus)
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

    def to_dot(self, name, verbose=False):
        stream = ["digraph ", name, " {\n"]

        for task in self.tasks:
            label = "{}\\n{:.2f}/{:.2f}".format(task.label, task.duration, task.size)
            stream.append("v{} [label=\"{}\"]\n".format(task.id, label))
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
