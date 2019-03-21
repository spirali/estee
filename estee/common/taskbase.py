from typing import List, Set, Dict


class TaskBase:

    def __init__(self, id: int, inputs: List["DataObjectBase"], outputs: List["DataObjectBase"]):
        self.id = id
        self.inputs = inputs
        self.outputs = outputs

    @property
    def is_leaf(self):
        return all(not o.consumers for o in self.outputs)

    @property
    def pretasks(self):
        return set(o.parent for o in self.inputs)

    def normalize(self):
        inputs = list(set(self.inputs))
        inputs.sort(key=lambda o: o.id)
        self.inputs = inputs

    def consumers(self):
        if not self.outputs:
            return set()
        return set.union(*[o.consumers for o in self.outputs])

    def is_predecessor_of(self, task):
        descendants = set()
        explore = [self]

        while explore:
            new = []
            for t in explore:
                for o in t.outputs:
                    for d in o.consumers:
                        if d in descendants:
                            continue
                        if d == task:
                            return True
                        descendants.add(d)
                        new.append(d)
            explore = new
        return False


class DataObjectBase:

    def __init__(self, id):
        self.id = id
        self.parent: TaskBase = None
        self.consumers: Set[TaskBase] = set()


class TaskGraphBase:

    def __init__(self, tasks: Dict[int, TaskBase] = None,
                 objects: Dict[int, DataObjectBase] = None):
        self.tasks = tasks or {}
        self.objects = objects or {}

    def source_tasks(self):
        return [t for t in self.tasks.values() if not t.inputs]

    def leaf_tasks(self):
        return [t for t in self.tasks.values() if t.is_leaf]

    @property
    def arcs(self):
        for task in self.tasks.values():
            for t in task.inputs:
                yield (task, t)

    def validate(self):
        objects = self.objects
        tasks = self.tasks
        for task_id, task in tasks.items():
            assert task.id == task_id
            task.validate()

            for o in task.inputs:
                assert objects[o.id] == o

            for o in task.outputs:
                assert objects[o.id] == o

        for o in objects.values():
            assert objects[o.id] == o
            assert o.parent.id in tasks
            for c in o.consumers:
                assert c.id in tasks

    def normalize(self):
        for t in self.tasks.values():
            t.normalize()

    def to_dot(self, name, verbose=False):
        stream = ["digraph ", name, " {\n"]

        for task in self.tasks.values():
            label = "{}\\n{}\\n{:.2f}".format(task.label, task.cpus, task.duration)
            stream.append("t{} [shape=oval,label=\"{}\"]\n".format(task.id, label))

        for output in self.objects.values():
            label = "{}\\n{:.2f}".format(output.id, output.size)
            stream.append("o{} [shape=box,label=\"{}\"]\n".format(output.id, label))

        for task in self.tasks.values():
            for o in task.inputs:
                stream.append("o{} -> t{}\n".format(o.id, task.id))
            for o in task.outputs:
                stream.append("t{} -> o{}\n".format(task.id, o.id))

        stream.append("}\n")
        return "".join(stream)

    def write_dot(self, filename):
        dot = self.to_dot("g")
        with open(filename, "w") as f:
            f.write(dot)

    def remove_task(self, task):
        objects = self.objects
        for o in task.outputs:
            del objects[o.id]
            for t in o.consumers:
                t.inputs.remove(o)
        del self.tasks[task.id]
        for o in task.inputs:
            o.consumers.remove(task)

    @property
    def task_count(self):
        return len(self.tasks)

    def __repr__(self):
        return "<{} #t={}>".format(self.__class__.__name__, len(self.tasks))
