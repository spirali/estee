

class Task:

    __slots__ = ("inputs", "outputs", "duration", "name", "id", "cpus", "pretasks")

    def __init__(self, name=None, outputs=(), duration=1, cpus=1, output_size=None):
        assert cpus >= 0
        assert duration >= 0

        self.inputs = []
        self.pretasks = []

        if output_size is not None:
            if outputs:
                raise Exception("Cannot set 'output_size' and 'outputs' at once")
            self.outputs = (TaskOutput(self, output_size),)
        else:
            self.outputs = tuple(TaskOutput(self, size) for size in outputs)

        self.name = name
        self.id = None

        self.duration = duration
        self.cpus = cpus

    def simple_copy(self):
        t = Task(self.name, duration=self.duration, cpus=self.cpus)
        t.outputs = [TaskOutput(t, o.size) for o in self.outputs]
        return t

    @property
    def output(self):
        outputs = self.outputs
        if not outputs:
            raise Exception("Task {} has no output", self)
        if len(outputs) > 1:
            raise Exception("Task {} has no unique output", self)
        return outputs[0]

    def consumers(self):
        if not self.outputs:
            return set()
        return set.union(*[o.consumers for o in self.outputs])

    @property
    def label(self):
        if self.name:
            return self.name
        else:
            return "id={}".format(self.id)

    def add_input(self, output):
        if isinstance(output, Task):
            output = output.output
        elif not isinstance(output, TaskOutput):
            raise Exception("Only 'Task' or 'TaskInstance' is expected, not {}".format(repr(output)))
        self.inputs.append(output)
        if output.parent not in self.pretasks:
            self.pretasks.append(output.parent)
        output.consumers.add(self)



    def add_inputs(self, tasks):
        for t in tasks:
            self.add_input(t)

    def add_dependancy(self, task):
        assert isinstance(task, Task)
        self.dependancies.append(task)

    def __repr__(self):
        if self.name:
            name = " '" + self.name + "'"
        else:
            name = ""

        if self.cpus != 1:
            cpus = " c={}".format(self.cpus)
        else:
            cpus = ""

        return "<T{}{} id={}>".format(name, cpus, self.id)

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

    def validate(self):
        assert self.duration >= 0
        assert not self.is_predecessor_of(self)
        for o in self.outputs:
            assert o.parent == self
            assert o.size >= 0


class TaskOutput:

    __slots__ = ("parent", "id", "size", "consumers")

    def __init__(self, parent, size):
        assert size >= 0

        self.parent = parent
        self.size = size
        self.consumers = set()
        self.id = None

    def __repr__(self):
        return "<O id={} p={} size={}>".format(self.id, repr(self.parent), self.size)
