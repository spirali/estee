

class Task:

    __slots__ = ("inputs", "outputs", "duration", "expected_duration", "name", "id", "cpus")

    def __init__(self, name=None,
                 outputs=(),
                 duration=1,
                 cpus=1,
                 output_size=None,
                 expected_duration=None):
        assert cpus >= 0
        assert duration >= 0
        assert expected_duration is None or expected_duration >= 0

        self.inputs = []

        if output_size is not None:
            if outputs:
                raise Exception("Cannot set 'output_size' and 'outputs' at once")
            self.outputs = (TaskOutput(output_size, output_size),)
        else:
            self.outputs = tuple(TaskOutput(s, s) if (isinstance(s, float) or isinstance(s, int))
                                 else s for s in outputs)

        for output in self.outputs:
            assert output.parent is None
            output.parent = self

        self.name = name
        self.id = None

        self.duration = duration
        self.expected_duration = expected_duration
        self.cpus = cpus

    def simple_copy(self):
        t = Task(self.name, duration=self.duration, cpus=self.cpus)
        t.outputs = [TaskOutput(o.size, o.expected_size) for o in self.outputs]
        for o in t.outputs:
            o.parent = t
        return t

    @property
    def is_leaf(self):
        return all(not o.consumers for o in self.outputs)

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

    @property
    def pretasks(self):
        return set(o.parent for o in self.inputs)

    def add_input(self, output):
        if isinstance(output, Task):
            output = output.output
        elif not isinstance(output, TaskOutput):
            raise Exception("Only 'Task' or 'TaskInstance' is expected, not {}"
                            .format(repr(output)))
        self.inputs.append(output)
        output.consumers.add(self)

    def add_inputs(self, tasks):
        for t in tasks:
            self.add_input(t)

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
        assert self.expected_duration is None or self.expected_duration >= 0
        assert not self.is_predecessor_of(self)
        for o in self.outputs:
            assert o.parent == self
            assert o.size >= 0
            assert o.expected_size is None or o.expected_size >= 0


class TaskOutput:

    __slots__ = ("parent", "id", "size", "consumers", "expected_size")

    def __init__(self, size, expected_size=None):
        assert size >= 0
        assert expected_size is None or expected_size >= 0

        self.parent = None
        self.size = size
        self.consumers = set()
        self.id = None
        self.expected_size = expected_size

    def __repr__(self):
        return "<O id={} p={} size={}>".format(self.id, repr(self.parent), self.size)
