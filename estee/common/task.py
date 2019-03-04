
from .taskbase import TaskBase, DataObjectBase


class Task(TaskBase):

    def __init__(self,
                 task_id,
                 name=None,
                 outputs=(),
                 duration=1,
                 cpus=1,
                 output_size=None,
                 expected_duration=None):
        assert cpus >= 0
        assert duration >= 0
        assert expected_duration is None or expected_duration >= 0

        if output_size is not None:
            if outputs:
                raise Exception("Cannot set 'output_size' and 'outputs' at once")
            outputs = (DataObject(None, output_size, output_size),)
        else:
            outputs = tuple(DataObject(None, s, s) if (isinstance(s, float) or isinstance(s, int))
                                 else s for s in outputs)

        for output in outputs:
            assert output.parent is None
            output.parent = self

        super().__init__(task_id, [], outputs)

        self.name = name
        self.duration = duration
        self.expected_duration = expected_duration
        self.cpus = cpus

    def to_dict(self):
        return {
            "id": self.id,
            "inputs": [o.id for o in self.inputs],
            "outputs": [o.id for o in self.outputs],
            "expected_duration": self.expected_duration,
            "cpus": self.cpus
        }

    def simple_copy(self):
        t = Task(self.id, self.name, duration=self.duration, expected_duration=self.expected_duration,
                 cpus=self.cpus)
        t.outputs = [DataObject(o.id, o.size, o.expected_size) for o in self.outputs]
        for o in t.outputs:
            o.parent = t
        return t

    @property
    def output(self):
        outputs = self.outputs
        if not outputs:
            raise Exception("Task {} has no output", self)
        if len(outputs) > 1:
            raise Exception("Task {} has no unique output", self)
        return outputs[0]

    @property
    def label(self):
        if self.name:
            return self.name
        else:
            return "id={}".format(self.id)

    def add_input(self, output):
        if isinstance(output, Task):
            output = output.output
        elif not isinstance(output, DataObject):
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

    def validate(self):
        assert self.duration >= 0
        assert self.expected_duration is None or self.expected_duration >= 0
        assert not self.is_predecessor_of(self)
        assert len(self.outputs) == len(set(self.outputs))
        for o in self.outputs:
            assert o.parent == self
            assert o.size >= 0
            assert o.expected_size is None or o.expected_size >= 0


class DataObject(DataObjectBase):

    def __init__(self, object_id=None, size=None, expected_size=None):
        assert size >= 0
        assert expected_size is None or expected_size >= 0

        super().__init__(object_id)
        self.size = size
        self.expected_size = expected_size

    def to_dict(self):
        return {
            "id": self.id,
            "expected_size": self.expected_size,
        }

    def __repr__(self):
        return "<O id={} p={} size={}>".format(self.id, repr(self.parent), self.size)
