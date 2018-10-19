

class Task:

    __slots__ = ("inputs", "consumers", "duration", "size", "name", "id", "info", "s_info", "cpus")

    def __init__(self, name=None, duration=1, size=0, cpus=1):
        assert cpus >= 0
        assert duration >= 0
        assert size >= 0

        self.inputs = []
        self.consumers = set()

        self.name = name
        self.id = None
        self.info = None
        self.s_info = None

        self.duration = duration
        self.size = size
        self.cpus = cpus

    def simple_copy(self):
        return Task(self.name, self.duration, self.size, self.cpus)

    @property
    def label(self):
        if self.name:
            return self.name
        else:
            return "id={}".format(self.id)

    def cleanup(self):
        self.info = None
        self.s_info = None

    def add_input(self, task):
        assert isinstance(task, Task)
        self.inputs.append(task)
        task.consumers.add(self)

    def add_inputs(self, tasks):
        for t in tasks:
            self.add_input(t)

    def __repr__(self):
        if self.name:
            name = " '" + self.name + "'"
        else:
            name = ""
        if self.s_info is not None:
            s_info = " " + repr(self.s_info)
        else:
            s_info = ""

        if self.cpus != 1:
            cpus = " c={}".format(self.cpus)
        else:
            cpus = ""

        return "<T{}{} id={}{}>".format(name, cpus, self.id, s_info)

    def is_predecessor_of(self, task):
        descendants = set()
        explore = [self]

        while explore:
            new = []
            for t in explore:
                for d in t.consumers:
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