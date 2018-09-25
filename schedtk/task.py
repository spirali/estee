

class Task:

    __slots__ = ("inputs", "consumers", "duration", "size", "name", "id", "info", "s_info")

    def __init__(self, name=None, duration=1, size=0):
        self.inputs = []
        self.consumers = set()

        self.name = name
        self.id = None
        self.info = None
        self.s_info = None

        self.duration = duration
        self.size = size

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
        return "<T{} id={}{}>".format(name, self.id, s_info)

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