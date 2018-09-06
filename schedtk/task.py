

class Task:

    __slots__ = ("inputs", "consumers", "duration", "name", "id", "info")

    def __init__(self, name=None, duration=1):
        self.inputs = []
        self.consumers = set()

        self.duration = duration
        self.name = name
        self.id = None
        self.info = None

    @property
    def label(self):
        return "{} [{}]\n{}".format(self.name, self.id, self.duration)

    def add_input(self, task):
        assert isinstance(task, Task)
        self.inputs.append(task)
        task.consumers.add(self)

    def add_inputs(self, tasks):
        for t in tasks:
            self.add_input(t)

    def __repr__(self):
        if self.name:
            name = " " + self.name
        else:
            name = ""
        return "<Task{} id={}>".format(name, self.id)

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