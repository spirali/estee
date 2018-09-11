from simpy import Store


class Worker:

    def __init__(self):
        self.assigned_tasks = []
        self.ready_tasks = []

        self.data = set()
        self.downloads = {}

    def _download(self, task):
        def _helper():
            yield self.connector.download(task.info.assigned_workers[0], self, task.size)
            del self.downloads[task]
            self.data.add(task)

        p = self.env.process(_helper())
        self.downloads[task] = p
        return p

    def assign_task(self, task):
        self.assigned_tasks.append(task)

        downloads = []
        for input in task.inputs:
            if input in self.data:
                continue
            d = self.downloads.get(input)
            if d is None:
                d = self._download(input)
            downloads.append(d)

        def _helper():
            yield self.env.all_of(downloads)
            self.ready_tasks.put(task)

        if not downloads:
            self.ready_tasks.put(task)
        else:
            self.env.process(_helper())

    def run(self, env, simulator, connector):
        self.env = env
        self.connector = connector
        self.ready_tasks = Store(env)

        while True:
            task = yield self.ready_tasks.get()
            yield env.timeout(task.duration)
            self.assigned_tasks.remove(task)
            self.data.add(task)
            simulator.on_task_finished(self, task)
