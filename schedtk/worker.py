from simpy import Store


class Worker:

    def __init__(self):
        self.assigned_tasks = []

    def assign_task(self, task):
        self.assigned_tasks.append(task)
        self.messages.put(task)

    def run(self, env, simulator):
        self.messages = Store(env)
        while True:
            task = yield self.messages.get()
            yield env.timeout(task.duration)
            self.assigned_tasks.remove(task)
            simulator.on_task_finished(self, task)
