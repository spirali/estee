from .scheduler import StaticScheduler
from .utils import compute_b_level_duration, estimate_schedule, create_scheduler_graph
from ..simulator import SimpleNetModel, TaskAssignment


def find_critical_path(graph):
    b_level = compute_b_level_duration(graph)
    tasks = graph.source_tasks()
    critical_path = []
    while tasks:
        task = max(tasks, key=lambda t: b_level[t])
        critical_path.append(task)
        tasks = task.consumers()
    return critical_path


def critical_path_clustering(graph):
    g = create_scheduler_graph(graph)
    clusters = []
    while g.tasks:
        path = find_critical_path(g)
        clusters.append(path)
        for task in path:
            g.remove_task(task)
    return clusters


class LcScheduler(StaticScheduler):
    def __init__(self):
        super().__init__("LinearClustering", 0)

    def static_schedule(self):
        if not self.task_graph.tasks or not self.workers:
            return

        g = create_scheduler_graph(self.task_graph)
        b_level = compute_b_level_duration(g)
        tasks = g.tasks

        for t in tasks.values():
            t.expected_duration = 0
            for o in t.outputs:
                o.expected_size = 0

        graph = self.task_graph

        worker = self.workers[list(self.workers.keys())[0]]
        result = [TaskAssignment(worker, t, b_level[t]) for t in tasks.values()]

        for cluster in critical_path_clustering(graph):
            best_t = None
            best_w = None
            prev = None
            for t in cluster:
                tt = tasks[t.id]
                tt.expected_duration = t.expected_duration
                for o in tt.inputs:
                    if prev and o.parent.id == prev.id:
                        o.expected_size = graph.objects[o.id].expected_size
                prev = t

            for w in self.workers.values():
                for t in cluster:
                    result[t.id].worker = w
                time = estimate_schedule(result, SimpleNetModel())
                if best_t is None or time < best_t:
                    best_t = time
                    best_w = w

            if w != best_w:
                for t in cluster:
                    result[t.id].worker = best_w

        for a in result:
            self.assign(self.workers[a.worker.worker_id], self.task_graph.tasks[a.task.id],
                        a.priority, a.block)
