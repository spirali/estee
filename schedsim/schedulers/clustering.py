from .scheduler import StaticScheduler
from ..simulator import TaskAssignment
from .utils import compute_b_level_duration, compute_independent_tasks, get_duration_estimate, \
    get_size_estimate, max_cpus_worker
import itertools
from ..simulator.utils import estimate_schedule
from ..communication import SimpleNetModel


def find_critical_path(graph):
    b_level = compute_b_level_duration(graph)
    tasks = graph.tasks
    critical_path = []
    while tasks:
        task = max(tasks, key=lambda t: b_level[t])
        critical_path.append(task)
        tasks = task.consumers()
    return critical_path


def critical_path_clustering(graph):
    g = graph.copy()
    clusters = []
    while g.tasks:
        path = find_critical_path(g)
        clusters.append([graph.tasks[t.id] for t in path])
        for task in path:
            g.remove_task(task)
    return clusters


class LcScheduler(StaticScheduler):

    def static_schedule(self):
        graph = self.simulator.task_graph

        g = graph.copy()
        b_level = compute_b_level_duration(g)
        tasks = g.tasks

        for t in tasks:
            t.expected_duration = 0
            for o in t.outputs:
                o.expected_size = 0

        original_tasks = graph.tasks
        original_outputs = graph.outputs

        result = [TaskAssignment(self.simulator.workers[0], t, b_level[t]) for t in tasks]

        for cluster in critical_path_clustering(graph):
            best_t = None
            best_w = None
            prev = None
            for t in cluster:
                tt = tasks[t.id]
                tt.expected_duration = t.expected_duration
                for o in tt.inputs:
                    if prev and o.parent.id == prev.id:
                        o.expected_size = original_outputs[o.id].expected_size
                prev = t

            for w in self.simulator.workers:
                prev = None
                for t in cluster:
                    result[t.id].worker = w
                time = estimate_schedule(result, g, SimpleNetModel())
                if best_t is None or time < best_t:
                    best_t = time
                    best_w = w

            if w != best_w:
                for t in cluster:
                    result[t.id].worker = best_w

        result = [TaskAssignment(a.worker, original_tasks[a.task.id], a.priority, a.block) for
                  a in result]
        return result