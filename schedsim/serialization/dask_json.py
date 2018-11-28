import json

from schedsim.common import TaskGraph


def json_serialize(graph):
    task_to_id = {}
    tasks = []
    output_to_index = {}

    for task in graph.tasks:
        ser = {
            "d": task.duration,
            "e_d": task.expected_duration,
            "cpus": task.cpus,
            "outputs": [{"s": o.size, "e_s": o.expected_size} for o in task.outputs]
        }
        task_to_id[task] = len(tasks)
        tasks.append(ser)
        for (index, output) in enumerate(task.outputs):
            output_to_index[output] = index

    for (i, task) in enumerate(graph.tasks):
        inputs = []
        for input in task.inputs:
            parent = input.parent
            output_index = output_to_index[input]
            inputs.append((task_to_id[parent], output_index))
        tasks[i]["inputs"] = inputs

    return json.dumps(tasks)


def json_deserialize(data):
    tasks = json.loads(data)

    graph = TaskGraph()
    id_to_task = {}
    for t in tasks:
        task = graph.new_task(
            duration=t["d"],
            expected_duration=t["e_d"],
            cpus=t["cpus"],
            outputs=[o["s"] for o in t["outputs"]]
        )
        for (index, output) in enumerate(task.outputs):
            output.expected_size = t["outputs"][index]["e_s"]
        id_to_task[len(id_to_task)] = task

    for (i, t) in enumerate(tasks):
        for (parent, output_index) in t["inputs"]:
            parent = id_to_task[parent]
            id_to_task[i].add_input(parent.outputs[output_index])
    return graph
