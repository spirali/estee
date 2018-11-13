import uuid

from lxml import etree as ET

from ..common.taskgraph import TaskGraph

XMLNS_PREFIX = ""


def dax_deserialize(file):
    tasks = {}
    ids = []

    root = ET.parse(file).getroot()

    for job in root.findall("{}job".format(XMLNS_PREFIX)):
        files = job.findall("{}uses".format(XMLNS_PREFIX))

        outputs = [
            {"name": f.get("file"), "size": float(f.get("size"))}
            for f in files if f.get("link") == "output"
        ]
        inputs = [f.get("file") for f in files if f.get("link") == "input"]

        id = job.get("id")
        assert id
        ids.append(id)

        name = job.get("name", id)

        cpus = int(job.get("cores", 1))
        runtime = float(job.get("runtime"))
        assert id not in tasks
        tasks[id] = {
            "name": name,
            "duration": runtime,
            "cpus": cpus,
            "outputs": outputs,
            "inputs": inputs
        }

    for child in root.findall("{}child".format(XMLNS_PREFIX)):
        child_task = tasks[child.get("ref")]

        parents = [tasks[p.get("ref")] for p in child.findall("{}parent".format(XMLNS_PREFIX))]
        for parent in parents:
            if not set(child_task["inputs"]).intersection([o["name"] for o in parent["outputs"]]):
                name = uuid.uuid4().hex
                parent["outputs"].append({
                    "name": name,
                    "size": 0.0
                })
                child_task["inputs"].append(name)

    tg = TaskGraph()
    task_outputs = {}
    task_by_id = {}

    for id in ids:
        definition = tasks[id]
        task = tg.new_task(name=definition["name"],
                           duration=definition["duration"],
                           cpus=definition["cpus"],
                           outputs=[o["size"] for o in definition["outputs"]])
        for (index, o) in enumerate(definition["outputs"]):
            assert o["name"] not in task_outputs
            task_outputs[o["name"]] = task.outputs[index]
        task_by_id[id] = task

    for id in ids:
        task = task_by_id[id]
        for input in tasks[id]["inputs"]:
            if input in task_outputs:
                task.add_input(task_outputs[input])

    tg.validate()

    return tg


def dax_serialize(task_graph, file):
    doc = ET.Element("adag")

    task_to_id = {}

    for task in task_graph.tasks:
        id = "task-{}".format(len(task_to_id))
        task_tree = ET.SubElement(doc, "job",
                                  id=id,
                                  name=task.name,
                                  runtime=str(task.duration),
                                  cores=str(task.cpus))
        for (index, output) in enumerate(task.outputs):
            name = "{}-o{}".format(id, index)
            ET.SubElement(task_tree, "uses", link="output", size=str(output.size), file=name)
        task_to_id[task] = (id, task_tree)

    for task in task_graph.tasks:
        (_, tree) = task_to_id[task]
        inputs = sorted(task.inputs, key=lambda i: task_to_id[i.parent][0])
        for (index, input) in enumerate(inputs):
            parent = input.parent
            (id, _) = task_to_id[parent]
            name = "{}-o{}".format(id, parent.outputs.index(input))
            ET.SubElement(tree, "uses", link="input", size=str(input.size), file=name)

    for task in task_graph.tasks:
        if task.inputs:
            elem = ET.SubElement(doc, "child", ref=task_to_id[task][0])
            parents = sorted({i.parent for i in task.inputs}, key=lambda t: task_to_id[t][0])
            for parent in parents:
                ET.SubElement(elem, "parent", ref=task_to_id[parent][0])

    tree = ET.ElementTree(doc)
    tree.write(file, pretty_print=True, xml_declaration=True, encoding="utf-8")
