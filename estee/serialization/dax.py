import uuid

from lxml import etree as ET

from ..common.taskgraph import TaskGraph


def add_artificial_outputs(root, xmlns_prefix, tasks):
    for child in root.findall("{}child".format(xmlns_prefix)):
        child_task = tasks[child.get("ref")]

        parents = [tasks[p.get("ref")] for p in child.findall("{}parent".format(xmlns_prefix))]
        child_task["parents"] = parents
        for parent in parents:
            if not set(child_task["inputs"]).intersection([o["name"] for o in parent["outputs"]]):
                name = uuid.uuid4().hex
                parent["outputs"].append({
                    "name": name,
                    "size": 0.0,
                    "expected_size": 0.0
                })
                child_task["inputs"].append(name)


def get_output_name(id, name):
    return "{}-{}".format(id, name)


def add_inputs(ids, task_by_id, task_outputs, tasks):
    for id in ids:
        task = task_by_id[id]
        for input_name in tasks[id]["inputs"]:
            parent_outputs = [get_output_name(p["id"], input_name) for p in tasks[id]["parents"]]
            parent_outputs = [o for o in parent_outputs if o in task_outputs]

            for o in parent_outputs:
                task.add_input(task_outputs[o])


def create_tasks(ids, tasks):
    tg = TaskGraph()
    task_by_id = {}
    task_outputs = {}

    for id in ids:
        definition = tasks[id]
        task = tg.new_task(name=definition["name"],
                           duration=definition["duration"],
                           expected_duration=definition["expected_duration"],
                           cpus=definition["cpus"],
                           outputs=[o["size"] for o in definition["outputs"]])
        for (output, parsed_output) in zip(task.outputs, definition["outputs"]):
            output.expected_size = parsed_output["expected_size"]

        for (index, o) in enumerate(definition["outputs"]):
            name = get_output_name(id, o["name"])
            assert name not in task_outputs
            task_outputs[name] = task.outputs[index]
        task_by_id[id] = task
    return (tg, task_by_id, task_outputs)


def dax_deserialize(file):
    tasks = {}
    ids = []

    def parse_value(val, convert=None, default=None):
        if val is None:         # value is not present
            if default is not None:
                return default
        elif val == 'None':     # value is present, but unset
            return None
        elif convert:
            return convert(val)
        return val

    def normalize_size(size):
        if size is None:
            return None
        return size / (1024 * 1024)

    root = ET.parse(file).getroot()

    xmlns_prefix = "{{{}}}".format(root.nsmap[None]) if None in root.nsmap else ""

    for job in root.findall("{}job".format(xmlns_prefix)):
        files = job.findall("{}uses".format(xmlns_prefix))

        outputs = [
            {"name": f.get("file"),
             "size": normalize_size(parse_value(f.get("size"), float, 1)),
             "expected_size": normalize_size(parse_value(f.get("expectedSize"), float, None))
             }
            for f in files if f.get("link") == "output"
        ]
        inputs = [f.get("file") for f in files if f.get("link") == "input"]

        id = job.get("id")
        assert id
        ids.append(id)

        name = job.get("name", id)

        cpus = parse_value(job.get("cores", 1), int, 1)
        duration = parse_value(job.get("runtime"), float, 1)
        expected_duration = parse_value(job.get("expectedRuntime"), float, None)

        assert id not in tasks
        tasks[id] = {
            "id": id,
            "name": name,
            "duration": duration,
            "expected_duration": expected_duration,
            "cpus": cpus,
            "outputs": outputs,
            "inputs": inputs,
            "parents": []
        }

    add_artificial_outputs(root, xmlns_prefix, tasks)

    (tg, task_by_id, task_outputs) = create_tasks(ids, tasks)

    add_inputs(ids, task_by_id, task_outputs, tasks)

    tg.validate()

    return tg


def dax_serialize(task_graph, file):
    doc = ET.Element("adag")

    task_to_id = {}

    MiB = 1024 * 1024

    tasks = list(task_graph.tasks.values())

    for task in tasks:
        id = "task-{}".format(len(task_to_id))
        task_tree = ET.SubElement(doc, "job",
                                  id=id,
                                  name=task.name or "",
                                  runtime=str(task.duration),
                                  expectedRuntime=str(task.expected_duration),
                                  cores=str(task.cpus))
        for (index, output) in enumerate(task.outputs):
            name = "{}-o{}".format(id, index)
            ET.SubElement(task_tree, "uses",
                          link="output",
                          size=str(output.size * MiB),
                          expectedSize=str(output.expected_size * MiB),
                          file=name)
        task_to_id[task] = (id, task_tree)

    for task in tasks:
        (_, tree) = task_to_id[task]
        inputs = sorted(task.inputs, key=lambda i: task_to_id[i.parent][0])
        for (index, input) in enumerate(inputs):
            parent = input.parent
            (id, _) = task_to_id[parent]
            name = "{}-o{}".format(id, parent.outputs.index(input))
            ET.SubElement(tree, "uses",
                          link="input",
                          size=str(input.size * MiB),
                          expectedSize=str(input.expected_size * MiB),
                          file=name)

    for task in tasks:
        if task.inputs:
            elem = ET.SubElement(doc, "child", ref=task_to_id[task][0])
            parents = sorted({i.parent for i in task.inputs}, key=lambda t: task_to_id[t][0])
            for parent in parents:
                ET.SubElement(elem, "parent", ref=task_to_id[parent][0])

    tree = ET.ElementTree(doc)
    tree.write(file, pretty_print=True, xml_declaration=True, encoding="utf-8")
