
import tensorflow as tf
import numpy as np
import collections

from utils.graph import Graph


class NodeType:

    def __init__(self, name, state_size, color=None):
        self.name = name
        self.state_size = state_size
        self.color = color

    def zeros(self, count):
        return np.zeros((count, self.state_size))

    def __repr__(self):
        return "<NodeType {}>".format(self.name)


class ArcType:

    def __init__(self, name, source_nt, target_nt, both_side_visibility=True):
        assert isinstance(source_nt, NodeType)
        assert isinstance(target_nt, NodeType)

        self.name = name
        self.source_nt = source_nt
        self.target_nt = target_nt
        self.both_side_visibility = both_side_visibility

    def __repr__(self):
        return "<ArcType {}>".format(self.name)


class GnDef:

    def __init__(self, node_types, arc_types):
        self.node_types = node_types
        self.arc_types = arc_types

    def node_zeros(self, counts):
        result = {}
        for nt in self.node_types:
            result[nt] = nt.zeros(counts[nt])
        return result

    def make_input_instance(self, arcs, inits, shifts=None):
        for at in arcs:
            assert at in self.arc_types
        return GraphInstance(self, arcs, inits, shifts)

    def merge_instances(self, instances):
        inits = {}
        shifts = {}

        for nt in self.node_types:
            ins = [instance.inits[nt] for instance in instances]
            inits[nt] = np.concatenate(ins)
            counts = [i.shape[0] for i in ins]
            counts.insert(0, 0)
            shifts[nt] = np.cumsum(counts)

        arcs = {}
        for at in self.arc_types:
            s_shifts = shifts[at.source_nt]
            t_shifts = shifts[at.target_nt]

            arrays = []
            for s_shift, t_shift, instance in zip(s_shifts, t_shifts, instances):
                a = np.array(instance.arcs[at])
                a[:, 0] += s_shift
                a[:, 1] += t_shift
                arrays.append(a)
            arcs[at] = np.concatenate(arrays)
        return self.make_input_instance(arcs, inits, shifts)

    def make_input_placeholders(self):
        phs = {}
        for at in self.arc_types:
            phs[at] = tf.placeholder(
                tf.int32, name="arc_{}".format(at.name), shape=(None, 2))

        for nt in self.node_types:
            phs[nt] = tf.placeholder(
                tf.float32,
                name="init_values_{}".format(nt.name),
                shape=(None, nt.state_size))
        return phs

    def build(self, depth, inputs, layer_builder):
        sources = {}
        targets = {}
        for at in self.arc_types:
            expr = inputs[at]
            sources[at] = expr[:, 0]
            targets[at] = expr[:, 1]

        node_counts = {}
        for nt in self.node_types:
            node_counts[nt] = tf.shape(inputs[nt])[0]

        state = inputs
        for _ in range(depth):
            new_state = {}
            for nt in self.node_types:
                messages = []
                for at in self.arc_types:
                    if nt == at.source_nt:
                        messages += message_passing(
                            state[at.target_nt],
                            targets[at], sources[at],
                            node_counts[nt], at.name + "_A")
                    if nt == at.target_nt and at.both_side_visibility:
                        messages += message_passing(
                            state[at.source_nt],
                            sources[at], targets[at],
                            node_counts[nt], at.name + "_B")
                new_state[nt] = layer_builder(nt, state[nt], messages)
            state = new_state
        return state


class GraphInstance:

    def __init__(self, gndef, arcs, inits, shifts=None):
        assert isinstance(arcs, dict)
        assert isinstance(inits, dict)
        self.gndef = gndef
        self.inits = inits
        self.arcs = arcs
        self.shifts = shifts

    def debug_graph(self, node_types=None):
        g = Graph()

        if node_types is None:
            node_types = self.gndef.node_types

        for nt in node_types:
            for i, data in enumerate(self.inits[nt]):
                n = g.node((nt, i))
                n.color = nt.color
                n.label = "{}\n{}\n{}".format(nt.name, i, data)
                n.shape = "box"

        for at, values in self.arcs.items():
            if at.source_nt not in node_types or at.target_nt not in node_types:
                continue
            for s, t in values:
                sn = g.node_get((at.source_nt, s))
                tn = g.node_get((at.target_nt, t))
                a = sn.add_arc(tn)

        return g

    def data(self):
        d = self.arcs.copy()
        d.update(self.inits)
        return d


def make_binding(vars, data):
    return {v: data[k] for k, v in vars.items()}


def message_passing(state, read_ids, write_ids, node_count, name):
    values = tf.nn.embedding_lookup(
        state, read_ids, name="lookup_" + name)

    s_sum = tf.unsorted_segment_sum(
        values, write_ids, node_count, name="segment_sum_" + name)

    s_max = tf.unsorted_segment_max(
        values, write_ids, node_count, name="segment_max_" + name)
    s_max = tf.maximum(s_max, -1.0)

    s_min = tf.unsorted_segment_min(
        values, write_ids, node_count, name="segment_min_" + name)
    s_min = tf.minimum(s_min, 1.0)

    return [s_sum, s_max, s_min]


# Makes sure of the shape for empty lists
def make_arcs(values):
    if values:
        return np.array(values, np.int32)
    else:
        return np.empty((0, 2), np.int32)