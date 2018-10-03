
from graphnet import NodeType, ArcType, GnDef, make_arcs, make_binding
import numpy as np
import tensorflow as tf

RED = NodeType("red", 16, color="red")
BLUE = NodeType("blue", 16, color="blue")
ORANGE = NodeType("orange", 24, color="orange")

ARC_RED_RED = ArcType("rr", RED, RED)
ARC_RED_BLUE = ArcType("rb", RED, BLUE)
ARC_BLUE_ORANGE = ArcType("bg", BLUE, ORANGE)
ARC_RED_ORANGE = ArcType("rg", RED, ORANGE)

GRAPH = GnDef([RED, ORANGE, BLUE],
              [ARC_RED_RED, ARC_RED_BLUE, ARC_BLUE_ORANGE, ARC_RED_ORANGE])


def random_graph():
    red_count = np.random.randint(6, 14)
    blue_count = np.random.randint(2, 5)
    inits = GRAPH.node_zeros({
        RED: red_count,
        BLUE: blue_count,
        ORANGE: 1,
    })

    rr = []
    rb = []
    for i in range(red_count):
        inits[RED][i, 0] = np.random.random()
        for j in range(i + 1, red_count):
            if np.random.rand() < 0.3:
                rr.append((i, j))
        for j in range(blue_count):
            if np.random.rand() < 0.21:
                rb.append((i, j))

    rr = make_arcs(rr)
    rb = make_arcs(rb)

    arcs = {
        ARC_BLUE_ORANGE: make_arcs(
            [(i, 0) for i in range(np.random.randint(blue_count))]),
        ARC_RED_ORANGE: make_arcs(
            [(i, 0) for i in range(np.random.randint(red_count))]),
        ARC_RED_RED: rr,
        ARC_RED_BLUE: rb
    }

    values = np.array([np.sum(inits[RED][np.unique(arcs[ARC_RED_BLUE][:,0])])])
    values = values.reshape((-1, 1))
    return GRAPH.make_input_instance(arcs, inits), values


def build_net():
    layers = {nt: [tf.layers.Dense(80, activation=tf.nn.leaky_relu),
                   tf.layers.Dense(nt.state_size, activation=None)]
              for nt in GRAPH.node_types}

    def layer_builder(nt, state, messages):
        hidden = tf.concat([state] + messages, axis=1)
        for layer in layers[nt]:
            hidden = layer(hidden)
        return hidden + state

    inputs = GRAPH.make_input_placeholders()
    outputs = GRAPH.build(6, inputs, layer_builder)

    output = outputs[ORANGE]
    output = tf.layers.dense(output, 32, activation=tf.nn.leaky_relu)
    output = tf.layers.dense(output, 1, activation=tf.nn.leaky_relu)
    return inputs, output


def build_training(input, output):
    labels = tf.placeholder(tf.float32, shape=(None, 1), name="labels")
    loss = tf.losses.mean_squared_error(labels, output, scope="loss")
    with tf.variable_scope("trainer", reuse=tf.AUTO_REUSE):
        trainer = tf.train.AdamOptimizer().minimize(loss, name="c_training")
    return trainer, labels, loss


def test_train():
    with tf.Session() as session:
        BATCH_SIZE = 120
        with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
            input, output = build_net()
        trainer, labels, loss = build_training(input, output)

        session.run(tf.global_variables_initializer())

        for i in range(1000):
            graphs, values = zip(*[random_graph() for _ in range(BATCH_SIZE)])
            big_graph = GRAPH.merge_instances(graphs)
            big_values = np.concatenate(values)

            binding = make_binding(input, big_graph.data())
            binding[labels] = big_values
            _, loss_value = session.run([trainer, loss], binding)
            if i % 20 == 0:
                print(loss_value)

        graphs, values = zip(*[random_graph() for _ in range(10)])
        big_graph = GRAPH.merge_instances(graphs)
        big_values = np.concatenate(values)
        binding = make_binding(input, big_graph.data())
        binding[labels] = big_values
        error = session.run(tf.concat([output, output - labels], axis=1), binding)
        print(error)


test_train()