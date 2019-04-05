
# This scripts generates graphs for
# outputs of benchmarks

import argparse
import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

LINE_STYLES = ["-", ":", "-.", "--"]
cmap = plt.cm.get_cmap('Dark2')
COLORS = [cmap(i) for i in range(5)]
MARKERS = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "x", "D"]


def style_gen():
    for line, color, marker in zip(
            itertools.cycle(LINE_STYLES),
            itertools.cycle(COLORS),
            itertools.cycle(MARKERS)):
        yield {
            "line": line,
            "color": color,
            "marker": marker
        }


class Data:

    def __init__(self, filename):
        self.raw_data = pd.read_csv(filename)

        self.raw_data.drop("graph_set", axis=1, inplace=True)
        exclude = ["tlevel-simple", "blevel-simple"]
        self.raw_data = pd.DataFrame(
            self.raw_data[~self.raw_data["scheduler_name"].isin(exclude)])

        mins = self.raw_data.groupby(
            ["graph_id", "cluster_name", "bandwidth", "netmodel"]
        )["time"].transform(pd.Series.min)

        self.raw_data["score"] = self.raw_data["time"] / mins

    def prepare(self,
                cluster_name=None,
                exclude_single=False,
                netmodel="maxmin",
                min_sched_interval=0.1,
                imode="exact"):
        rd = self.raw_data

        if netmodel:
            f = rd["netmodel"] == netmodel
        else:
            f = rd["netmodel"].isin(["simple", "maxmin"])

        if min_sched_interval is not None:
            f &= rd["min_sched_interval"] == min_sched_interval
        else:
            f &= rd["min_sched_interval"].isin([0.0, 0.1, 0.4, 1.6, 6.4])

        if imode is not None:
            f &= rd["imode"] == imode
        else:
            f &= rd["imode"].isin(["exact", "mean", "user"])

        if cluster_name:
            f &= rd["cluster_name"] == cluster_name
        if cluster_name:
            f &= rd["scheduler_name"] != "single"
        return pd.DataFrame(rd[f])


def splot(data, col, row, x, y,
          style_col=None, sharex=False, sharey=True, ylim=None, style_values=None):
    if style_col is None:
        values = style_values
    else:
        values = sorted(data[style_col].unique())

    def draw(gdata, ax):
        ax.set_xscale("log", nonposx='clip')
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        if ylim is not None:
            ax.set(ylim=ylim)
        for v, style in zip(values, style_gen()):
            fdata = gdata[gdata[style_col] == v]
            ax.plot(fdata[x], fdata[y], 'ro',
                    markersize=5, color=style["color"], marker=style["marker"])
            means = fdata.groupby(x)[y].mean()
            ax.plot(means.index, means, linestyle=style["line"], color=style["color"])

    rows = sorted(data[row].unique())
    cols = sorted(data[col].unique())
    idata = data.copy()
    idata.set_index([idata[col], idata[row]], inplace=True)
    idata.sort_index(inplace=True)
    fig, axes = plt.subplots(nrows=len(rows),
                             ncols=len(cols),
                             figsize=(len(cols) * 4 + 1, len(rows) * 4))

    if not cols or not rows:
        return

    if len(cols) == 1 and len(rows) == 1:
        axes = np.array([[axes]])
    elif len(rows) == 1:
        axes = axes.reshape((1, -1))
    elif len(cols) == 1:
        axes = axes.reshape((-1, 1))

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, size='large')

    for i, c in enumerate(cols):
        for j, r in enumerate(rows):
            idx = (c, r)
            if idx in idata.index:
                gdata = idata.loc[idx]
                ax = axes[j, i]
                draw(gdata, ax)

    fig.legend(handles=[Line2D([], [], color=style["color"],
                               label=v, linestyle=style["line"], marker=style["marker"])
                        for v, style in zip(values, style_gen())],
               bbox_to_anchor=(0, 1),
               loc="lower left",
               ncol=20)
    plt.tight_layout()


def savefig(name):
    plt.savefig("outputs/" + name + ".pdf", bbox_inches='tight')


def process(filename, args):
    print("processing " + filename)

    name = os.path.basename(filename)
    if "." in name:
        name = name[:name.rfind(".")]

    data = Data(filename)

    # ----- Schedulers -----
    if args.all or args.schedulers:
        print("Schedulers ...")
        dataset = data.prepare()

        if len(dataset) > 0:
            splot(dataset, "cluster_name", "graph_name", x="bandwidth", y="score",
                style_col="scheduler_name", ylim=(1, 3))
            savefig(name + "-schedulers-score")

            splot(dataset, "cluster_name", "graph_name", x="bandwidth", y="time",
                style_col="scheduler_name", sharey=False)
            savefig(name + "-schedulers-time")

            splot(dataset, "cluster_name", "graph_name", x="bandwidth", y="total_transfer",
                style_col="scheduler_name", sharey=False)
            savefig(name + "-schedulers-transfer")

    # ----- Netmodel -----
    if args.all or args.netmodels:
        print("Netmodels ...")
        dataset = data.prepare(cluster_name=args.cluster, netmodel=None, exclude_single=True)

        if len(dataset) > 0:
            splot(dataset, "graph_name", "scheduler_name", x="bandwidth", y="time",
                style_col="netmodel", sharey=False)
            savefig(name + "-{}-netmodel-time".format(args.cluster))

            splot(dataset, "graph_name", "scheduler_name", x="bandwidth", y="total_transfer",
                style_col="netmodel", sharey=False)
            savefig(name + "-{}-netmodel-transfer".format(args.cluster))

            groups = dataset.groupby(
                ["graph_name", "graph_id", "cluster_name", "bandwidth", "scheduler_name"])

            def normalize(x):
                mean = x[x["netmodel"] == "simple"]["time"].mean()
                x["time"] /= mean
                return x

            dataset["norms"] = groups.apply(normalize)["time"]
            splot(dataset, "graph_name", "scheduler_name", x="bandwidth", y="norms",
                sharey=False, style_col="netmodel")
            savefig(name + "-{}-netmodel-score".format(args.cluster))

    # ----- MinSchedTime
    if args.all or args.msd:
        print("MSD ...")
        dataset = data.prepare(cluster_name="16x4", min_sched_interval=None, exclude_single=True)

        if len(dataset) > 0:
            splot(dataset, "graph_name", "scheduler_name", x="bandwidth", y="time",
                style_col="min_sched_interval", sharey=False)
            savefig(name + "-16x4-schedtime-time")

            groups = dataset.groupby(
                ["graph_name", "graph_id", "cluster_name", "bandwidth", "scheduler_name"])

            def normalize(x):
                mean = x[x["min_sched_interval"] == 0.0]["time"].mean()
                x["time"] /= mean
                return x

            dataset["norms"] = groups.apply(normalize)["time"]
            splot(dataset, "graph_name", "scheduler_name", x="bandwidth", y="norms",
                sharey=False, style_col="min_sched_interval")
            savefig(name + "-16x4-schedtime-score")

    # ----- Imodes
    if args.all or args.imodes:
        print("Imodes ...")
        dataset = data.prepare(cluster_name="16x4", exclude_single=True, imode=None)

        if len(dataset) > 0:
            splot(dataset, "graph_name", "scheduler_name", x="bandwidth", y="time",
                style_col="imode", sharey=False)
            savefig(name + "-16x4-imode-time")

            groups = dataset.groupby(
                ["graph_name", "graph_id", "cluster_name", "bandwidth", "scheduler_name"])

            def normalize_imode(x):
                mean = x[x["imode"] == "exact"]["time"].mean()
                x["time"] /= mean
                return x

            dataset["norms_imode"] = groups.apply(normalize_imode)["time"]
            splot(dataset, "graph_name", "scheduler_name", x="bandwidth", y="norms_imode",
                sharey=False, style_col="imode")
            savefig(name + "-16x4-imode-score")
    return name


if __name__ == "__main__":
    if not os.path.isdir("outputs"):
        os.mkdir("outputs")

    parser = argparse.ArgumentParser()
    parser.add_argument("resultset")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--schedulers", action="store_true")
    parser.add_argument("--msd", action="store_true")
    parser.add_argument("--imodes", action="store_true")
    parser.add_argument("--netmodels", action="store_true")
    parser.add_argument("--cluster", default="16x4")
    args = parser.parse_args()

    process(args.resultset, args)
