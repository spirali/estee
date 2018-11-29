import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--violin", action="store_true")
    parser.add_argument("--boxplot", action="store_true")
    parser.add_argument("--lineplot", action="store_true")
    parser.add_argument("--graph", action="append")
    parser.add_argument("--split-graphs", action="store_true")
    return parser.parse_args()


def draw_heatmap(*args, **kw):
    df = kw["data"].groupby(["bandwidth", "scheduler_name"])["score"].mean()
    df = df.unstack()
    seaborn.heatmap(df, cmap='RdYlGn_r', linewidths=0.5, annot=True)


def draw_violin(*args, **kw):
    seaborn.violinplot(data=kw["data"], x="scheduler_name", y="score")


def draw_boxplot(*args, **kw):
    g = seaborn.boxplot(data=kw["data"], x="scheduler_name", y="score")
    g.set(ylabel="Score")


def draw_lineplot(*args, **kw):
    data = kw["data"]
    data['bandwidth'] = data['bandwidth'].astype(str)
    g = seaborn.lineplot(data=data, x="bandwidth", y="score", hue="scheduler_name")

    maxval = max(data["score"])
    if maxval > 4:
        g.set(ylim=(1, 4), xlabel="Bandwidth", ylabel="Score")


def draw_frame(frame, args, title=None):
    # normalize by minimum schedule found for each graph/cluster/bandwidth/netmodel combination
    mins = frame.groupby(["graph_id", "cluster_name", "bandwidth", "netmodel"])["time"].transform(pd.Series.min)
    frame["score"] = frame["time"] / mins

    # calculate average for each graph/cluster/bandwidth/netmodel/scheduler/imode combination
    data = frame.groupby(["graph_id", "cluster_name", "bandwidth", "netmodel", "scheduler_name", "imode"]) \
        ["score"].mean().reset_index()

    # merge bandwidth and netmodel to a single column
    if len(data["netmodel"].unique()) > 1:
        data["bandwidth"] = ["{}/{}".format(b, n)
                             for (b, n) in zip(data["bandwidth"], data["netmodel"])]

    if args.heatmap:
        fg = seaborn.FacetGrid(data, col='cluster_name')
        fg.map_dataframe(draw_heatmap)

    if args.violin:
        fg = seaborn.FacetGrid(data, col='cluster_name', row='bandwidth')
        fg.map_dataframe(draw_violin)

    if args.boxplot:
        fg = seaborn.FacetGrid(data, col='cluster_name', row='bandwidth')
        fg.map_dataframe(draw_boxplot)

    if args.lineplot:
        fg = seaborn.FacetGrid(data, col='cluster_name')
        fg.map_dataframe(draw_lineplot).add_legend()

    if title:
        plt.gcf().canvas.set_window_title(title)


def main():
    args = parse_args()
    data = pd.read_pickle(args.dataset)

    if args.graph:
        data = data[data["graph_name"].isin(args.graph)].reset_index(drop=True)

    if args.split_graphs:
        graphs = data["graph_name"].unique()
        frames = [data[data["graph_name"] == g] for g in graphs]
        for (i, frame) in enumerate(frames):
            draw_frame(frame, args, graphs[i])
            print(graphs[i])
    else:
        draw_frame(data, args)

    plt.show()


if __name__ == "__main__":
    main()
