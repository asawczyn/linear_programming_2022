import operator

import pandas as pd
from matplotlib import pyplot as plt

import networkx as nx


def plot_graph_cut(x, edges):
    G = nx.Graph()
    node_color_map = []
    for node_id, node_type in x.items():
        G.add_node(node_id)
        node_color_map.append("yellow" if node_type else "lightgreen")

    for node_i, node_j in edges:
        color = "black" if x[node_i] == x[node_j] else "red"
        G.add_edge(node_i, node_j, color=color)

    nx.draw(G, with_labels=True, font_weight="bold")
    plt.savefig("output/graph_cut_1.png")

    plt.show()
    edge_color_map = [G[u][v]["color"] for u, v in G.edges()]
    nx.draw(
        G,
        node_color=node_color_map,
        edge_color=edge_color_map,
        with_labels=True,
        font_weight="bold",
    )
    plt.savefig("output/graph_cut_2.png")
    plt.show()


def plot_gant(df):
    MAX_TIME = 12
    fig, gnt = plt.subplots()
    gnt.set_ylim(8, 41)
    gnt.set_xlim(0, MAX_TIME)

    gnt.set_xlabel("Starting time")
    gnt.set_ylabel("Machine")

    gnt.set_yticks([15, 25, 35])
    gnt.hlines(
        y=[9.5, 19.5, 29.5, 39.5],
        xmin=[0],
        xmax=[MAX_TIME],
        colors="purple",
        linestyles="--",
        lw=1,
        label="Multiple Lines",
    )

    gnt.set_yticklabels(["3", "2", "1"])
    gnt.set_xticks(list(range(MAX_TIME + 1)))

    for i, color in zip(range(1, df.job.max() + 1), ["tab:orange", "tab:red", "tab:blue"]):
        for ij_, s_, p_, m_ in df[df.job == i][
            ["i,j", "starting_time", "processing_time", "machine"]
        ].values:
            m_ = (df.machine.max() - m_ + 1) * 10
            gnt.broken_barh([(s_, p_)], (m_ + (i - 1) * 3, 3), facecolors=color)
            gnt.text(
                x=s_ + p_ / 2,
                y=m_ + (i - 1) * 3 + 1.2,
                s=f"O_{ij_[0]},{ij_[1]}",
                ha="center",
                va="center",
                size=8,
            )

    plt.savefig("output/job_shop_gant.png")
    plt.show()


def convert_output_to_dataframe(data, i):
    df = pd.DataFrame()
    df["i,j"] = data[None]["m"].keys()
    df["job"] = df["i,j"].map(operator.itemgetter(0))
    df["operation"] = df["i,j"].map(operator.itemgetter(1))
    df["processing_time"] = data[None]["p"].values()
    df["machine"] = data[None]["m"].values()
    df["starting_time"] = df["i,j"].map(i.s.get_values()).astype(int)
    return df