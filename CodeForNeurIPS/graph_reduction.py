import argparse

import numpy as np
from GLGraph import GLGraph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test script for GLGraph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-file",
        default="./Jazz_Edges.npy",
        help="Path to input edgelist .npy file. Format is a list of node "
        "pairs, where nodes are integers in [0, n-1].",
    )

    parser.add_argument(
        "--reduction-target",
        default="edges",
        choices=["edges", "nodes"],
        help="Target item to reduce.",
    )

    parser.add_argument(
        "--action-switch",
        default="both",
        choices=["both", "delete"],
        help="Choosing 'delete' does not allow contraction.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of edges to sample, setting to 0 means 'all', "
        "which samples the entire graph when q=1, and a maximal "
        "matching when q>1.",
    )

    parser.add_argument(
        "--qos",
        type=float,
        default=0.125,
        help="Perturbed edges per sampled edges, a fraction bewteen 0 and 1. "
        "Seting to 0 gives q=1 per round using the single-edge method.",
    )

    parser.add_argument(
        "--min-prob-per-act",
        type=float,
        default=0.25,
        help="Minimum expected (traget items removed) / (num actions taken), "
        "a fraction between 0 and 1.We tend to use around d=1/4, but go to "
        "town and experiment if you want.",
    )

    parser.add_argument(
        "--min-target-items",
        type=int,
        default=1024,
        help="End the reduction when the number of target items it below this "
        "number. Setting to 0 means 'all', i.e. reduce until one cannot.",
    )

    parser.add_argument(
        "--plot-error",
        action="store_true",
        help="Decide whether or not to compute the hyperbolic alignment of "
        "the output of the original eigenvectors.",
    )

    parser.add_argument(
        "--igraph",
        action="store_true",
        help="Do you have iGraph for Python, and is it working correctly?",
    )

    return parser.parse_args()


args = parse_args()
reduction_target = args.reduction_target
action_switch = args.action_switch
num_samples = args.num_samples
qos = args.qos
min_prob_per_act = args.min_prob_per_act
min_target_items = args.min_target_items
plot_error = args.plot_error
igraph = args.igraph

edgelist = np.load(args.input_file)

print("Starting")
# Because I couldn't find a general 'flatten' in Python.
flatten = lambda l: [item for sublist in l for item in sublist]
if igraph:
    # Import iGraph if you have it.
    from igraph import *

    def print_graph(graph, layout="auto"):
        iGraph = Graph()
        iGraph.add_vertices(g.nodeList)
        iGraph.add_edges(g.edgeList)
        plot(iGraph, vertex_color=[0, 0, 0], layout=layout)

if min_target_items == "none":
    if reduction_target == "nodes":
        min_target_items = 2
    elif reduction_target == "edges" and action_switch == "both":
        min_target_items = 1
    elif reduction_target == "edges" and action_switch == "delete":
        min_target_items = len(set(flatten(edgelist)))

# In case the graph becomes disconnected, try again.
connected = False
while not connected:
    print("Initializing GLGraph")
    g = GLGraph(
        edgelist,
        edgeWeights="none",
        nodeWeights="none",
        plot_error=plot_error,
        layout="random",
    )
    edgeNumList = []  # List of edges in the reduced graphs if target is 'edges'.
    nodeNumList = []  # List of nodes in the reduced graphs if target is 'nodes'.
    eigenAlignList = (
        []
    )  # List of hyperbolic distance of eigenvector output if plot_error is True.
    edgeNumList.append(len(g.edgeList))
    nodeNumList.append(len(g.nodeList))

    if plot_error:
        eigenAlignList.append(g.get_eigenvector_alignment()[1:])

    iteration = 0
    while True:
        iteration += 1
        # Say where we are in the reduction.
        if np.mod(iteration, 1) == 0:
            print(
                f"Iteration {iteration}, {len(g.edgeList)} "
                f"/ {len(g.edgeListIn)} edges, "
                f"{len(g.nodeList)} / {len(g.nodeListIn)} nodes",
                end="\r",
                flush=True,
            )

        if qos > 0:
            # If q is determined by a fraction of s, use reduce_graph_multi_edge,
            # as the edges should form a matching.
            g.reduce_graph_multi_edge(
                num_samples=num_samples,
                qFraction=qos,
                pMin=min_prob_per_act,
                reduction_type=action_switch,
                reduction_target=reduction_target,
                maxReweightFactor=0,
            )
        else:
            # If q is fixed at 1 (ie, qos==0), use reduce_graph_single_edge,
            # as we do not care if the edges form a matching.
            g.reduce_graph_single_edge(
                minSamples=num_samples,
                pMin=min_prob_per_act,
                reduction_type=action_switch,
                reduction_target=reduction_target,
                maxReweightFactor=0,
            )

        # If targeting nodes, save data whenever the number of nodes is reduced.
        if reduction_target == "nodes":
            if len(g.nodeList) < nodeNumList[-1]:
                edgeNumList.append(len(g.edgeList))
                nodeNumList.append(len(g.nodeList))
                if plot_error:
                    eigenAlignList.append(g.get_eigenvector_alignment()[1:])

        # If targeting edges, save data whenever the number of edges is reduced.
        if reduction_target == "edges":
            if len(g.edgeList) < edgeNumList[-1]:
                edgeNumList.append(len(g.edgeList))
                nodeNumList.append(len(g.nodeList))
                if plot_error:
                    eigenAlignList.append(g.get_eigenvector_alignment()[1:])

        # If we can merge nodes, go until there are only two left.
        if action_switch == "both":
            if (
                len(g.nodeList) < 3
                or (reduction_target == "edges" and len(g.edgeList) < min_target_items)
                or (reduction_target == "nodes" and len(g.nodeList) < min_target_items)
            ):
                break

        # If we cannot merge nodes, go until we have a spanning tree.
        if action_switch == "delete":
            if len(g.edgeList) < len(nodes) or (
                reduction_target == "edges" and len(g.edgeList) < min_target_items
            ):
                break

    if igraph:
        # If you have iGraph, use to check if the resulting graph was
        # disconnected (should not happen for qos==0)
        iGraph = Graph()
        iGraph.add_vertices(g.nodeList)
        iGraph.add_edges(g.edgeList)
        if not iGraph.is_connected():
            print("Whoops! Disconnected. Retrying.")
        else:
            connected = True
    else:
        # If your iGraph is not working, just hope that everything worked.
        # We will implement an iGraph-independent version soon.
        print(
            "Did not check if graph is disconnected: "
            "Results invalid if graph is disconnected."
        )
        connected = True

##########################################
### Outputs.
##########################################

g.update_inverse_laplacian()

# This is the reduced node-weighted laplacian of size $\tilde{V} \times \tilde{V}$
reducedLaplacian = g.nodeWeightedInverseLaplacian

# This is the reduced node-weighted laplacian appropriately projected back to
# $V \times V$. Use this to get approximate solutions to your Lx=b problems.
reducedLaplacianOriginalDimension = g.project_reduced_to_original(reducedLaplacian)

if igraph:
    print_graph(g)
