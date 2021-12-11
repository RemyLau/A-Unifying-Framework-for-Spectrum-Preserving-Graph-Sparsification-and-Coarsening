import argparse
import logging

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
        "--has-igraph",
        action="store_true",
        help="Do you have iGraph for Python, and is it working correctly?",
    )

    parser.add_argument(
        "--log-level",
        default="NONE",
        choices=["NONE", "INFO", "TIME"],
        help="Logging leve: NONE to log nothing, INFO to log what its doing, "
        "TIME to log timing information",
    )

    return parser.parse_args()


def setup_logging(args):
    if args.log_level != "NONE":
        logging.TIME = 15
        logging.basicConfig(level=getattr(logging, args.log_level))


def main(args):
    reduction_target = args.reduction_target
    action_switch = args.action_switch
    num_samples = args.num_samples
    qos = args.qos
    min_prob_per_act = args.min_prob_per_act
    min_target_items = args.min_target_items
    plot_error = args.plot_error
    has_igraph = args.has_igraph
    log_level = args.log_level

    edgelist = np.load(args.input_file)

    if has_igraph:  # import iGraph if you have it
        from igraph import Graph, plot

        def print_graph(g, layout="auto"):
            igraph_graph = Graph()
            igraph_graph.add_vertices(g.nodes)
            igraph_graph.add_edges(g.edges)
            plot(igraph_graph, vertex_color=[0, 0, 0], layout=layout)

    print("Starting")
    if min_target_items == "none":
        if reduction_target == "nodes":
            min_target_items = 2
        elif reduction_target == "edges" and action_switch == "both":
            min_target_items = 1
        elif reduction_target == "edges" and action_switch == "delete":
            min_target_items = edgelist.shape[0]

    # In case the graph becomes disconnected, try again.
    connected = False
    while not connected:
        print("Initializing GLGraph")
        g = GLGraph(
            edgelist,
            edge_weights=None,
            node_weights=None,
            plot_error=plot_error,
            layout="random",
        )

        edge_num_list = []  # List of edges in the reduced graphs if target is 'edges'.
        node_num_list = []  # List of nodes in the reduced graphs if target is 'nodes'.
        # List of hyperbolic distance of eigenvector output if plot_error is True.
        eigen_align_list = []

        edge_num_list.append(len(g.edges))
        node_num_list.append(len(g.nodes))

        if plot_error:
            eigen_align_list.append(g.get_eigenvector_alignment()[1:])

        iteration = 0
        while True:
            iteration += 1
            # Say where we are in the reduction.
            if np.mod(iteration, 1) == 0 and log_level == "NONE":
                print(
                    f"Iteration {iteration}, {len(g.edges)} "
                    f"/ {len(g.edges_in)} edges, "
                    f"{len(g.nodes)} / {len(g.nodes)} nodes",
                    end="\r",
                    flush=True,
                )

            if qos > 0:
                # If q is determined by a fraction of s, use reduce_graph_multi_edge,
                # as the edges should form a matching.
                g.reduce_graph_multi_edge(
                    num_samples=num_samples,
                    q_frac=qos,
                    p_min=min_prob_per_act,
                    reduction_type=action_switch,
                    reduction_target=reduction_target,
                    max_reweight_factor=0,
                )
            else:
                # If q is fixed at 1 (ie, qos==0), use reduce_graph_single_edge,
                # as we do not care if the edges form a matching.
                g.reduce_graph_single_edge(
                    num_samples=num_samples,
                    p_min=min_prob_per_act,
                    reduction_type=action_switch,
                    reduction_target=reduction_target,
                    max_reweight_factor=0,
                )

            # If targeting nodes, save data whenever the number of nodes is reduced.
            if reduction_target == "nodes":
                if len(g.nodes) < node_num_list[-1]:
                    edge_num_list.append(len(g.edges))
                    node_num_list.append(len(g.nodes))
                    if plot_error:
                        eigen_align_list.append(g.get_eigenvector_alignment()[1:])

            # If targeting edges, save data whenever the number of edges is reduced.
            if reduction_target == "edges":
                if len(g.edges) < edge_num_list[-1]:
                    edge_num_list.append(len(g.edges))
                    node_num_list.append(len(g.nodes))
                    if plot_error:
                        eigen_align_list.append(g.get_eigenvector_alignment()[1:])

            # If we can merge nodes, go until there are only two left.
            if action_switch == "both":
                if (
                    len(g.nodes) < 3
                    or (reduction_target == "edges" and len(g.edges) < min_target_items)
                    or (reduction_target == "nodes" and len(g.nodes) < min_target_items)
                ):
                    break

            # If we cannot merge nodes, go until we have a spanning tree.
            if action_switch == "delete":
                if len(g.edges) < len(nodes) or (
                    reduction_target == "edges" and len(g.edges) < min_target_items
                ):
                    break

        if has_igraph:
            # If you have iGraph, use to check if the resulting graph was
            # disconnected (should not happen for qos==0)
            igraph_graph = Graph()
            igraph_graph.add_vertices(g.nodes)
            igraph_graph.add_edges(g.edges)
            if not igraph_graph.is_connected():
                print("Whoops! Disconnected. Retrying.")
            else:
                connected = True
        else:
            # If your iGraph is not working, just hope that everything worked.
            # We will implement an iGraph-independent version soon.
            print(
                "Did not check if graph is disconnected: "
                "Results invalid if graph is disconnected.",
            )
            connected = True

    ##########################################
    ### Outputs.
    ##########################################

    g.update_inverse_laplacian()

    # This is the reduced node-weighted laplacian of size $\tilde{V} \times \tilde{V}$
    reduced_lap = g.node_weighted_inv_lap

    # This is the reduced node-weighted laplacian appropriately projected back to
    # $V \times V$. Use this to get approximate solutions to your Lx=b problems.
    reduced_lap_orig_dim = g.project_reduced_to_original(reduced_lap)

    if has_igraph:
        print_graph(g)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args)
    main(args)
