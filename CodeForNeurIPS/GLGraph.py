import random
import time
import logging
import itertools

import numpy as np


class Time:
    def __init__(self, level=15):
        self.level = level

    def __call__(self, func):
        def wrapped_func(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logging.log(self.level, f"{func.__name__}: {elapsed}")
            return result

        return wrapped_func


class GLGraph:
    @Time()
    def __init__(
        self,
        edges,
        edge_weights=None,
        node_weights=None,
        plot_error=False,
        layout="random",
    ):
        logging.info("Making GLGraph")
        self.problem = False
        self.edges_in = np.array(edges)
        self.nodes_in = sorted(set(itertools.chain.from_iterable(edges)))

        self.edge_weights_in = edge_weights
        self.node_weights_in = node_weights

        # setting up the reduced graph
        self.edges = self.edges_in.copy()
        self.nodes = self.nodes_in.copy()
        self.edge_weights = self.edge_weights_in.copy()
        self.node_weight_list = self.node_weights_in.copy()
        self.node_weight_list_old = self.node_weights_in.copy()

        # Making matrices
        logging.info("Making matrices")
        self.adj = self.make_adjacency(
            self.edges_in,
            self.nodes_in,
            self.edge_weights_in,
        )
        self.laplacian = self.adjacency_to_laplacian(self.adj)
        self.node_weighted_lap_in = ((self.laplacian).T / self.node_weights_in).T
        self.node_weighted_lap = self.node_weighted_lap_in.copy()
        self.j_mat_in = np.tile(
            self.node_weights_in,
            (len(self.node_weights_in), 1),
        ) / np.sum(self.node_weights_in)
        self.j_mat = self.j_mat_in.copy()
        self.contracted_nodes_to_nodes = np.identity(len(self.nodes_in))

        # initializing layout
        logging.info("Making layout")
        if len(np.shape(layout)) == 0:
            if layout == "random":
                self.layout = np.array(
                    [tuple(np.random.random(2)) for item in range(len(self.nodes_in))],
                )
                self.boundaries = np.array([[0.0, 0.0], [1.0, 1.0]])
            else:
                # Making igraph object
                logging.info("Making graph")
                import igraph as ig

                self.igraph_in = ig.Graph()
                (self.igraph_in).add_vertices(self.nodes_in)
                (self.igraph_in).add_edges(self.edges_in)
                self.layout = (self.igraph_in).layout(layout)
                self.boundaries = np.array((self.layout).boundaries())
                boundary_tmp = np.max(
                    [np.max(self.boundaries), np.max(-self.boundaries)],
                )
                self.boundaries = np.array(
                    [[-boundary_tmp, -boundary_tmp], [boundary_tmp, boundary_tmp]],
                )
        else:
            boundary_tmp = np.max([np.max(layout), -np.min(layout)])
            self.boundaries = np.array(
                [[-boundary_tmp, -boundary_tmp], [boundary_tmp, boundary_tmp]],
            )
            self.layout = np.array([tuple(item) for item in layout])

        # computing the inverse and initial eigenvectors
        logging.info("Making inverses")
        self.node_weighted_inv_lap_in = self.invert_laplacian(
            self.node_weighted_lap_in,
            self.j_mat,
        )
        self.node_weighted_inv_lap = self.node_weighted_inv_lap_in.copy()
        if not plot_error:
            self.eigvals_in = np.zeros(len(self.node_weighted_lap_in))
            self.eigvecs_in = np.zeros(np.shape(self.node_weighted_lap_in))
        else:
            eigvals_tmp, eigvecs_tmp = np.linalg.eig(self.node_weighted_lap_in)
            order_tmp = np.argsort(eigvals_tmp)
            self.eigvals_in = eigvals_tmp[order_tmp]
            self.eigvecs_in = eigvecs_tmp.T[order_tmp]
        self.orig_eigvec_out = np.array(
            [
                np.dot(self.node_weighted_inv_lap_in, eigvec)
                for eigvec in self.eigvecs_in
            ],
        )
        self.updated_inv = True
        self.update_list = []
        self.rows_to_del = []

    @property
    def edge_weights_in(self):
        return self._edge_weights_in

    @edge_weights_in.setter
    def edge_weights_in(self, edge_weights):
        if edge_weights is None:
            self._edge_weights_in = np.ones(len(self.edges_in))
        else:
            self._edge_weights_in = np.array(edge_weights)
    @property
    def node_weights_in(self):
        return self._node_weights_in

    @node_weights_in.setter
    def node_weights_in(self, node_weights):
        if node_weights is None:
            self._node_weights_in = np.ones(len(self.nodes_in))
        else:
            self._node_weights_in = np.array(node_weights)

    def make_adjacency(self, edges_in, nodes_in=["none"], edge_weights_in=["none"]):
        if np.any([item == "none" for item in nodes_in]):
            nodelist_tmp = sorted(set(itertools.chain.from_iterable(edges_in)))
        else:
            nodelist_tmp = np.array(nodes_in)
        if np.any([item == "none" for item in edge_weights_in]):
            edge_weights_tmp = np.ones(len(edges_in))
        else:
            edge_weights_tmp = np.array(edge_weights_in)

        adj_out = np.zeros((len(nodelist_tmp), len(nodelist_tmp)))
        for index, edge in enumerate(edges_in):
            position0 = list(nodelist_tmp).index(edge[0])
            position1 = list(nodelist_tmp).index(edge[1])
            adj_out[position0, position1] += edge_weights_tmp[index]
            adj_out[position1, position0] += edge_weights_tmp[index]
        return adj_out

    def adjacency_to_laplacian(self, adj_in):
        lap_out = -adj_in.copy()
        for index in range(len(adj_in)):
            lap_out[index, index] = -np.sum(lap_out[index])
        return lap_out

    def invert_laplacian(self, lap_in, j_mat_in):
        return np.linalg.inv(lap_in + j_mat_in) - j_mat_in

    def hyperbolic_distance(self, vector0, vector1):
        dist = np.arccosh(
            1.0
            + (np.linalg.norm(np.array(vector1) - np.array(vector0))) ** 2
            / (2 * np.dot(np.array(vector0), np.array(vector1))),
        )
        if np.isnan(dist):
            print("NAN in compare_vectors")
        return dist

    def project_reduced_to_original(self, mat_in):
        return np.dot(
            self.contracted_nodes_to_nodes,
            np.dot(
                mat_in,
                np.dot(
                    np.diag(1.0 / self.node_weight_list),
                    self.contracted_nodes_to_nodes.T,
                ),
            ),
        )

    def get_eigenvector_alignment(self):
        if not self.updated_inv:
            self.update_inverse_laplacian()
        dist_list_out = np.zeros(len(self.orig_eigvec_out))
        projected_node_weighted_inv_lap = self.project_reduced_to_original(
            self.node_weighted_inv_lap,
        )
        for index in range(len(self.orig_eigvec_out)):
            dist_list_out[index] = self.hyperbolic_distance(
                self.orig_eigvec_out[index],
                np.dot(projected_node_weighted_inv_lap, self.eigvecs_in[index]),
            )
        return dist_list_out

    @Time()
    def make_womega_m_tau(self, method="random", num_samples=0):
        if method == "random":
            if num_samples == 0:
                edges_to_sample = range(len(self.edge_weights))
            elif num_samples >= len(self.edge_weights):
                edges_to_sample = range(len(self.edge_weights))
            else:
                edges_to_sample = sorted(
                    np.random.choice(
                        len(self.edge_weights),
                        num_samples,
                        replace=False,
                    ),
                )
        elif method == "RM":
            edges_to_sample = self.get_edgelist_proposal_rm(num_samples)
        effective_resistence_out = np.zeros(len(edges_to_sample))
        edge_importance_out = np.zeros(len(edges_to_sample))
        num_triangles_out = np.zeros(len(edges_to_sample))

        for index, edge_num in enumerate(edges_to_sample):
            vertex0 = self.edges[edge_num][0]
            vertex1 = self.edges[edge_num][1]
            inv_dot_u_tmp = (
                self.node_weighted_inv_lap[:, vertex0] / self.node_weight_list[vertex0]
                - self.node_weighted_inv_lap[:, vertex1]
                / self.node_weight_list[vertex1]
            )
            v_tmp_dot_inv = (
                self.node_weighted_inv_lap[vertex0]
                - self.node_weighted_inv_lap[vertex1]
            )
            effective_resistence_out[index] = (
                inv_dot_u_tmp[vertex0] - inv_dot_u_tmp[vertex1]
            )
            edge_importance_out[index] = np.dot(inv_dot_u_tmp, v_tmp_dot_inv)
            neighbors0 = [
                index_inner
                for index_inner, item in enumerate(self.adj[vertex0])
                if item > 0
            ]
            neighbors1 = [
                index_inner
                for index_inner, item in enumerate(self.adj[vertex1])
                if item > 0
            ]
            num_triangles_out[index] = len(
                [item for item in neighbors0 if item in neighbors1],
            )

        return [
            edges_to_sample,
            effective_resistence_out * self.edge_weights[edges_to_sample],
            edge_importance_out * self.edge_weights[edges_to_sample],
            num_triangles_out,
        ]

    @Time()
    def womega_m_to_betastar(
        self,
        womega_in,
        m_in,
        tau_in,
        p_min=0.125,
        reduction_type="both",
        reduction_target="edges",
        max_reweight_factor=0,
    ):
        if reduction_type == "delete" and reduction_target == "nodes":
            print("Cannot do deletion only when targeting reduction of nodes")
            return
        if womega_in < -1.0e-12 or womega_in > 1.0 + 1.0e-12:
            print("ERROR IN WR")
        if reduction_target == "edges":
            if reduction_type == "delete":
                if womega_in > 1.0 - 10e-6:
                    return [0.0, [0.0, 0.0, 1.0, 1.0]]
                min_betastar_tmp = m_in / (1 - womega_in) / (1 - p_min)
                del_prob_tmp = p_min
                contraction_prob_tmp = 0.0
                reweight_prob_tmp = 1.0 - p_min
                reweight_factor_tmp = (1.0 - del_prob_tmp / (1.0 - womega_in)) ** -1
                if max_reweight_factor > 0:
                    if del_prob_tmp > (1.0 - max_reweight_factor ** -1) * (
                        1.0 - womega_in
                    ):
                        del_prob_tmp = (1.0 - max_reweight_factor ** -1) * (
                            1.0 - womega_in
                        )
                        reweight_prob_tmp = 1.0 - del_prob_tmp
                        min_betastar_tmp = m_in / (1 - womega_in) / (1 - del_prob_tmp)
                        reweight_factor_tmp = (
                            1.0 - del_prob_tmp / (1.0 - womega_in)
                        ) ** -1
                act_prob_reweight_tmp = [
                    del_prob_tmp,
                    contraction_prob_tmp,
                    reweight_prob_tmp,
                    reweight_factor_tmp,
                ]
            elif reduction_type == "contract":
                min_betastar_tmp = (
                    m_in / womega_in / (1.0 - p_min) / (1.0 + tau_in) ** 0.5
                )
                del_prob_tmp = 0.0
                contraction_prob_tmp = p_min
                reweight_prob_tmp = 1.0 - p_min
                reweight_factor_tmp = 1.0 - contraction_prob_tmp / womega_in
                if contraction_prob_tmp > womega_in:
                    min_betastar_tmp = (
                        m_in
                        / womega_in
                        / (1.0 - womega_in)
                        / (1 + (1.0 + tau_in) ** 0.5)
                    )
                    del_prob_tmp = 1.0 - womega_in
                    contraction_prob_tmp = womega_in
                    reweight_prob_tmp = 0.0
                    reweight_factor_tmp = 1.0
                act_prob_reweight_tmp = [
                    del_prob_tmp,
                    contraction_prob_tmp,
                    reweight_prob_tmp,
                    reweight_factor_tmp,
                ]
            elif reduction_type == "both":
                if womega_in > 1.0 - 10e-14:
                    min_betastar_tmp = (
                        m_in / womega_in / (1.0 - p_min) / (1.0 + tau_in) ** 0.5
                    )
                    del_prob_tmp = 0.0
                    contraction_prob_tmp = p_min
                    reweight_prob_tmp = 1.0 - p_min
                    reweight_factor_tmp = 1.0 - contraction_prob_tmp / womega_in
                    if max_reweight_factor > 0:
                        if reweight_factor_tmp < max_reweight_factor ** -1:
                            contraction_prob_tmp = (1.0 - max_reweight_factor ** -1) * (
                                womega_in
                            )
                            reweight_prob_tmp = 1.0 - contraction_prob_tmp
                            min_betastar_tmp = (
                                m_in
                                / womega_in
                                / (1.0 - contraction_prob_tmp)
                                / (1.0 + tau_in) ** 0.5
                            )
                            reweight_factor_tmp = 1.0 - contraction_prob_tmp / womega_in
                    act_prob_reweight_tmp = [
                        del_prob_tmp,
                        contraction_prob_tmp,
                        reweight_prob_tmp,
                        reweight_factor_tmp,
                    ]
                else:
                    min_betastar_tmp_list = [
                        m_in / (1.0 - womega_in) / (1.0 - p_min),
                        m_in / womega_in / (1.0 - p_min) / (1.0 + tau_in) ** 0.5,
                    ]
                    min_betastar_index = np.argmin(min_betastar_tmp_list)
                    if (
                        min_betastar_index == 0
                        and min_betastar_tmp_list[0] != min_betastar_tmp_list[1]
                    ):
                        min_betastar_tmp = min_betastar_tmp_list[0]
                        del_prob_tmp = p_min
                        contraction_prob_tmp = 0.0
                        reweight_prob_tmp = 1.0 - p_min
                        reweight_factor_tmp = (
                            1.0 - del_prob_tmp / (1.0 - womega_in)
                        ) ** -1
                    else:
                        min_betastar_tmp = min_betastar_tmp_list[1]
                        del_prob_tmp = 0.0
                        contraction_prob_tmp = p_min
                        reweight_prob_tmp = 1.0 - p_min
                        reweight_factor_tmp = 1.0 - contraction_prob_tmp / womega_in
                    if contraction_prob_tmp > womega_in:
                        min_betastar_tmp = (
                            m_in
                            / womega_in
                            / (1.0 - womega_in)
                            / (1 + (1.0 + tau_in) ** 0.5)
                        )
                        del_prob_tmp = 1.0 - womega_in
                        contraction_prob_tmp = womega_in
                        reweight_prob_tmp = 0.0
                        reweight_factor_tmp = 1.0
                    if del_prob_tmp > 1.0 - womega_in:
                        min_betastar_tmp = (
                            m_in
                            / womega_in
                            / (1.0 - womega_in)
                            / (1 + (1.0 + tau_in) ** 0.5)
                        )
                        del_prob_tmp = 1.0 - womega_in
                        contraction_prob_tmp = womega_in
                        reweight_prob_tmp = 0.0
                        reweight_factor_tmp = 1.0
                    act_prob_reweight_tmp = [
                        del_prob_tmp,
                        contraction_prob_tmp,
                        reweight_prob_tmp,
                        reweight_factor_tmp,
                    ]

        if reduction_target == "nodes":
            min_betastar_tmp = m_in / womega_in / (1.0 - p_min)
            del_prob_tmp = 0.0
            contraction_prob_tmp = p_min
            reweight_prob_tmp = 1.0 - p_min
            reweight_factor_tmp = 1.0 - contraction_prob_tmp / womega_in
            if contraction_prob_tmp > womega_in:
                min_betastar_tmp = m_in / womega_in / (1.0 - womega_in)
                del_prob_tmp = 1.0 - womega_in
                contraction_prob_tmp = womega_in
                reweight_prob_tmp = 0.0
                reweight_factor_tmp = 1.0
            act_prob_reweight_tmp = [
                del_prob_tmp,
                contraction_prob_tmp,
                reweight_prob_tmp,
                reweight_factor_tmp,
            ]

        return min_betastar_tmp, act_prob_reweight_tmp

    @Time()
    def womega_m_to_betastar_list(
        self,
        womega_list_in,
        m_list_in,
        tau_list_in,
        p_min=0.125,
        reduction_type="both",
        reduction_target="edges",
        max_reweight_factor=0,
    ):
        min_betastart_list_out = np.zeros(len(womega_list_in))
        act_prob_reweight_list_out = np.zeros((len(womega_list_in), 4))
        for index in range(len(womega_list_in)):
            min_betastar_tmp, act_prob_reweight_tmp = self.womega_m_to_betastar(
                womega_list_in[index],
                m_list_in[index],
                tau_list_in[index],
                p_min=p_min,
                reduction_type=reduction_type,
                reduction_target=reduction_target,
                max_reweight_factor=max_reweight_factor,
            )
            min_betastart_list_out[index] = min_betastar_tmp
            act_prob_reweight_list_out[index] = act_prob_reweight_tmp
        return min_betastart_list_out, act_prob_reweight_list_out

    @Time()
    def reduce_graph_single_edge(
        self,
        num_samples=1,
        p_min=0.125,
        reduction_type="both",
        reduction_target="edges",
        max_reweight_factor=0,
    ):
        if not self.updated_inv:
            self.update_inverse_laplacian()
        (
            sampled_edgelist,
            sampled_womega_list,
            sampled_m_list,
            sampled_tout_list,
        ) = self.make_womega_m_tau(method="random", num_samples=num_samples)
        (
            sampled_min_betastar_list,
            sampled_act_prob_reweight_list,
        ) = self.womega_m_to_betastar_list(
            sampled_womega_list,
            sampled_m_list,
            sampled_tout_list,
            p_min=p_min,
            reduction_type=reduction_type,
            reduction_target=reduction_target,
            max_reweight_factor=max_reweight_factor,
        )
        nonzeros = [
            index
            for index, item in enumerate(sampled_act_prob_reweight_list)
            if not (item[0] == 0.0 and item[1] == 0.0)
        ]
        if len(nonzeros) == 0:
            return
        chosen_edges = nonzeros[np.argmin(sampled_min_betastar_list[nonzeros])]

        chosen_edge_real_index = sampled_edgelist[chosen_edges]
        chosen_act_prob_reweight = sampled_act_prob_reweight_list[chosen_edges]
        edge_act_probs = chosen_act_prob_reweight[0:3]
        edge_act = np.random.choice(range(3), p=edge_act_probs)

        if edge_act == 0:
            logging.info(f"deleting edge {self.edges[chosen_edge_real_index]}")
            self.delete_edge(chosen_edge_real_index)
        if edge_act == 1:
            logging.info(f"contracting edge {self.edges[chosen_edge_real_index]}")
            self.contract_edge(chosen_edge_real_index)
        if edge_act == 2 and chosen_act_prob_reweight[3] != 1.0:
            logging.info(
                f"reweighting edge { self.edges[chosen_edge_real_index]} "
                f"by factor {chosen_act_prob_reweight[3]}",
            )
            self.reweight_edge(chosen_edge_real_index, chosen_act_prob_reweight[3])

    @Time()
    def delete_edge(self, edge_index_in):
        change_tmp = -1.0 * self.edge_weights[edge_index_in]
        nodes_tmp = self.edges[edge_index_in]
        self.adj[nodes_tmp[0], nodes_tmp[1]] = 0.0
        self.adj[nodes_tmp[1], nodes_tmp[0]] = 0.0
        self.laplacian = self.adjacency_to_laplacian(self.adj)
        self.node_weighted_lap = (((self.laplacian).T) / self.node_weight_list).T
        self.edges = np.delete(self.edges, edge_index_in, 0)
        self.edge_weights = np.delete(self.edge_weights, edge_index_in, 0)

        self.updated_inv = False
        (self.update_list).append([nodes_tmp, 1.0 / change_tmp])

    @Time()
    def reweight_edge(self, edge_index_in, reweight_fact_in):
        change_tmp = (reweight_fact_in - 1.0) * self.edge_weights[edge_index_in]
        nodes_tmp = self.edges[edge_index_in]
        self.adj[nodes_tmp[0], nodes_tmp[1]] += change_tmp
        self.adj[nodes_tmp[1], nodes_tmp[0]] += change_tmp
        self.laplacian = self.adjacency_to_laplacian(self.adj)
        self.node_weighted_lap = (((self.laplacian).T) / self.node_weight_list).T
        self.edge_weights[edge_index_in] += change_tmp

        self.updated_inv = False
        (self.update_list).append([nodes_tmp, 1.0 / change_tmp])

    @Time()
    def contract_edge(self, edge_index_in):
        nodes_to_contract = [
            int(self.edges[int(edge_index_in), 0]),
            int(self.edges[int(edge_index_in), 1]),
        ]
        edge_weight_to_contract = self.edge_weights[edge_index_in]
        layout_tmp = self.layout
        temp_element_layout_tmp = np.array(
            [
                (
                    layout_tmp[nodes_to_contract[0]][index]
                    * self.node_weight_list[nodes_to_contract[0]]
                    + layout_tmp[nodes_to_contract[1]][index]
                    * self.node_weight_list[nodes_to_contract[1]]
                )
                for index in range(len(layout_tmp[nodes_to_contract[0]]))
            ],
        ) / (
            self.node_weight_list[nodes_to_contract[0]]
            + self.node_weight_list[nodes_to_contract[1]]
        )
        layout_tmp[nodes_to_contract[0]] = tuple(temp_element_layout_tmp)
        if nodes_to_contract[1] == 0:
            layout_tmp = layout_tmp[(nodes_to_contract[1] + 1) :]
        elif nodes_to_contract[1] == len(layout_tmp) - 1:
            layout_tmp = layout_tmp[0 : nodes_to_contract[1]]
        else:
            layout_tmp = np.concatenate(
                (
                    layout_tmp[0 : nodes_to_contract[1]],
                    layout_tmp[(nodes_to_contract[1] + 1) :],
                ),
            )
        self.layout = layout_tmp

        # self.node_weight_list_old = self.node_weight_list.copy()

        self.contracted_nodes_to_nodes[
            :,
            nodes_to_contract[0],
        ] += self.contracted_nodes_to_nodes[:, nodes_to_contract[1]]
        self.contracted_nodes_to_nodes = (
            np.delete(self.contracted_nodes_to_nodes.T, nodes_to_contract[1], 0)
        ).T

        self.nodes = np.delete(self.nodes, nodes_to_contract[1], 0)
        self.node_weight_list = np.dot(
            self.contracted_nodes_to_nodes.T,
            self.node_weights_in,
        )

        self.adj[nodes_to_contract[0], nodes_to_contract[1]] = 0.0
        self.adj[nodes_to_contract[1], nodes_to_contract[0]] = 0.0
        self.adj[nodes_to_contract[0], :] += self.adj[nodes_to_contract[1], :]
        self.adj[:, nodes_to_contract[0]] += self.adj[:, nodes_to_contract[1]]
        self.adj = np.delete(self.adj, nodes_to_contract[1], 0)
        self.adj = (np.delete(self.adj.T, nodes_to_contract[1], 0)).T

        edgelist_tmp = []
        edge_weights_tmp = []
        for i in range(len(self.adj)):
            for j in range(i, len(self.adj)):
                if self.adj[i, j] > 0:
                    edgelist_tmp.append([i, j])
                    edge_weights_tmp.append(self.adj[i, j])

        self.edges = np.array(edgelist_tmp)
        self.edge_weights = np.array(edge_weights_tmp)

        self.laplacian = self.adjacency_to_laplacian(self.adj)
        self.node_weighted_lap = (((self.laplacian).T) / self.node_weight_list).T

        self.updated_inv = False
        (self.update_list).append([nodes_to_contract, 0.0])
        (self.rows_to_del).append(nodes_to_contract)

    def make_incidence_row(self, num_tot_in, edge_in):
        raw_out = np.zeros(num_tot_in)
        raw_out[edge_in[0]] = 1
        raw_out[edge_in[1]] = -1
        return raw_out

    @Time()
    def update_inverse_laplacian(self):
        edges_to_change = [item[0] for item in self.update_list]
        inv_change = [item[1] for item in self.update_list]

        indicence_tmp = np.array(
            [
                self.make_incidence_row(len(self.node_weight_list_old), edge)
                for edge in edges_to_change
            ],
        )

        u_tmp = (indicence_tmp / self.node_weight_list_old).T
        v_tmp = indicence_tmp

        try:
            easier_inv = np.linalg.inv(
                np.diag(inv_change)
                + np.dot(v_tmp, np.dot(self.node_weighted_inv_lap, u_tmp)),
            )
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                self.problem = True
                print("Problem: singular matrix when updating Laplacian")
                return
            else:
                raise

        if np.shape(easier_inv) == (1, 1):
            inv_lap_update = -easier_inv[0, 0] * np.outer(
                np.dot(self.node_weighted_inv_lap, u_tmp),
                np.dot(v_tmp, self.node_weighted_inv_lap),
            )
        else:
            inv_lap_update = -np.dot(
                np.dot(np.dot(self.node_weighted_inv_lap, u_tmp), easier_inv),
                np.dot(v_tmp, self.node_weighted_inv_lap),
            )

        self.node_weighted_inv_lap += inv_lap_update
        if len(self.rows_to_del) > 0:
            for row_to_del in self.rows_to_del:
                self.node_weighted_inv_lap[
                    :,
                    row_to_del[0],
                ] += self.node_weighted_inv_lap[:, row_to_del[1]]
                self.node_weighted_inv_lap = np.delete(
                    self.node_weighted_inv_lap,
                    row_to_del[1],
                    0,
                )
                self.node_weighted_inv_lap = (
                    np.delete(self.node_weighted_inv_lap.T, row_to_del[1], 0)
                ).T

        self.updated_inv = True
        self.update_list = []
        self.rows_to_del = []
        self.node_weight_list_old = self.node_weight_list.copy()

    def get_edgelist_proposal_rm(self, num_samples_in=0):
        adjacency_tmp = self.adj
        edgelist_tmp = list([list(item) for item in self.edges])
        rand_node_order_tmp = np.random.permutation(len(adjacency_tmp))
        node_pairs_out = []
        matched_nodes_tmp = []

        if num_samples_in == 0:
            num_samples = len(self.edges)
        else:
            num_samples = num_samples_in

        for node1 in rand_node_order_tmp:
            if node1 not in matched_nodes_tmp:
                unmatched_nbrs_tmp = [
                    index
                    for index, item in enumerate(adjacency_tmp[node1])
                    if item > 0 and index not in matched_nodes_tmp
                ]
                if len(unmatched_nbrs_tmp) > 0:
                    node2 = np.random.choice(unmatched_nbrs_tmp)
                    node_pairs_out.append(sorted([node1, node2]))
                    matched_nodes_tmp.append(node1)
                    matched_nodes_tmp.append(node2)
            if len(node_pairs_out) >= num_samples:
                break

        proposed_edgelist_out = [edgelist_tmp.index(item) for item in node_pairs_out]
        return proposed_edgelist_out

    def reduce_graph_multi_edge(
        self,
        num_samples=0,
        q_frac=0.0625,
        p_min=0.125,
        reduction_type="both",
        reduction_target="edges",
        max_reweight_factor=0,
    ):
        if not self.updated_inv:
            self.update_inverse_laplacian()
        (
            sampled_edgelist,
            sampled_womega_list,
            sampled_m_list,
            sampled_tout_list,
        ) = self.make_womega_m_tau(method="RM", num_samples=num_samples)
        (
            sampled_min_betastar_list,
            sampled_act_prob_reweight_list,
        ) = self.womega_m_to_betastar_list(
            sampled_womega_list,
            sampled_m_list,
            sampled_tout_list,
            p_min=p_min,
            reduction_type=reduction_type,
            reduction_target=reduction_target,
            max_reweight_factor=max_reweight_factor,
        )
        nonzeros = [
            index
            for index, item in enumerate(sampled_act_prob_reweight_list)
            if not (item[0] == 0.0 and item[1] == 0.0)
        ]
        if len(nonzeros) == 0:
            return

        num_pert_tmp = np.max([1, int(round(q_frac * len(nonzeros)))])
        chosen_edges_indices = np.array(nonzeros)[
            list(
                np.argsort(np.array(sampled_min_betastar_list)[nonzeros])[
                    :num_pert_tmp
                ],
            )
        ]
        chosen_edges_real_indices = np.array(sampled_edgelist)[chosen_edges_indices]
        chosen_act_prob_reweight_list = np.array(sampled_act_prob_reweight_list)[
            chosen_edges_indices
        ]

        edges_to_del = []
        edges_to_contract = []
        for index, chosen_edge_real_index in enumerate(chosen_edges_real_indices):

            edge_act_probs = chosen_act_prob_reweight_list[index][0:3]
            edge_act = np.random.choice(range(3), p=edge_act_probs)

            if edge_act == 0:
                logging.info(f"deleting edge {chosen_edge_real_index}")
                edges_to_del.append(chosen_edge_real_index)
            if edge_act == 1:
                logging.info(f"contracting edge {chosen_edge_real_index}")
                edges_to_contract.append(chosen_edge_real_index)
            if edge_act == 2 and chosen_act_prob_reweight_list[index][3] != 1.0:
                logging.info(
                    f"reweighting edge {chosen_edge_real_index}"
                    f" by factor {chosen_act_prob_reweight_list[index][3]}",
                )
                self.reweight_edge(
                    chosen_edge_real_index,
                    chosen_act_prob_reweight_list[index][3],
                )
        edges_to_del = sorted(edges_to_del)

        contract_switch = True
        if edges_to_contract == []:
            shifted_edges_to_contract = []
            contract_switch = False
        else:
            shifted_edges_to_contract = [
                int(
                    edge_to_contract
                    - len([item for item in edges_to_del if edge_to_contract > item]),
                )
                for edge_to_contract in edges_to_contract
            ]

        # self.node_weight_list_old = self.node_weight_list.copy()
        self.delete_multiple_edges(edges_to_del)
        if contract_switch:
            self.contract_multiple_edges(shifted_edges_to_contract)

    @Time()
    def delete_multiple_edges(self, edge_index_list_in):
        for edge_index in edge_index_list_in:
            change_tmp = -1.0 * self.edge_weights[edge_index]
            nodes_tmp = self.edges[edge_index]
            self.adj[nodes_tmp[0], nodes_tmp[1]] = 0.0
            self.adj[nodes_tmp[1], nodes_tmp[0]] = 0.0
            (self.update_list).append([nodes_tmp, 1.0 / change_tmp])

        self.laplacian = self.adjacency_to_laplacian(self.adj)
        self.node_weighted_lap = (((self.laplacian).T) / self.node_weight_list).T
        self.edges = np.delete(self.edges, edge_index_list_in, 0)
        self.edge_weights = np.delete(self.edge_weights, edge_index_list_in, 0)

        self.updated_inv = False

    def contract_multiple_edges(self, edge_index_list_in):
        # ONLY WORKS WITH EDGES THAT DON'T SHARE NODES!!!
        nodes_to_contract = np.array(
            [
                sorted([int(self.edges[int(edge), 0]), int(self.edges[int(edge), 1])])
                for edge in edge_index_list_in
            ],
        )
        edge_sorting_args = np.argsort(-np.array(nodes_to_contract[:, 1]))

        sorted_nodes_to_contract = [
            nodes_to_contract[index] for index in edge_sorting_args
        ]
        sorted_edges_to_contract = [
            edge_index_list_in[index] for index in edge_sorting_args
        ]

        edge_weights_tmp = np.array(
            [self.edge_weights[int(edge)] for edge in edge_index_list_in],
        )
        sorted_edge_weights_tmp = [
            edge_weights_tmp[index] for index in edge_sorting_args
        ]
        for index in range(len(edge_sorting_args)):
            (self.update_list).append([sorted_nodes_to_contract[index], 0.0])
            (self.rows_to_del).append(sorted_nodes_to_contract[index])

        for index, node_pair in enumerate(sorted_nodes_to_contract):
            self.contract_node_pair(node_pair, sorted_edge_weights_tmp[index])

    @Time()
    def contract_node_pair(self, node_pair, edge_weight_in=1.0):
        nodes_to_contract = node_pair
        edge_weight_to_contract = edge_weight_in
        layout_tmp = self.layout
        temp_element_layout_tmp = np.array(
            [
                (
                    layout_tmp[nodes_to_contract[0]][index]
                    * self.node_weight_list[nodes_to_contract[0]]
                    + layout_tmp[nodes_to_contract[1]][index]
                    * self.node_weight_list[nodes_to_contract[1]]
                )
                for index in range(len(layout_tmp[nodes_to_contract[0]]))
            ],
        ) / (
            self.node_weight_list[nodes_to_contract[0]]
            + self.node_weight_list[nodes_to_contract[1]]
        )
        layout_tmp[nodes_to_contract[0]] = tuple(temp_element_layout_tmp)
        if nodes_to_contract[1] == 0:
            layout_tmp = layout_tmp[(nodes_to_contract[1] + 1) :]
        elif nodes_to_contract[1] == len(layout_tmp) - 1:
            layout_tmp = layout_tmp[0 : nodes_to_contract[1]]
        else:
            layout_tmp = np.concatenate(
                (
                    layout_tmp[0 : nodes_to_contract[1]],
                    layout_tmp[(nodes_to_contract[1] + 1) :],
                ),
            )
        self.layout = layout_tmp

        self.contracted_nodes_to_nodes[
            :,
            nodes_to_contract[0],
        ] += self.contracted_nodes_to_nodes[:, nodes_to_contract[1]]
        self.contracted_nodes_to_nodes = (
            np.delete(self.contracted_nodes_to_nodes.T, nodes_to_contract[1], 0)
        ).T

        self.nodes = np.delete(self.nodes, nodes_to_contract[1], 0)
        self.node_weight_list = np.dot(
            self.contracted_nodes_to_nodes.T,
            self.node_weights_in,
        )

        self.adj[nodes_to_contract[0], nodes_to_contract[1]] = 0.0
        self.adj[nodes_to_contract[1], nodes_to_contract[0]] = 0.0
        self.adj[nodes_to_contract[0], :] += self.adj[nodes_to_contract[1], :]
        self.adj[:, nodes_to_contract[0]] += self.adj[:, nodes_to_contract[1]]
        self.adj = np.delete(self.adj, nodes_to_contract[1], 0)
        self.adj = (np.delete(self.adj.T, nodes_to_contract[1], 0)).T

        edgelist_tmp = []
        edge_weights_tmp = []
        for i in range(len(self.adj)):
            for j in range(i, len(self.adj)):
                if self.adj[i, j] > 0:
                    edgelist_tmp.append([i, j])
                    edge_weights_tmp.append(self.adj[i, j])

        self.edges = np.array(edgelist_tmp)
        self.edge_weights = np.array(edge_weights_tmp)

        self.laplacian = self.adjacency_to_laplacian(self.adj)
        self.node_weighted_lap = (((self.laplacian).T) / self.node_weight_list).T

        self.updated_inv = False
