import random
import time
import logging

import numpy as np

flatten = lambda l: [item for sublist in l for item in sublist]


class GLGraph(object):
    def __init__(
        self,
        edges,
        edge_weights="none",
        node_weights="none",
        plot_error=False,
        layout="random",
    ):
        logging.info("Making GLGraph")
        startTime = time.perf_counter()
        self.thereIsAProblem = False
        self.edges_in = np.array(edges)
        self.nodes_in = sorted(list(set(flatten(self.edges_in))))

        if len(np.shape(edge_weights)) == 0:
            self.edge_weights_in = np.ones(len(self.edges_in))
        else:
            self.edge_weights_in = np.array(edge_weights)

        if len(np.shape(node_weights)) == 0:
            self.node_weights_in = np.ones(len(self.nodes_in))
        else:
            self.node_weights_in = np.array(node_weights)

        # setting up the reduced graph
        self.edges = np.copy(self.edges_in)
        self.nodes = np.copy(self.nodes_in)
        self.edge_weight_list = np.copy(self.edge_weights_in)
        self.node_weight_list = np.copy(self.node_weights_in)
        self.node_weight_list_old = np.copy(self.node_weights_in)

        # making matrices
        logging.info("making matrices")
        self.adjacency = self.make_adjacency(
            self.edges_in, self.nodes_in, self.edge_weights_in
        )
        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.node_weighted_lap_in = (((self.laplacian).T) / self.node_weights_in).T
        self.node_weighted_lap = np.copy(self.node_weighted_lap_in)
        self.jMatIn = np.outer(
            np.ones(len(self.node_weights_in)), self.node_weights_in
        ) / np.sum(self.node_weights_in)
        self.jMat = np.copy(self.jMatIn)
        self.contractedNodesToNodes = np.identity(len(self.nodes_in))

        # initializing layout
        logging.info("making layout")
        if len(np.shape(layout)) == 0:
            if layout == "random":
                self.layout = np.array(
                    [tuple(np.random.random(2)) for item in range(len(self.nodes_in))]
                )
                self.boundaries = np.array([[0.0, 0.0], [1.0, 1.0]])
            else:
                # making igraph object
                logging.info("making graph")
                import igraph as ig

                self.igraphIn = ig.Graph()
                (self.igraphIn).add_vertices(self.nodes_in)
                (self.igraphIn).add_edges(self.edges_in)
                self.layout = (self.igraphIn).layout(layout)
                self.boundaries = np.array((self.layout).boundaries())
                boundaryTemp = np.max(
                    [np.max(self.boundaries), np.max(-self.boundaries)]
                )
                self.boundaries = np.array(
                    [[-boundaryTemp, -boundaryTemp], [boundaryTemp, boundaryTemp]]
                )
        else:
            boundaryTemp = np.max([np.max(layout), -np.min(layout)])
            self.boundaries = np.array(
                [[-boundaryTemp, -boundaryTemp], [boundaryTemp, boundaryTemp]]
            )
            self.layout = np.array([tuple(item) for item in layout])

        # computing the inverse and initial eigenvectors
        logging.info("making inverses")
        self.node_weighted_inv_lap_in = self.invert_laplacian(
            self.node_weighted_lap_in, self.jMat
        )
        self.node_weighted_inv_lap = np.copy(self.node_weighted_inv_lap_in)
        if not plot_error:
            self.eigenvaluesIn = np.zeros(len(self.node_weighted_lap_in))
            self.eigenvectorsIn = np.zeros(np.shape(self.node_weighted_lap_in))
        else:
            eigenvaluesTemp, eigenvectorsTemp = np.linalg.eig(
                self.node_weighted_lap_in
            )
            orderTemp = np.argsort(eigenvaluesTemp)
            self.eigenvaluesIn = eigenvaluesTemp[orderTemp]
            self.eigenvectorsIn = eigenvectorsTemp.T[orderTemp]
        self.originalEigenvectorOutput = np.array(
            [
                np.dot(self.node_weighted_inv_lap_in, eigVec)
                for eigVec in self.eigenvectorsIn
            ]
        )
        self.updatedInverses = True
        self.updateList = []
        self.rowsToDelete = []

        endTime = time.perf_counter()
        logging.log(15, f"__init__: {endTime - startTime}")

    def make_adjacency(
        self, edges_in, nodes_in=["none"], edge_weight_list_in=["none"]
    ):
        if np.any([item == "none" for item in nodes_in]):
            nodeListTemp = sorted(list(set(flatten(edges_in))))
        else:
            nodeListTemp = np.array(nodes_in)
        if np.any([item == "none" for item in edge_weight_list_in]):
            edge_weight_list_temp = np.ones(len(edges_in))
        else:
            edge_weight_list_temp = np.array(edge_weight_list_in)

        adjOut = np.zeros((len(nodeListTemp), len(nodeListTemp)))
        for index, edge in enumerate(edges_in):
            position0 = list(nodeListTemp).index(edge[0])
            position1 = list(nodeListTemp).index(edge[1])
            adjOut[position0, position1] += edge_weight_list_temp[index]
            adjOut[position1, position0] += edge_weight_list_temp[index]
        return adjOut

    def adjacency_to_laplacian(self, adjIn):
        lapOut = np.copy(-adjIn)
        for index in range(len(adjIn)):
            lapOut[index, index] = -np.sum(lapOut[index])
        return lapOut

    def invert_laplacian(self, lapIn, jMatIn):
        return np.linalg.inv(lapIn + jMatIn) - jMatIn

    def hyperbolic_distance(self, vector0, vector1):
        hyperbolicDistance = np.arccosh(
            1.0
            + (np.linalg.norm(np.array(vector1) - np.array(vector0))) ** 2
            / (2 * np.dot(np.array(vector0), np.array(vector1)))
        )
        if np.isnan(hyperbolicDistance):
            print("NAN in compare_vectors")
        return hyperbolicDistance

    def project_reduced_to_original(self, matIn):
        return np.dot(
            self.contractedNodesToNodes,
            np.dot(
                matIn,
                np.dot(
                    np.diag(1.0 / self.node_weight_list), self.contractedNodesToNodes.T
                ),
            ),
        )

    def get_eigenvector_alignment(self):
        if not self.updatedInverses:
            self.update_inverse_laplacian()
        distanceListOut = np.zeros(len(self.originalEigenvectorOutput))
        projectedNodeWeightedInverseLaplacian = self.project_reduced_to_original(
            self.node_weighted_inv_lap
        )
        for index in range(len(self.originalEigenvectorOutput)):
            distanceListOut[index] = self.hyperbolic_distance(
                self.originalEigenvectorOutput[index],
                np.dot(
                    projectedNodeWeightedInverseLaplacian, self.eigenvectorsIn[index]
                ),
            )
        return distanceListOut

    def make_wOmega_m_tau(self, method="random", num_samples=0):
        startTime = time.perf_counter()
        if method == "random":
            if num_samples == 0:
                edgesToSample = range(len(self.edge_weight_list))
            elif num_samples >= len(self.edge_weight_list):
                edgesToSample = range(len(self.edge_weight_list))
            else:
                edgesToSample = sorted(
                    np.random.choice(
                        len(self.edge_weight_list), num_samples, replace=False
                    )
                )
        elif method == "RM":
            edgesToSample = self.get_edgeList_proposal_RM(num_samples)
        effectiveResistanceOut = np.zeros(len(edgesToSample))
        edgeImportanceOut = np.zeros(len(edgesToSample))
        numTrianglesOut = np.zeros(len(edgesToSample))

        for index, edgeNum in enumerate(edgesToSample):
            vertex0 = self.edges[edgeNum][0]
            vertex1 = self.edges[edgeNum][1]
            invDotUTemp = (
                self.node_weighted_inv_lap[:, vertex0]
                / self.node_weight_list[vertex0]
                - self.node_weighted_inv_lap[:, vertex1]
                / self.node_weight_list[vertex1]
            )
            vTempDotInv = (
                self.node_weighted_inv_lap[vertex0]
                - self.node_weighted_inv_lap[vertex1]
            )
            effectiveResistanceOut[index] = invDotUTemp[vertex0] - invDotUTemp[vertex1]
            edgeImportanceOut[index] = np.dot(invDotUTemp, vTempDotInv)
            neighbors0 = [
                indexInner
                for indexInner, item in enumerate(self.adjacency[vertex0])
                if item > 0
            ]
            neighbors1 = [
                indexInner
                for indexInner, item in enumerate(self.adjacency[vertex1])
                if item > 0
            ]
            numTrianglesOut[index] = len(
                [item for item in neighbors0 if item in neighbors1]
            )

        endTime = time.perf_counter()
        logging.log(15, f"make_wOmega_m_tau: {endTime - startTime}")
        return [
            edgesToSample,
            effectiveResistanceOut * self.edge_weight_list[edgesToSample],
            edgeImportanceOut * self.edge_weight_list[edgesToSample],
            numTrianglesOut,
        ]

    def wOmega_m_to_betaStar(
        self,
        wOmegaIn,
        mIn,
        tauIn,
        p_min=0.125,
        reduction_type="both",
        reduction_target="edges",
        max_reweight_factor=0,
    ):
        startTime = time.perf_counter()
        if reduction_type == "delete" and reduction_target == "nodes":
            print("Cannot do deletion only when targeting reduction of nodes")
            return
        if wOmegaIn < -1.0e-12 or wOmegaIn > 1.0 + 1.0e-12:
            print("ERROR IN WR")
        if reduction_target == "edges":
            if reduction_type == "delete":
                if wOmegaIn > 1.0 - 10e-6:
                    return [0.0, [0.0, 0.0, 1.0, 1.0]]
                minBetaStarTemp = mIn / (1 - wOmegaIn) / (1 - p_min)
                deletionProbTemp = p_min
                contractionProbTemp = 0.0
                reweightProbTemp = 1.0 - p_min
                reweightFactorTemp = (1.0 - deletionProbTemp / (1.0 - wOmegaIn)) ** -1
                if max_reweight_factor > 0:
                    if deletionProbTemp > (1.0 - max_reweight_factor ** -1) * (
                        1.0 - wOmegaIn
                    ):
                        deletionProbTemp = (1.0 - max_reweight_factor ** -1) * (
                            1.0 - wOmegaIn
                        )
                        reweightProbTemp = 1.0 - deletionProbTemp
                        minBetaStarTemp = mIn / (1 - wOmegaIn) / (1 - deletionProbTemp)
                        reweightFactorTemp = (
                            1.0 - deletionProbTemp / (1.0 - wOmegaIn)
                        ) ** -1
                actionProbReweightTemp = [
                    deletionProbTemp,
                    contractionProbTemp,
                    reweightProbTemp,
                    reweightFactorTemp,
                ]
            elif reduction_type == "contract":
                minBetaStarTemp = mIn / wOmegaIn / (1.0 - p_min) / (1.0 + tauIn) ** 0.5
                deletionProbTemp = 0.0
                contractionProbTemp = p_min
                reweightProbTemp = 1.0 - p_min
                reweightFactorTemp = 1.0 - contractionProbTemp / wOmegaIn
                if contractionProbTemp > wOmegaIn:
                    minBetaStarTemp = (
                        mIn / wOmegaIn / (1.0 - wOmegaIn) / (1 + (1.0 + tauIn) ** 0.5)
                    )
                    deletionProbTemp = 1.0 - wOmegaIn
                    contractionProbTemp = wOmegaIn
                    reweightProbTemp = 0.0
                    reweightFactorTemp = 1.0
                actionProbReweightTemp = [
                    deletionProbTemp,
                    contractionProbTemp,
                    reweightProbTemp,
                    reweightFactorTemp,
                ]
            elif reduction_type == "both":
                if wOmegaIn > 1.0 - 10e-14:
                    minBetaStarTemp = (
                        mIn / wOmegaIn / (1.0 - p_min) / (1.0 + tauIn) ** 0.5
                    )
                    deletionProbTemp = 0.0
                    contractionProbTemp = p_min
                    reweightProbTemp = 1.0 - p_min
                    reweightFactorTemp = 1.0 - contractionProbTemp / wOmegaIn
                    if max_reweight_factor > 0:
                        if reweightFactorTemp < max_reweight_factor ** -1:
                            contractionProbTemp = (1.0 - max_reweight_factor ** -1) * (
                                wOmegaIn
                            )
                            reweightProbTemp = 1.0 - contractionProbTemp
                            minBetaStarTemp = (
                                mIn
                                / wOmegaIn
                                / (1.0 - contractionProbTemp)
                                / (1.0 + tauIn) ** 0.5
                            )
                            reweightFactorTemp = 1.0 - contractionProbTemp / wOmegaIn
                    actionProbReweightTemp = [
                        deletionProbTemp,
                        contractionProbTemp,
                        reweightProbTemp,
                        reweightFactorTemp,
                    ]
                else:
                    minBetaStarTempList = [
                        mIn / (1.0 - wOmegaIn) / (1.0 - p_min),
                        mIn / wOmegaIn / (1.0 - p_min) / (1.0 + tauIn) ** 0.5,
                    ]
                    minBetaStarIndex = np.argmin(minBetaStarTempList)
                    if (
                        minBetaStarIndex == 0
                        and minBetaStarTempList[0] != minBetaStarTempList[1]
                    ):
                        minBetaStarTemp = minBetaStarTempList[0]
                        deletionProbTemp = p_min
                        contractionProbTemp = 0.0
                        reweightProbTemp = 1.0 - p_min
                        reweightFactorTemp = (
                            1.0 - deletionProbTemp / (1.0 - wOmegaIn)
                        ) ** -1
                    else:
                        minBetaStarTemp = minBetaStarTempList[1]
                        deletionProbTemp = 0.0
                        contractionProbTemp = p_min
                        reweightProbTemp = 1.0 - p_min
                        reweightFactorTemp = 1.0 - contractionProbTemp / wOmegaIn
                    if contractionProbTemp > wOmegaIn:
                        minBetaStarTemp = (
                            mIn
                            / wOmegaIn
                            / (1.0 - wOmegaIn)
                            / (1 + (1.0 + tauIn) ** 0.5)
                        )
                        deletionProbTemp = 1.0 - wOmegaIn
                        contractionProbTemp = wOmegaIn
                        reweightProbTemp = 0.0
                        reweightFactorTemp = 1.0
                    if deletionProbTemp > 1.0 - wOmegaIn:
                        minBetaStarTemp = (
                            mIn
                            / wOmegaIn
                            / (1.0 - wOmegaIn)
                            / (1 + (1.0 + tauIn) ** 0.5)
                        )
                        deletionProbTemp = 1.0 - wOmegaIn
                        contractionProbTemp = wOmegaIn
                        reweightProbTemp = 0.0
                        reweightFactorTemp = 1.0
                    actionProbReweightTemp = [
                        deletionProbTemp,
                        contractionProbTemp,
                        reweightProbTemp,
                        reweightFactorTemp,
                    ]

        if reduction_target == "nodes":
            minBetaStarTemp = mIn / wOmegaIn / (1.0 - p_min)
            deletionProbTemp = 0.0
            contractionProbTemp = p_min
            reweightProbTemp = 1.0 - p_min
            reweightFactorTemp = 1.0 - contractionProbTemp / wOmegaIn
            if contractionProbTemp > wOmegaIn:
                minBetaStarTemp = mIn / wOmegaIn / (1.0 - wOmegaIn)
                deletionProbTemp = 1.0 - wOmegaIn
                contractionProbTemp = wOmegaIn
                reweightProbTemp = 0.0
                reweightFactorTemp = 1.0
            actionProbReweightTemp = [
                deletionProbTemp,
                contractionProbTemp,
                reweightProbTemp,
                reweightFactorTemp,
            ]

        endTime = time.perf_counter()
        logging.log(15, f"wOmega_m_to_betaStar: {endTime - startTime}")
        return minBetaStarTemp, actionProbReweightTemp

    def wOmega_m_to_betaStarList(
        self,
        wOmegaListIn,
        mListIn,
        tauListIn,
        p_min=0.125,
        reduction_type="both",
        reduction_target="edges",
        max_reweight_factor=0,
    ):
        startTime = time.perf_counter()
        minBetaStarListOut = np.zeros(len(wOmegaListIn))
        actionProbReweightListOut = np.zeros((len(wOmegaListIn), 4))
        for index in range(len(wOmegaListIn)):
            minBetaStarTemp, actionProbReweightTemp = self.wOmega_m_to_betaStar(
                wOmegaListIn[index],
                mListIn[index],
                tauListIn[index],
                p_min=p_min,
                reduction_type=reduction_type,
                reduction_target=reduction_target,
                max_reweight_factor=max_reweight_factor,
            )
            minBetaStarListOut[index] = minBetaStarTemp
            actionProbReweightListOut[index] = actionProbReweightTemp
        endTime = time.perf_counter()
        logging.log(15, f"wOmega_m_to_betaStarList: {endTime - startTime}")
        return minBetaStarListOut, actionProbReweightListOut

    def reduce_graph_single_edge(
        self,
        num_samples=1,
        p_min=0.125,
        reduction_type="both",
        reduction_target="edges",
        max_reweight_factor=0,
    ):
        startTime = time.perf_counter()
        if not self.updatedInverses:
            self.update_inverse_laplacian()
        (
            sampledEdgeList,
            sampledWOmegaList,
            sampledMList,
            sampledTauList,
        ) = self.make_wOmega_m_tau(method="random", num_samples=num_samples)
        (
            sampledMinBetaStarList,
            sampledActionProbReweightList,
        ) = self.wOmega_m_to_betaStarList(
            sampledWOmegaList,
            sampledMList,
            sampledTauList,
            p_min=p_min,
            reduction_type=reduction_type,
            reduction_target=reduction_target,
            max_reweight_factor=max_reweight_factor,
        )
        nonzeroIndices = [
            index
            for index, item in enumerate(sampledActionProbReweightList)
            if not (item[0] == 0.0 and item[1] == 0.0)
        ]
        if len(nonzeroIndices) == 0:
            return
        chosenEdgeIndex = nonzeroIndices[
            np.argmin(sampledMinBetaStarList[nonzeroIndices])
        ]

        chosenEdgeRealIndex = sampledEdgeList[chosenEdgeIndex]
        chosenActionProbReweight = sampledActionProbReweightList[chosenEdgeIndex]
        edgeActionProbs = chosenActionProbReweight[0:3]
        edgeAction = np.random.choice(range(3), p=edgeActionProbs)

        if edgeAction == 0:
            logging.info(f"deleting edge {self.edges[chosenEdgeRealIndex]}")
            self.delete_edge(chosenEdgeRealIndex)
        if edgeAction == 1:
            logging.info(f"contracting edge {self.edges[chosenEdgeRealIndex]}")
            self.contract_edge(chosenEdgeRealIndex)
        if edgeAction == 2 and chosenActionProbReweight[3] != 1.0:
            logging.info(
                f"reweighting edge { self.edges[chosenEdgeRealIndex]} "
                f"by factor {chosenActionProbReweight[3]}",
            )
            self.reweight_edge(chosenEdgeRealIndex, chosenActionProbReweight[3])

        endTime = time.perf_counter()
        logging.log(15, f"reduce_graph_single_edge: {endTime - startTime}")

    def delete_edge(self, edgeIndexIn):
        startTime = time.perf_counter()
        changeTemp = -1.0 * self.edge_weight_list[edgeIndexIn]
        nodesTemp = self.edges[edgeIndexIn]
        self.adjacency[nodesTemp[0], nodesTemp[1]] = 0.0
        self.adjacency[nodesTemp[1], nodesTemp[0]] = 0.0
        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.node_weighted_lap = (((self.laplacian).T) / self.node_weight_list).T
        self.edges = np.delete(self.edges, edgeIndexIn, 0)
        self.edge_weight_list = np.delete(self.edge_weight_list, edgeIndexIn, 0)

        self.updatedInverses = False
        (self.updateList).append([nodesTemp, 1.0 / changeTemp])
        endTime = time.perf_counter()
        logging.log(15, f"delete_edge: {endTime - startTime}")

    def reweight_edge(self, edgeIndexIn, reweightFactorIn):
        startTime = time.perf_counter()
        changeTemp = (reweightFactorIn - 1.0) * self.edge_weight_list[edgeIndexIn]
        nodesTemp = self.edges[edgeIndexIn]
        self.adjacency[nodesTemp[0], nodesTemp[1]] += changeTemp
        self.adjacency[nodesTemp[1], nodesTemp[0]] += changeTemp
        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.node_weighted_lap = (((self.laplacian).T) / self.node_weight_list).T
        self.edge_weight_list[edgeIndexIn] += changeTemp

        self.updatedInverses = False
        (self.updateList).append([nodesTemp, 1.0 / changeTemp])
        endTime = time.perf_counter()
        logging.log(15, f"reweight_edge: {endTime - startTime}")

    def contract_edge(self, edgeIndexIn):
        startTime = time.perf_counter()
        nodesToContract = [
            int(self.edges[int(edgeIndexIn), 0]),
            int(self.edges[int(edgeIndexIn), 1]),
        ]
        edge_weight_to_contract = self.edge_weight_list[edgeIndexIn]
        layoutTemp = self.layout
        tempElementLayoutTemp = np.array(
            [
                (
                    layoutTemp[nodesToContract[0]][index]
                    * self.node_weight_list[nodesToContract[0]]
                    + layoutTemp[nodesToContract[1]][index]
                    * self.node_weight_list[nodesToContract[1]]
                )
                for index in range(len(layoutTemp[nodesToContract[0]]))
            ]
        ) / (
            self.node_weight_list[nodesToContract[0]]
            + self.node_weight_list[nodesToContract[1]]
        )
        layoutTemp[nodesToContract[0]] = tuple(tempElementLayoutTemp)
        if nodesToContract[1] == 0:
            layoutTemp = layoutTemp[(nodesToContract[1] + 1) :]
        elif nodesToContract[1] == len(layoutTemp) - 1:
            layoutTemp = layoutTemp[0 : nodesToContract[1]]
        else:
            layoutTemp = np.concatenate(
                (
                    layoutTemp[0 : nodesToContract[1]],
                    layoutTemp[(nodesToContract[1] + 1) :],
                )
            )
        self.layout = layoutTemp

        # self.node_weight_list_old = np.copy(self.node_weight_list)

        self.contractedNodesToNodes[
            :, nodesToContract[0]
        ] += self.contractedNodesToNodes[:, nodesToContract[1]]
        self.contractedNodesToNodes = (
            np.delete(self.contractedNodesToNodes.T, nodesToContract[1], 0)
        ).T

        self.nodes = np.delete(self.nodes, nodesToContract[1], 0)
        self.node_weight_list = np.dot(self.contractedNodesToNodes.T, self.node_weights_in)

        self.adjacency[nodesToContract[0], nodesToContract[1]] = 0.0
        self.adjacency[nodesToContract[1], nodesToContract[0]] = 0.0
        self.adjacency[nodesToContract[0], :] += self.adjacency[nodesToContract[1], :]
        self.adjacency[:, nodesToContract[0]] += self.adjacency[:, nodesToContract[1]]
        self.adjacency = np.delete(self.adjacency, nodesToContract[1], 0)
        self.adjacency = (np.delete(self.adjacency.T, nodesToContract[1], 0)).T

        edgeListTemp = []
        edge_weight_list_temp = []
        for i in range(len(self.adjacency)):
            for j in range(i, len(self.adjacency)):
                if self.adjacency[i, j] > 0:
                    edgeListTemp.append([i, j])
                    edge_weight_list_temp.append(self.adjacency[i, j])

        self.edges = np.array(edgeListTemp)
        self.edge_weight_list = np.array(edge_weight_list_temp)

        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.node_weighted_lap = (((self.laplacian).T) / self.node_weight_list).T

        self.updatedInverses = False
        (self.updateList).append([nodesToContract, 0.0])
        (self.rowsToDelete).append(nodesToContract)
        endTime = time.perf_counter()
        logging.log(15, f"contract_edge: {endTime - startTime}")

    def make_incidence_row(self, numTotalIn, edgeIn):
        rowOut = np.zeros(numTotalIn)
        rowOut[edgeIn[0]] = 1
        rowOut[edgeIn[1]] = -1
        return rowOut

    def update_inverse_laplacian(self):
        startTime = time.perf_counter()

        edgesToChange = [item[0] for item in self.updateList]
        inverseChange = [item[1] for item in self.updateList]

        incidenceTemp = np.array(
            [
                self.make_incidence_row(len(self.node_weight_list_old), edge)
                for edge in edgesToChange
            ]
        )

        uTemp = (incidenceTemp / self.node_weight_list_old).T
        vTemp = incidenceTemp

        try:
            easierInverse = np.linalg.inv(
                np.diag(inverseChange)
                + np.dot(vTemp, np.dot(self.node_weighted_inv_lap, uTemp))
            )
        except np.linalg.LinAlgError as err:
            if "Singular matrix" in str(err):
                self.thereIsAProblem = True
                print("Problem: singular matrix when updating Laplacian")
                return
            else:
                raise

        if np.shape(easierInverse) == (1, 1):
            invLapUpdate = -easierInverse[0, 0] * np.outer(
                np.dot(self.node_weighted_inv_lap, uTemp),
                np.dot(vTemp, self.node_weighted_inv_lap),
            )
        else:
            invLapUpdate = -np.dot(
                np.dot(np.dot(self.node_weighted_inv_lap, uTemp), easierInverse),
                np.dot(vTemp, self.node_weighted_inv_lap),
            )

        self.node_weighted_inv_lap += invLapUpdate
        if len(self.rowsToDelete) > 0:
            for rowToDelete in self.rowsToDelete:
                self.node_weighted_inv_lap[
                    :, rowToDelete[0]
                ] += self.node_weighted_inv_lap[:, rowToDelete[1]]
                self.node_weighted_inv_lap = np.delete(
                    self.node_weighted_inv_lap, rowToDelete[1], 0
                )
                self.node_weighted_inv_lap = (
                    np.delete(self.node_weighted_inv_lap.T, rowToDelete[1], 0)
                ).T

        self.updatedInverses = True
        self.updateList = []
        self.rowsToDelete = []
        self.node_weight_list_old = np.copy(self.node_weight_list)

        endTime = time.perf_counter()
        logging.log(15, f"update_inverse_laplacian: {endTime - startTime}")

    def get_edgeList_proposal_RM(self, num_samples_in=0):
        adjacencyTemp = self.adjacency
        edgeListTemp = list([list(item) for item in self.edges])
        randomNodeOrderTemp = np.random.permutation(len(adjacencyTemp))
        nodePairsOut = []
        matchedNodesTemp = []

        if num_samples_in == 0:
            num_samples = len(self.edges)
        else:
            num_samples = num_samples_in

        for firstNode in randomNodeOrderTemp:
            if firstNode not in matchedNodesTemp:
                unmatchedNeighborsTemp = [
                    index
                    for index, item in enumerate(adjacencyTemp[firstNode])
                    if item > 0 and index not in matchedNodesTemp
                ]
                if len(unmatchedNeighborsTemp) > 0:
                    secondNode = np.random.choice(unmatchedNeighborsTemp)
                    nodePairsOut.append(sorted([firstNode, secondNode]))
                    matchedNodesTemp.append(firstNode)
                    matchedNodesTemp.append(secondNode)
            if len(nodePairsOut) >= num_samples:
                break

        proposedEdgeListOut = [edgeListTemp.index(item) for item in nodePairsOut]
        return proposedEdgeListOut

    def reduce_graph_multi_edge(
        self,
        num_samples=0,
        q_frac=0.0625,
        p_min=0.125,
        reduction_type="both",
        reduction_target="edges",
        max_reweight_factor=0,
    ):
        if not self.updatedInverses:
            self.update_inverse_laplacian()
        (
            sampledEdgeList,
            sampledWOmegaList,
            sampledMList,
            sampledTauList,
        ) = self.make_wOmega_m_tau(method="RM", num_samples=num_samples)
        (
            sampledMinBetaStarList,
            sampledActionProbReweightList,
        ) = self.wOmega_m_to_betaStarList(
            sampledWOmegaList,
            sampledMList,
            sampledTauList,
            p_min=p_min,
            reduction_type=reduction_type,
            reduction_target=reduction_target,
            max_reweight_factor=max_reweight_factor,
        )
        nonzeroIndices = [
            index
            for index, item in enumerate(sampledActionProbReweightList)
            if not (item[0] == 0.0 and item[1] == 0.0)
        ]
        if len(nonzeroIndices) == 0:
            return

        numPerturbationsTemp = np.max([1, int(round(q_frac * len(nonzeroIndices)))])
        chosenEdgesIndices = np.array(nonzeroIndices)[
            list(
                np.argsort(np.array(sampledMinBetaStarList)[nonzeroIndices])[
                    :numPerturbationsTemp
                ]
            )
        ]
        chosenEdgesRealIndices = np.array(sampledEdgeList)[chosenEdgesIndices]
        chosenActionProbReweightList = np.array(sampledActionProbReweightList)[
            chosenEdgesIndices
        ]

        edgesToDelete = []
        edgesToContract = []
        for index, chosenEdgeRealIndex in enumerate(chosenEdgesRealIndices):

            edgeActionProbs = chosenActionProbReweightList[index][0:3]
            edgeAction = np.random.choice(range(3), p=edgeActionProbs)

            if edgeAction == 0:
                logging.info(f"deleting edge {chosenEdgeRealIndex}")
                edgesToDelete.append(chosenEdgeRealIndex)
            if edgeAction == 1:
                logging.info(f"contracting edge {chosenEdgeRealIndex}")
                edgesToContract.append(chosenEdgeRealIndex)
            if edgeAction == 2 and chosenActionProbReweightList[index][3] != 1.0:
                logging.info(
                    f"reweighting edge {chosenEdgeRealIndex}"
                    f" by factor {chosenActionProbReweightList[index][3]}",
                )
                self.reweight_edge(
                    chosenEdgeRealIndex, chosenActionProbReweightList[index][3]
                )
        edgesToDelete = sorted(edgesToDelete)

        contractSwitch = True
        if edgesToContract == []:
            shiftedEdgesToContract = []
            contractSwitch = False
        else:
            shiftedEdgesToContract = [
                int(
                    edgeToContract
                    - len([item for item in edgesToDelete if edgeToContract > item])
                )
                for edgeToContract in edgesToContract
            ]

        # self.node_weight_list_old = np.copy(self.node_weight_list)
        self.delete_multiple_edges(edgesToDelete)
        if contractSwitch:
            self.contract_multiple_edges(shiftedEdgesToContract)

    def delete_multiple_edges(self, edgeIndexListIn):
        startTime = time.perf_counter()
        for edgeIndex in edgeIndexListIn:
            changeTemp = -1.0 * self.edge_weight_list[edgeIndex]
            nodesTemp = self.edges[edgeIndex]
            self.adjacency[nodesTemp[0], nodesTemp[1]] = 0.0
            self.adjacency[nodesTemp[1], nodesTemp[0]] = 0.0
            (self.updateList).append([nodesTemp, 1.0 / changeTemp])

        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.node_weighted_lap = (((self.laplacian).T) / self.node_weight_list).T
        self.edges = np.delete(self.edges, edgeIndexListIn, 0)
        self.edge_weight_list = np.delete(self.edge_weight_list, edgeIndexListIn, 0)

        self.updatedInverses = False
        endTime = time.perf_counter()
        logging.log(15, f"delete_edge: {endTime - startTime}")

    def contract_multiple_edges(
        self, edgeIndexListIn
    ):  # ONLY WORKS WITH EDGES THAT DON'T SHARE NODES!!!
        startContractTime = time.perf_counter()

        nodesToContract = np.array(
            [
                sorted(
                    [int(self.edges[int(edge), 0]), int(self.edges[int(edge), 1])]
                )
                for edge in edgeIndexListIn
            ]
        )
        edgeSortingArgs = np.argsort(-np.array(nodesToContract[:, 1]))

        sortedNodesToContract = [nodesToContract[index] for index in edgeSortingArgs]
        sortedEdgesToContract = [edgeIndexListIn[index] for index in edgeSortingArgs]

        edge_weight_list_temp = np.array(
            [self.edge_weight_list[int(edge)] for edge in edgeIndexListIn]
        )
        sortedEdgeWeightListTemp = [
            edge_weight_list_temp[index] for index in edgeSortingArgs
        ]
        for index in range(len(edgeSortingArgs)):
            (self.updateList).append([sortedNodesToContract[index], 0.0])
            (self.rowsToDelete).append(sortedNodesToContract[index])

        for index, nodePair in enumerate(sortedNodesToContract):
            self.contract_nodePair(nodePair, sortedEdgeWeightListTemp[index])

    def contract_nodePair(self, nodePair, edge_weight_in=1.0):
        startTime = time.perf_counter()
        nodesToContract = nodePair
        edge_weight_to_contract = edge_weight_in
        layoutTemp = self.layout
        tempElementLayoutTemp = np.array(
            [
                (
                    layoutTemp[nodesToContract[0]][index]
                    * self.node_weight_list[nodesToContract[0]]
                    + layoutTemp[nodesToContract[1]][index]
                    * self.node_weight_list[nodesToContract[1]]
                )
                for index in range(len(layoutTemp[nodesToContract[0]]))
            ]
        ) / (
            self.node_weight_list[nodesToContract[0]]
            + self.node_weight_list[nodesToContract[1]]
        )
        layoutTemp[nodesToContract[0]] = tuple(tempElementLayoutTemp)
        if nodesToContract[1] == 0:
            layoutTemp = layoutTemp[(nodesToContract[1] + 1) :]
        elif nodesToContract[1] == len(layoutTemp) - 1:
            layoutTemp = layoutTemp[0 : nodesToContract[1]]
        else:
            layoutTemp = np.concatenate(
                (
                    layoutTemp[0 : nodesToContract[1]],
                    layoutTemp[(nodesToContract[1] + 1) :],
                )
            )
        self.layout = layoutTemp

        self.contractedNodesToNodes[
            :, nodesToContract[0]
        ] += self.contractedNodesToNodes[:, nodesToContract[1]]
        self.contractedNodesToNodes = (
            np.delete(self.contractedNodesToNodes.T, nodesToContract[1], 0)
        ).T

        self.nodes = np.delete(self.nodes, nodesToContract[1], 0)
        self.node_weight_list = np.dot(self.contractedNodesToNodes.T, self.node_weights_in)

        self.adjacency[nodesToContract[0], nodesToContract[1]] = 0.0
        self.adjacency[nodesToContract[1], nodesToContract[0]] = 0.0
        self.adjacency[nodesToContract[0], :] += self.adjacency[nodesToContract[1], :]
        self.adjacency[:, nodesToContract[0]] += self.adjacency[:, nodesToContract[1]]
        self.adjacency = np.delete(self.adjacency, nodesToContract[1], 0)
        self.adjacency = (np.delete(self.adjacency.T, nodesToContract[1], 0)).T

        edgeListTemp = []
        edge_weight_list_temp = []
        for i in range(len(self.adjacency)):
            for j in range(i, len(self.adjacency)):
                if self.adjacency[i, j] > 0:
                    edgeListTemp.append([i, j])
                    edge_weight_list_temp.append(self.adjacency[i, j])

        self.edges = np.array(edgeListTemp)
        self.edge_weight_list = np.array(edge_weight_list_temp)

        self.laplacian = self.adjacency_to_laplacian(self.adjacency)
        self.node_weighted_lap = (((self.laplacian).T) / self.node_weight_list).T

        self.updatedInverses = False
        endTime = time.perf_counter()
        logging.log(15, f"contract_nodePair: {endTime - startTime}")
