"""
(Approximate) Traveling Salesman Problem (TSP) solutions for two dimensional
grids
"""

import multiprocessing
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.approximation import greedy_tsp, christofides
from python_tsp.heuristics import solve_tsp_lin_kernighan, solve_tsp_local_search

import fitting_polynomials_2d


# Number of processes
NPROC = 8

INITIALIZE_WITH_CHRISTOFIDES = True
# Timeout settings (all in seconds)
# 2-opt initial optimization
TIMEOUT_2OPT = 600
# Lin-Kernighan
CYCLE_LK = 1  # checks for completion every cycle
TIMEOUT_LK = 30  # total timeout

# Grid dimensions
ngrid = 28


def distance_matrix(xy):
    """Create symmetric distance matrix via numpy.

    Args:
        xy: array-like where each row contains the coordinates of a 2D point

    Returns:
        Square numpy array of dimensions len(xy) by len(xy), containing a
        symmetric matrix of euclidean distances between each pair of points.
    """
    return np.sqrt(
        ((np.atleast_2d(xy)[:, :, None] - np.atleast_2d(xy)[:, :, None].T)**2).sum(axis=1))


def add_edges_from_distance_matrix(graph, distance_matrix):
    """Add weighted edges to an input graph from the upper triangle of an input
    distance matrix.

    Args:
        graph:
            networkx.Graph (complete) of N nodes with edge distances provided by
            the input distance_matrix
        distance_matrix:
            float array-like of shape (N, N) containing, in its upper triangular
            section, the edge weights between each pair of nodes

    Returns: complete networkx.Graph with edge weights set by the distance_matrix
    """
    _nn = distance_matrix.shape[0]
    if _nn != len(graph.nodes):
        raise IndexError(f"mismatch between {distance_matrix.shape[0]=} and {len(graph.nodes)=}")

    for i in range(_nn):
        for j in range(1 + i, _nn):
            graph[i][j]["weight"] = distance_matrix[i, j]
    return graph


def plot_path(xy, path, figsize=(6, 5.5)):
    """Plot a path through xy coordinates.

    Args:
        xy: array-like where each row contains the coordinates of a 2D point
        path: index of xy rows defining the path to plot
    """
    x, y = zip(*[xy[_i] for _i in path])
    fig = plt.figure(figsize=figsize)
    plt.plot(x, y, "ko")
    plt.plot(x, y)
    plt.tight_layout()
    plt.show()


def process_local_search(
    pk,
    results_dict,
    distance_matrix=None,
    x0=None,
    max_processing_time=None,
    perturbation_scheme="two_opt",
):
    """Target function for multiprocessing approximate TSP solutions using local
    search heuristics.

    Stores output of call to python_tsp.heuristics.solve_tsp_local_search in
    results_dict[pk].

    Args:
        pk: process key for result storage
        results_dict:
            dict-like as returned by multiprocessing.Manager().dict(), to be
            used for shared memory storage of solution results
        distance_matrix: passed to solve_tsp_local_search
        x0: start path, if any, passed to solve_tsp_local_search
        max_processing_time: passed to solve_tsp_local_search
        perturbation_scheme: passed to solve_tsp_local_search
    """
    if distance_matrix is None:
        raise ValueError("distance_matrix must be supplied")

    results_dict[pk] = solve_tsp_local_search(
        distance_matrix=distance_matrix,
        x0=x0,
        perturbation_scheme=perturbation_scheme,
        max_processing_time=max_processing_time,
        verbose=False,
    )


def process_lin_kernigan(pk, results_dict, distance_matrix=None, x0=None):
    """Target function for multiprocessing approximate TSP solutions using the
    Lin-Kernighan algorithm.

    Stores output of call to python_tsp.heuristics.solve_tsp_lin_kernighan in
    results_dict[pk].

    Args:
        pk: process key for result storage
        results_dict:
            dict-like as returned by multiprocessing.Manager().dict(), to be
            used for shared memory storage of solution results
        distance_matrix: passed to solve_tsp_lin_kernighan
        x0: start path, if any, passed to solve_tsp_lin_kernighan
    """
    if distance_matrix is None:
        raise ValueError("distance_matrix must be supplied")

    results_dict[pk] = solve_tsp_lin_kernighan(
        distance_matrix=distance_matrix,
        x0=x0,
        verbose=False,
    )


def multiprocess_local_search(
    dm,
    x0=None,
    max_processing_time=TIMEOUT_2OPT,
    nproc=NPROC,
    perturbation_scheme="two_opt",
):
    """Launch and gather results from multiprocessing of approximate TSP
    solutions using local search heuristics.
    """
    processes = {}
    manager = multiprocessing.Manager()
    results_storage = manager.dict()
    if x0 is None:
        x0 = [None] * nproc

    for ip in range(nproc):
        print(f"Launching process {ip} for {perturbation_scheme} TSP, {max_processing_time=}s")
        processes[ip] = multiprocessing.Process(
            target=process_local_search,
            args=(ip, results_storage),
            kwargs={
                "distance_matrix": dm, "x0": x0[ip], "max_processing_time": max_processing_time
            },
        )
        processes[ip].start()

    for ip in range(nproc):
        processes[ip].join(max_processing_time)

    return results_storage


def multiprocess_lk(dm, x0=None, max_processing_time=TIMEOUT_LK, nproc=NPROC):
    """Launch and gather results from multiprocessing of approximate TSP
    solutions using the Lin-Kernighan algorithm.
    """
    processes = {}
    manager = multiprocessing.Manager()
    results_storage = manager.dict()
    if x0 is None:
        x0 = [None] * nproc

    for ip in range(nproc):
        print(f"Launching process {ip} for Lin-Kernigan TSP")
        processes[ip] = multiprocessing.Process(
            target=process_lin_kernigan,
            args=(ip, results_storage),
            kwargs={"distance_matrix": dm, "x0": x0[ip]},
        )
        processes[ip].start()

    # collect results
    t0 = time.time()
    marked_completed = [False] * nproc
    incomplete = list(range(nproc))
    while len(incomplete) > 0:
        time.sleep(CYCLE_LK)
        for ip in incomplete:
            if processes[ip].exitcode == 0:
                processes[ip].join(1)
                incomplete.remove(ip)
                print(f"Success: process={ip}, {len(incomplete)=}, elapsed={time.time() - t0}s")

        time_exceeded = (time.time() - t0) >= max_processing_time
        if time_exceeded:
            for ip in incomplete:
                if processes[ip].is_alive():
                    processes[ip].terminate()
                    print(
                        f"Failure: process={ip} terminated incomplete, {len(incomplete)=}, "
                        f"elapsed={time.time() - t0}s"
                    )
                processes[ip].join(1)
            break

    return zip(*results_storage.values())


if __name__ == "__main__":

    G = nx.complete_graph(ngrid**2)
    grid_points = [(i, j) for i in range(ngrid) for j in range(ngrid)]

    dm = distance_matrix(grid_points)
    # Add edges with weights = distance between points
    print(f"Adding edge distance weights for complete {ngrid}x{ngrid} grid graph")
    t0 = time.time()
    G = add_edges_from_distance_matrix(G, dm)
    print(f"Time taken: {time.time() - t0:.2f}s")

    if INITIALIZE_WITH_CHRISTOFIDES:
        print("Calculating Christofides approximation")
        tc = time.time()
        p0 = nx.algorithms.approximation.traveling_salesman_problem(G, method=christofides)[:-1]
        print(f"Time taken: {time.time() - tc:.2f}s")
        p0_weight = nx.path_weight(G, p0 + [0], weight='weight')
        print(f"Christofides path_weight = {p0_weight}")
        #plot_path(grid_points, p0)
    else:
        p0 = None

    # Solve local (2-opt)
    results_2o = multiprocess_local_search(
        dm=dm,
        x0=[p0] * NPROC,
        max_processing_time=TIMEOUT_2OPT,
        nproc=NPROC,
        perturbation_scheme="two_opt",
    )
    x02o, total_weights_2o = zip(*results_2o.values())
    print("Total weights after two_opt local search:")
    print(pd.Series(total_weights_2o))

    # Refine with Lin-Kernighan
    x0lk, total_weights_lk = multiprocess_lk(
        dm=dm, x0=x02o, max_processing_time=TIMEOUT_LK, nproc=NPROC)
    print("Total weights after Lin-Kernighan:")
    print(pd.Series(total_weights_lk))
