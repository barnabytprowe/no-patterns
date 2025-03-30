"""
(Approximate) Traveling Salesman Problem (TSP) solutions for two dimensional
grids
"""

import multiprocessing
import pickle
import time
from itertools import repeat

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.approximation import greedy_tsp, christofides
from python_tsp.heuristics import solve_tsp_lin_kernighan, solve_tsp_local_search

import fitting_polynomials_2d


# Number of processes to run simultaneously, and total runs to attempt
NPROC = 10
NRUNS = 30

INITIALIZE_WITH_CHRISTOFIDES = True
# Timeout settings (all in seconds)
# 2-opt initial optimization
TIMEOUT_2OPT = 300
# Lin-Kernighan
CYCLE_LK = 1  # checks for completion every cycle
TIMEOUT_LK = 30  # total timeout

# Grid dimensions
ngrid = 28

# Pickle file output
output_pickle = f"tsp_grid_2d_{ngrid}x{ngrid}_{NPROC}_{TIMEOUT_2OPT}.pkl"


# Pool starmap utilities

def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)


def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


# Graph & plot utilities

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


def plot_path(xy, path, figsize=(6, 5.5), title=None):
    """Plot a path through xy coordinates.

    Args:
        xy: array-like where each row contains the coordinates of a 2D point
        path: index of xy rows defining the path to plot

    Returns: matplotlib.Axes instance for the plot
    """
    x, y = zip(*[xy[_i] for _i in path])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.plot(x, y, "ko")
    ax.plot(x, y)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return ax


# Multi processing

def process_lin_kernighan(pk, results_dict, distance_matrix=None, x0=None):
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
    distance_matrix,
    nruns,
    x0=None,
    max_processing_time=TIMEOUT_2OPT,
    nproc=NPROC,
    perturbation_scheme="two_opt",
):
    """Launch and gather results from multiprocessing of approximate TSP
    solutions using local search heuristics

    Args:
        distance_matrix: passed to python_tsp.heuristics.solve_tsp_local_search
        nruns:
            number of runs to attempt in total, probably some multiple of nproc
        x0:
            if set, a length nruns iterable of previous TSP path approximations
            to use as starting points for local search, an element of which is
            passed to python_tsp.heuristics.solve_tsp_local_search for each run
        max_processing_time:
            passed to python_tsp.heuristics.solve_tsp_local_search
        nproc: int number of processes to run simultaneously via multiprocessing
        perturbation_scheme:
            passed to python_tsp.heuristics.solve_tsp_local_search

    Returns:
        dict of returns from python_tsp.heuristics.solve_tsp_local_search keyed by
        integer process label, one of range(nproc)
    """
    processes = {}
    manager = multiprocessing.Manager()
    results_storage = manager.dict()
    if x0 is None:
        x0 = [None] * nruns

    with multiprocessing.Pool(nproc) as pool:
        print(f"Launching {nruns} runs for {perturbation_scheme} TSP, {max_processing_time=}s")
        results = starmap_with_kwargs(
            pool,
            solve_tsp_local_search,
            repeat([]),
            [
                {
                    "distance_matrix": distance_matrix,
                    "x0": x0[_p],
                    "perturbation_scheme": perturbation_scheme,
                    "max_processing_time": max_processing_time,
                    "verbose": False,
                }
                for _p in range(nruns)
            ],
        )

    return {_p: _result for _p, _result in enumerate(results)}


def multiprocess_lk(dm, x0=None, max_processing_time=TIMEOUT_LK, nproc=NPROC):
    """Launch and gather results from multiprocessing of approximate TSP
    solutions using the Lin-Kernighan algorithm.

    Args:
        distance_matrix: passed to python_tsp.heuristics.process_lin_kernighan
        x0:
            if set, a length nproc iterable of previous TSP path approximations
            to use as starting points for local search, an element of which is
            passed to python_tsp.heuristics.process_lin_kernighan for each run
        max_processing_time:
            maximum time in second allowed to any Lin-Kernighan process before it
            is terminated (failed)
        nproc: int number of processes to run simultaneously via multiprocessing
        perturbation_scheme: passed to python_tsp.heuristics.process_lin_kernighan

    Returns:
        dict of returns from python_tsp.heuristics.process_lin_kernighan keyed by
        integer process label, one of range(nproc)
    """
    processes = {}
    manager = multiprocessing.Manager()
    results_storage = manager.dict()
    if x0 is None:
        x0 = [None] * nproc

    for ip in range(nproc):
        print(f"Launching process {ip} for Lin-Kernighan TSP")
        processes[ip] = multiprocessing.Process(
            target=process_lin_kernighan,
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

    return results_storage


if __name__ == "__main__":

    # Initialize complete (but unweighted edge) graph and distance matrix
    G = nx.complete_graph(ngrid**2)
    grid_points = [(i, j) for i in range(ngrid) for j in range(ngrid)]
    dm = distance_matrix(grid_points)

    # Add edges between all node pairs with weight = distance between points
    print(f"Adding edge distance weights for complete {ngrid}x{ngrid} grid graph")
    t0 = time.time()
    G = add_edges_from_distance_matrix(G, dm)
    print(f"Time taken: {time.time() - t0:.2f}s")

    # Initialize
    odict = {}
    if INITIALIZE_WITH_CHRISTOFIDES:
        print("Calculating Christofides approximation")
        tc = time.time()
        p0 = nx.algorithms.approximation.traveling_salesman_problem(G, method=christofides)[:-1]
        print(f"Time taken: {time.time() - tc:.2f}s")
        total_weights_p0 = nx.path_weight(G, p0 + [0], weight='weight')
        print(f"Christofides path_weight = {total_weights_p0}")
        odict["p0"] = p0
        odict["total_weights_p0"] = total_weights_p0
    else:
        p0 = None

    # Solve local (2-opt)
    results_2opt = multiprocess_local_search(
        distance_matrix=dm,
        nruns=NRUNS,
        x0=[p0] * NRUNS,
        max_processing_time=TIMEOUT_2OPT,
        nproc=NPROC,
        perturbation_scheme="two_opt",
    )
    p2opt, total_weights_2opt = zip(*results_2opt.values())
    odict["results_2opt"] = results_2opt
    odict["p2opt"] = p2opt
    odict["total_weights_2opt"] = total_weights_2opt
    print("Total weights after two_opt local search:")
    print(pd.Series(total_weights_2opt))

    import ipdb; ipdb.set_trace()

    # Refine with Lin-Kernighan
    results_lk = multiprocess_lk(
        dm=dm, x0=p2opt, max_processing_time=TIMEOUT_LK, nproc=NPROC)
    plk, total_weights_lk = zip(*results_lk.values())
    odict["results_lk"] = results_lk
    odict["plk"] = plk
    odict["total_weights_lk"] = total_weights_lk
    print("Total weights after Lin-Kernighan:")
    print(pd.Series(total_weights_lk))

    print(f"Saving results to {output_pickle}")
    with open(output_pickle, "wb") as fout:
        pickle.dump(odict, fout)

    if INITIALIZE_WITH_CHRISTOFIDES:
        ax = plot_path(grid_points, p0, title="Christofides approximation")
        plt.show()

    for i, _p2opt in enumerate(p2opt):
        ax = plot_path(grid_points, _p2opt, title=f"2-opt local search")
        plt.show()

    for i, _plk in enumerate(plk):
        ax = plot_path(grid_points, _plk, title=f"2-opt, Lin-Kernighan algorithm")
        plt.show()
