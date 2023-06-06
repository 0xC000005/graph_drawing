## custom
from utils import utils, vis
from utils import poly_point_isect as bo   ##bentley-ottmann sweep line
import criteria as C
import quality as Q
import gd2


## third party
import networkx as nx

from PIL import Image
from natsort import natsorted


## sys
import random
import time
from glob import glob
import math
from collections import defaultdict
import os
from pathlib import Path
import itertools
import pickle as pkl

## numeric
import numpy as np
import scipy.io as io
import torch
from torch import nn, optim
import torch.nn.functional as F


## vis
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits import mplot3d
from matplotlib import collections  as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.gridspec import GridSpec

## notebook
# from IPython import display
# from IPython.display import clear_output
# from tqdm.notebook import tqdm

import graph_tool.all as gt
import pandas as pd

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = 'cpu'
plt.style.use('ggplot')
plt.style.use('seaborn-colorblind')


# GRAPH_NAME = 'price_10000nodes'
# MAX_ITER = int(1e4)
# MAT_DIR = 'input_graphs/'

GRAPH_NAME = 'dwt_307'
MAX_ITER = int(1e4)
MAT_DIR = 'input_graphs/SuiteSparse Matrix Collection'


# Load the graph
G = utils.load_mat(f'{MAT_DIR}/{GRAPH_NAME}.mat')

import importlib

importlib.reload(C)
importlib.reload(Q)
importlib.reload(utils)
importlib.reload(vis)
import gd2

importlib.reload(gd2)
from gd2 import GD2

criteria_weights = dict(
    stress=1,
    ideal_edge_length=0.05,
    neighborhood_preservation=0.5,
    crossings=0.2,
    crossing_angle_maximization=0.1,
    aspect_ratio=3,
    angular_resolution=1,
    vertex_resolution=1,
    gabriel=0.1,
)

sample_sizes = dict(
    stress=32,
    ideal_edge_length=32,
    neighborhood_preservation=16,
    crossings=128,
    crossing_angle_maximization=16,
    aspect_ratio=max(128, int(len(G) ** 0.5)),
    angular_resolution=128,
    vertex_resolution=max(256, int(len(G) ** 0.5)),
    gabriel=64,
)

## choose criteria
criteria_all = [
    'stress',
    'ideal_edge_length',
    'neighborhood_preservation',
    'crossings',
    'crossing_angle_maximization',
    'aspect_ratio',
    'angular_resolution',
    'vertex_resolution',
    'gabriel',
]




def run_optimization(G):
    criteria_pairs = [
        (ci, cj) for (i, ci), (j, cj)
        in list(
            itertools.product(
                enumerate(criteria_all),
                enumerate(criteria_all)
            ))
        #     if i<=j
        if i <= j and 'angular_resolution' in (ci, cj)
    ]

    for ci, cj in criteria_pairs:
        criteria_pair = {ci, cj}
        print(criteria_pair)
        print(criteria_pair)
        print(criteria_pair)
        print(criteria_pair)

        seed = 2337
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        gd = GD2(G)

        result = gd.optimize(
            criteria_weights={c: criteria_weights[c] for c in criteria_pair},
            sample_sizes=sample_sizes,
            evaluate=criteria_all,

            max_iter=1000,
            evaluate_interval=1000 // 40,
            vis_interval=-1,
            #             clear_output=True,
            criteria_kwargs=dict(
                aspect_ratio=dict(target=[1, 1]),
            ),
            #         optimizer_kwargs = dict(mode='Adam', lr=0.005),
            optimizer_kwargs=dict(mode='SGD', lr=2),
            scheduler_kwargs=dict(verbose=False),
        )
        print(result['qualities'])


def print_nodes_and_edges(gd):
    pos = gd.pos.detach().numpy().tolist()
    pos_G = {k: pos[gd.k2i[k]] for k in gd.G.nodes}

    print('nodes')
    for node_id, pos in pos_G.items():
        print(f'{node_id}, {pos[0]}, {pos[1]}')

    print('edges')
    for e in gd.G.edges:
        print(f'{e[0]}, {e[1]}')


def visualize_and_save(gd, result):
    pos = gd.pos.detach().numpy().tolist()
    pos_G = {k: pos[gd.k2i[k]] for k in gd.G.nodes}

    vis.plot(
        gd.G, pos_G,
        [gd.iters, gd.loss_curve],
        result['iter'], result['runtime'],
        criteria_weights, MAX_ITER,
        node_size=1,
        edge_width=1,
    )

    plt.savefig(f'{GRAPH_NAME}.png', dpi=300)
    plt.close()


def save_layout_as_dot(gd, filename):
    pos = gd.pos.detach().numpy().tolist()
    pos_G = {k: pos[gd.k2i[k]] for k in gd.G.nodes}

    # Add positions as node attributes
    for node_id, position in pos_G.items():
        gd.G.nodes[node_id]['pos'] = position

    # Save as .dot file
    nx.drawing.nx_pydot.write_dot(gd.G, filename)


# turn gd into a position dataframe, format: index, vertex, x, y
def gd_to_df(gd):
    pos = gd.pos.detach().numpy().tolist()
    pos_G = {k: pos[gd.k2i[k]] for k in gd.G.nodes}

    df = pd.DataFrame.from_dict(pos_G, orient='index')
    df = df.reset_index()
    df.columns = ['vertex', 'x', 'y']
    df['index'] = df.index
    df = df[['index', 'vertex', 'x', 'y']]
    return df


# Run the optimization and print the results
gd, result = run_optimization(G)
print_nodes_and_edges(gd)

# Visualize the result
visualize_and_save(gd, result)

save_layout_as_dot(gd, 'final_layout.dot')

pos_df = gd_to_df(gd)
# print the first 5 rows of the dataframe
print(pos_df.head())

# gt_graph = gt.load_graph('../netviz/sample_graphs/price_10000nodes.graphml')
# pos_df_to_graph(graphml=gt_graph, pos_df=pos_df, name='price_10000nodes_layout_by_gt')
