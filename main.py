from utils import utils, vis
# from utils import poly_point_isect as bo
import criteria as C
import quality as Q
# import gd2
from gd2 import GD2
import utils.weight_schedule as ws
import networkx as nx
# from PIL import Image
from natsort import natsorted
import numpy as np
import pandas as pd
# import scipy.io as io
import torch
from torch import nn, optim
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits import mplot3d
from matplotlib import collections as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pos_df_to_graph import pos_df_to_graph
from collections import defaultdict
import random
import time
from glob import glob
import math
import os
from pathlib import Path
import itertools
import pickle as pkl
import graph_tool.all as gt

# Set the style for the plots
plt.style.use('ggplot')
plt.style.use('seaborn-colorblind')

# Constants
DEVICE = 'cpu'
SEED = 2337
GRAPH_NAME = 'price_10000nodes'
MAX_ITER = int(1e4)
MAT_DIR = 'input_graphs/'

# Set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load the graph
G = utils.load_mat(f'{MAT_DIR}/{GRAPH_NAME}.mat')

CRITERIA_WEIGHTS = dict(
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

SAMPLE_SIZES = dict(
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
CRITERIA_ALL = [
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
    gd = GD2(G)
    result = gd.optimize(
        criteria_weights=CRITERIA_WEIGHTS,
        sample_sizes=SAMPLE_SIZES,
        evaluate=CRITERIA_ALL,
        max_iter=MAX_ITER,
        time_limit=3600,
        evaluate_interval=MAX_ITER//10, evaluate_interval_unit='iter',
        vis_interval=-1, vis_interval_unit='sec',
        clear_output=True,
        criteria_kwargs=dict(aspect_ratio=dict(target=[1, 1])),
        optimizer_kwargs=dict(mode='SGD', lr=2),
        scheduler_kwargs=dict(verbose=True),
    )
    return gd, result


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
        CRITERIA_WEIGHTS, MAX_ITER,
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

gt_graph = gt.load_graph('../netviz/sample_graphs/price_10000nodes.graphml')
pos_df_to_graph(graphml=gt_graph, pos_df=pos_df, name='price_10000nodes_layout_by_gt')
