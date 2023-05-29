# custom
from utils import utils, vis
# from utils import poly_point_isect as bo   ##bentley-ottmann sweep line
import criteria as C
import quality as Q
# import gd2
from gd2 import GD2
import utils.weight_schedule as ws

# third party
import networkx as nx
# from PIL import Image
from natsort import natsorted

# numeric
import numpy as np
# import scipy.io as io
import torch
from torch import nn, optim
import torch.nn.functional as F

# vis
import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits import mplot3d
from matplotlib import collections  as mc
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# sys
from collections import defaultdict
import random
import time
from glob import glob
import math
import os
from pathlib import Path
import itertools
import pickle as pkl

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

# Set the criteria and the weights
CRITERIA_WEIGHTS = dict(
    stress=ws.SmoothSteps([MAX_ITER / 4, MAX_ITER], [1, 0.05]),
    ideal_edge_length=ws.SmoothSteps([0, MAX_ITER * 0.2, MAX_ITER * 0.6, MAX_ITER], [0, 0, 0.2, 0]),
    aspect_ratio=ws.SmoothSteps([0, MAX_ITER * 0.2, MAX_ITER * 0.6, MAX_ITER], [0, 0, 0.5, 0]),
)
CRITERIA = list(CRITERIA_WEIGHTS.keys())

# Set the sample sizes
SAMPLE_SIZES = {
    'stress': 16,
    'ideal_edge_length': 16,
    'neighborhood_preservation': 16,
    'crossings': 128,
    'crossing_angle_maximization': 64,
    'aspect_ratio': max(128, int(len(G) ** 0.5)),
    'angular_resolution': 16,
    'vertex_resolution': max(256, int(len(G) ** 0.5)),
    'gabriel': 64,
}
SAMPLE_SIZES = {c: SAMPLE_SIZES[c] for c in CRITERIA}


def run_optimization(G):
    gd = GD2(G)
    result = gd.optimize(
        criteria_weights=CRITERIA_WEIGHTS,
        sample_sizes=SAMPLE_SIZES,
        evaluate=set(CRITERIA),
        max_iter=MAX_ITER,
        time_limit=3600,
        evaluate_interval=MAX_ITER, evaluate_interval_unit='iter',
        vis_interval=-1, vis_interval_unit='sec',
        clear_output=True,
        grad_clamp=20,
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

# Run the optimization and print the results
gd, result = run_optimization(G)
print_nodes_and_edges(gd)

# Visualize the result
visualize_and_save(gd, result)

save_layout_as_dot(gd, 'final_layout.dot')
