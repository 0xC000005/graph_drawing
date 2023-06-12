## custom
from utils import utils, vis
from utils import poly_point_isect as bo   ##bentley-ottmann sweep line
import criteria as C
import quality as Q
import gd2


## third party
import networkx as nx
import pandas as pd
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


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
plt.style.use('ggplot')
plt.style.use('seaborn-colorblind')

# G = nx.balanced_tree(2,5)
# graph_str = 'tree_2_5'
# dir_out = './analysis/criteria_pairs/tree_2_5-t1636413236'

# G = nx.grid_2d_graph(6,10)
# graph_str = 'grid_6_10'
# dir_out = './analysis/criteria_pairs/grid_6_10-t1636413141'

# load graphml using graph-tool
print('loading graph')
# G = nx.cycle_graph(10000)
df = pd.read_csv('../netviz/sample_graphs/dolphins-edges.csv')
df['~from'] = df['~from'].str.replace('n', '')
df['~to'] = df['~to'].str.replace('n', '')
G = nx.from_pandas_edgelist(df, source='~from', target='~to', create_using=nx.Graph())
print('the graph is loaded')


import importlib

importlib.reload(C)
importlib.reload(Q)
importlib.reload(utils)
importlib.reload(vis)
import gd2

importlib.reload(gd2)
from gd2 import GD2

def generate_weights(step_size=0.25):
    weights_list = []
    # iterate over possible values for each weight
    for weight1 in np.arange(0, 1 + step_size, step_size):
        for weight2 in np.arange(0, 1 - weight1 + step_size, step_size):
            weight3 = 1 - weight1 - weight2
            if weight3 >= 0:
                weights_list.append((weight1, weight2, weight3))
    return weights_list


WEIGHTS_LIST = generate_weights(step_size=0.25)

# WEIGHTS = WEIGHTS_LIST[3]
# GRAPH_NAME = 'dolphins'
# MAX_ITER = int(1e4)
#
# IDEAL_EDGE_LENGTH_WEIGHT = WEIGHTS[0]
# CROSSINGS_WEIGHT = WEIGHTS[1]
# CROSSING_ANGLE_MAXIMIZATION_WEIGHT = WEIGHTS[2]

IDEAL_EDGE_LENGTH_WEIGHT = 0.25
CROSSINGS_WEIGHT = 0.5
CROSSING_ANGLE_MAXIMIZATION_WEIGHT = 0.25

criteria_weights = dict(
    ideal_edge_length=IDEAL_EDGE_LENGTH_WEIGHT,
    crossings=CROSSINGS_WEIGHT,
    crossing_angle_maximization=CROSSING_ANGLE_MAXIMIZATION_WEIGHT,
)

sample_sizes = dict(
    ideal_edge_length=32,
    crossings=128,
    crossing_angle_maximization=16,
)

## choose criteria
criteria_all = [
    'ideal_edge_length',
    'crossings',
    'crossing_angle_maximization',
]

seed = 2337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

gd = GD2(G)
print('gd is initialized')
result = gd.optimize(
    criteria_weights=criteria_weights,
    sample_sizes=sample_sizes,
    evaluate=criteria_all,

    max_iter=MAX_ITER,
    evaluate_interval=1000,
    vis_interval=-1,
    criteria_kwargs=dict(
        aspect_ratio=dict(target=[1, 1]),
    ),
    optimizer_kwargs=dict(mode='SGD', lr=2),
    scheduler_kwargs=dict(verbose=False),
)
print(result['qualities'])

## output
pos = gd.pos.detach().numpy().tolist()
pos_G = {k:pos[gd.k2i[k]] for k in gd.G.nodes}

print('nodes')
for node_id, pos in pos_G.items():
    print(f'{node_id}, {pos[0]}, {pos[1]}')

print('edges')
for e in gd.G.edges:
    print(f'{e[0]}, {e[1]}')


# visulized the network from netwokrx
## vis
vis.plot(
    gd.G, pos_G,
    [gd.iters, gd.loss_curve],
    result['iter'], result['runtime'],
    criteria_weights, MAX_ITER,
    # show=True, save=False,
    node_size=1,
    edge_width=1,
)
# plt.show()

# save the plot as img.png
plt.savefig(f'{GRAPH_NAME}.png',
            dpi=300)

plt.close()