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
G = nx.cycle_graph(10000)
# G = nx.read_graphml('../netviz/sample_graphs/price_10000nodes.graphml')
print('the graph is loaded')


import importlib

importlib.reload(C)
importlib.reload(Q)
importlib.reload(utils)
importlib.reload(vis)
import gd2

importlib.reload(gd2)
from gd2 import GD2

criteria_weights = dict(
    ideal_edge_length=0.05,
    crossings=0.2,
    crossing_angle_maximization=0.1,
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

    max_iter=1000,
    evaluate_interval=1000,
    vis_interval=-1,
    criteria_kwargs=dict(
        aspect_ratio=dict(target=[1, 1]),
    ),
    optimizer_kwargs=dict(mode='SGD', lr=2),
    scheduler_kwargs=dict(verbose=False),
)
print(result['qualities'])



# visulized the network from netwokrx
