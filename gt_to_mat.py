import graph_tool.all as gt
import scipy.sparse as sp
import scipy.io as sio
import numpy as np

# Load the graphml file
gt_graph = gt.load_graph('../netviz/sample_graphs/price_10000nodes.graphml')

# Get the adjacency matrix as a scipy sparse matrix
adj_matrix = gt.adjacency(gt_graph)

# Convert the adjacency matrix to CSC format
csc_matrix = adj_matrix.tocsc()

# Define the metadata
metadata = {
    'description': 'SOCIAL NETWORK CONNECTION TABLE',
    'name': 'graph',
    'id': 121,  # Arbitrary ID
    'year': '2023',  # Current year
    'authors': 'Unknown',
    'editors': 'Unknown',
    'problem': 'social network problem'
}

# Create a dictionary in the format that the s_gd2 script expects
mat_file_dict = {
    '__header__': b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Sat Sep  6 04:06:08 2008',
    '__version__': '1.0',
    '__globals__': [],
    'Problem': {
        'A': csc_matrix,
        'name': metadata['name'],
        'id': metadata['id'],
        'date': metadata['year'],
        'author': metadata['authors'],
        'editor': metadata['editors'],
        'kind': metadata['problem']
    }
}

# Save the dictionary in a .mat file
sio.savemat('input_graphs/price_10000nodes.mat', mat_file_dict, do_compression=True)
