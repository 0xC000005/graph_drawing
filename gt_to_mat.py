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
    'description': np.array(['SOCIAL NETWORK CONNECTION TABLE'], dtype='<U30'),
    'matrix': csc_matrix,
    'name': np.array(['graph'], dtype='<U10'),
    'id': np.array([[121]], dtype=np.uint8),  # Arbitrary ID
    'year': np.array(['2023'], dtype='<U4'),  # Current year
    'authors': np.array(['Unknown'], dtype='<U7'),
    'editors': np.array(['Unknown'], dtype='<U7'),
    'problem': np.array(['social network problem'], dtype='<U21')
}

# Create a structured numpy array to hold the metadata
structured_array = np.array([(metadata['description'], metadata['matrix'], metadata['name'],
                              metadata['id'], metadata['year'], metadata['authors'],
                              metadata['editors'], metadata['problem'])],
                            dtype=[('description', 'O'), ('matrix', 'O'), ('name', 'O'),
                                   ('id', 'O'), ('year', 'O'), ('authors', 'O'),
                                   ('editors', 'O'), ('problem', 'O')])

# Create a dictionary in the format that the s_gd2 script expects
mat_file_dict = {
    '__header__': b'MATLAB 5.0 MAT-file, Platform: GLNXA64, Created on: Sat Sep  6 04:06:08 2008',
    '__version__': '1.0',
    '__globals__': [],
    'Problem': {
        'A': structured_array,
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
