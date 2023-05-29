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

# Save the structured array in a .mat file
sio.savemat('input_graphs/price_10000nodes.mat', {'Problem': structured_array}, do_compression=True)
