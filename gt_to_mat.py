import graph_tool.all as gt
import networkx as nx
import scipy.io
from scipy.sparse import coo_matrix

# Convert graph_tool graph to networkx graph
def convert_gt_to_nx(g):
    nx_graph = nx.Graph()
    for edge in g.edges():
        nx_graph.add_edge(*edge)
    return nx_graph

# Convert networkx graph to scipy sparse matrix
def convert_nx_to_scipy(nx_graph):
    adj_matrix = nx.to_scipy_sparse_matrix(nx_graph)
    return adj_matrix

# Save scipy sparse matrix to .mat file
def save_scipy_to_mat(adj_matrix, filename):
    scipy.io.savemat(filename, {'graph': adj_matrix})


# Load a csv edgelists into a networkx graph and save as .mat file
def load_csv_to_mat(filename):
    nx_graph = nx.read_edgelist(filename, delimiter=',')
    adj_matrix = nx.to_scipy_sparse_matrix(nx_graph)
    scipy.io.savemat(filename.replace('.csv', '.mat'), {'graph': adj_matrix})


if __name__ == '__main__':
    # Use the functions
    # gt_graph = gt.Graph()  # Replace with your graph_tool graph
    # load graphml from ../netviz/sample_graphs/price_10000nodes.graphml
    gt_graph = gt.load_graph('../netviz/sample_graphs/price_10000nodes.graphml')

    nx_graph = convert_gt_to_nx(gt_graph)
    scipy_matrix = convert_nx_to_scipy(nx_graph)
    save_scipy_to_mat(scipy_matrix, 'input_graphs/price_10000nodes.mat')
