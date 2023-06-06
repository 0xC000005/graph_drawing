import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
import random

n = 9  # number of nodes
p = 0.5  # probability of edge creation

# Create a random graph
G = nx.erdos_renyi_graph(n, p)

# Assign random positions to each node
for i in range(n):
    G.nodes[i]['x'] = random.randint(1, 6)
    G.nodes[i]['y'] = random.randint(1, 6)

# Compute weights and distances
weights = nx.adjacency_matrix(G).todense()
distances = nx.floyd_warshall_numpy(G)

# Prepare the data for JSON
data = {
    "nodes": [{"index": i, "id": i, "x": G.nodes[i]['x'], "y": G.nodes[i]['y']} for i in G.nodes],
    "edges": [{"source": u, "target": v} for u, v in G.edges],
    "width": 6,
    "height": 6,
    "weights": weights.tolist(),
    "graphDistance": distances.tolist()
}

# Save as JSON
with open('graph.json', 'w') as f:
    json.dump(data, f)

# Draw the graph
nx.draw(G, with_labels=True)
plt.show()

if __name__ == '__main__':
    pass