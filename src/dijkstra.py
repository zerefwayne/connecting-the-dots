import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx
from collections import deque, defaultdict
import heapq
from matplotlib.table import Table
from copy import deepcopy

def dijkstra_capture_states(G, start_node):
    # Distance from start_node to all other nodes, initialized to infinity for all except the start_node
    distances = defaultdict(lambda: float('inf'))
    
    # Initialize distance to infinity for all nodes
    for node in G.nodes():
        distances[node] = float('inf')
    
    distances[start_node] = 0
    
    # Priority queue for selecting the next node with the smallest distance
    pq = [(0, start_node)]
    
    # Visited nodes to keep track of processed nodes
    visited = set()
    
    # Capture states: (current node, distances, priority queue)
    states = [([], dict(distances), list(pq), deepcopy(distances))]
    
    while pq:
        # Pop the node with the smallest distance
        current_distance, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue

        visited.add(current_node)
        
        # Capture state after selecting a node but before exploring its neighbors
        states.append(([(current_node, current_distance)], dict(distances), list(pq), deepcopy(distances)))
        
        for neighbor, data in G[current_node].items():
            weight = data['weight']
            distance_through_current = current_distance + weight
            
            # If a shorter path to neighbor is found
            if distance_through_current < distances[neighbor]:
                distances[neighbor] = distance_through_current
                heapq.heappush(pq, (distance_through_current, neighbor))
                
        # Capture state after exploring neighbors
        states.append(([(current_node, current_distance)], dict(distances), list(pq), deepcopy(distances)))
    
    # Final state after all nodes are processed
    states.append(([], dict(distances), list(pq), deepcopy(distances)))
    
    return states

# Initialize graph and positions
G = nx.Graph()

# Adjacency matrix for the graph
adjacency_matrix = [
    [0, 2, 3, 0, 0, 5],  # Example weights for A's connections
    [2, 0, 0, 2, 0, 4],  # Example weights for B's connections
    [3, 0, 0, 0, 3, 0],  # Example weights for C's connections
    [0, 2, 0, 0, 2, 1],  # Example weights for D's connections
    [0, 0, 3, 2, 0, 2],  # Example weights for E's connections
    [5, 1, 0, 1, 2, 0]   # Example weights for F's connections
]

# Nodes corresponding to the matrix rows and columns
nodes = ['A', 'B', 'C', 'D', 'E', 'F']

# Add edges to the graph based on the adjacency matrix
for i, row in enumerate(adjacency_matrix):
    for j, val in enumerate(row):
        if val > 0:
            G.add_edge(nodes[i], nodes[j], weight=val)

pos = nx.spring_layout(G)  # positions for all nodes

# Perform BFS and capture states
states = dijkstra_capture_states(G, 'A')

[ print(s) for s in states ]

# print(states)

fig = plt.figure(figsize=(12, 10))
graph_ax = fig.add_axes([0.05, 0.2, 0.5, 0.7])  # Placeholder for the graph
table_ax = fig.add_axes([0.71, 0.3, 0.15, 0.4])  # Placeholder for the table

# Adjust buttons' positions if needed
axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
axprev = plt.axes([0.1, 0.05, 0.1, 0.075])

# Create buttons
bnext = Button(axnext, 'Next')
bprev = Button(axprev, 'Previous')

# Button event handlers
def on_next(event):
    print("NEXT", current_state, len(states))
    if current_state[0] < len(states) - 1:
        current_state[0] += 1
        draw_state(current_state[0])
        print("STATE INDEX", current_state[0])
        fig.canvas.draw_idle()

def on_previous(event):
    print("PREV")
    if current_state[0] > 0:
        current_state[0] -= 1
        draw_state(current_state[0])
        print("STATE INDEX", current_state[0])
        fig.canvas.draw_idle()

bnext.on_clicked(on_next)
bprev.on_clicked(on_previous)

def draw_state(state_index):
    graph_ax.clear()
    table_ax.clear()

    # Your existing drawing code for nodes and edges, now using graph_ax instead of ax
    current_node, visited, queue, distances = states[state_index]

    # Display the current queue
    queue_text = f'Current Node: {current_node}, Queue: {queue}'
    graph_ax.set_title(queue_text, fontsize=16, loc='left', y=1.05)

    nx.draw_networkx(G, pos, ax=graph_ax, node_color='lightgrey', node_size=1000, with_labels=False)
    nx.draw_networkx_nodes(G, pos, nodelist=[x for x in visited.keys()], node_color='skyblue', node_size=1000, ax=graph_ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[x[1] for x in queue], node_color='lightgreen', node_size=1000, ax=graph_ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[x[0] for x in current_node], node_color='yellow', node_size=1000, ax=graph_ax)
    nx.draw_networkx_edges(G, pos, ax=graph_ax, width=2)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=16, ax=graph_ax)

    node_labels = {node: f"{node} {dist}" for node, dist in distances.items()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=graph_ax, font_size=10)

    # Creating the table on the table_ax
    table_ax.clear()  # Clear previous table/plot
    table = Table(table_ax, bbox=[0, 0, 1, 1])  # Use the entire right subplot for the table
    nrows, ncols = len(distances) + 1, 2
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add the first row as a header
    table.add_cell(-1, -1, width, height, text='Node', loc='center', facecolor='lightgrey')
    table.add_cell(-1, 0, width, height, text='Distance', loc='center', facecolor='lightgrey')
    
    for i, (node, dist) in enumerate(distances.items()):
        table.add_cell(i, -1, width, height, text=node, loc='center')
        table.add_cell(i, 0, width, height, text=str(dist), loc='center')

    table.set_fontsize(14)
    table.scale(1, 1.2)
    table_ax.add_table(table)
    table_ax.axis('off')  # Hide the axes around the table

# Initial drawing
current_state = [0]  # Use a list for mutable integer to track the current state
draw_state(current_state[0])

plt.show()