import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx

def dfs_capture_states(G, start_node):
    visited = []
    stack = [start_node]  # Use stack for DFS
    states = [([], [], [])]  # Initial state: (current node, visited, stack)

    while stack:
        node = stack.pop()  # Pop from stack
        if node not in visited:
            visited.append(node)
            states.append(([node], list(visited), list(stack)))  # Capture state after visiting a node
            for neighbor in sorted(G.neighbors(node), reverse=True):  # Optional: reverse sorting for consistent order
                if neighbor not in visited and neighbor not in stack:
                    stack.append(neighbor)  # Push neighbors to stack
                    # Note: State capture is not here in DFS, since we capture after popping
            states.append(([node], list(visited), list(stack)))

    states.append(([], list(visited), list(stack)))  # Final state

    return states

# Initialize graph and positions
G = nx.Graph()

# Adjacency matrix for the graph
adjacency_matrix = [
    [0, 1, 1, 0, 0, 1],  # A's connections to B, C
    [1, 0, 0, 1, 0, 1],  # B's connections to A, D
    [1, 1, 0, 0, 1, 0],  # C's connections to A, E
    [0, 1, 0, 0, 1, 1],  # D's connections to B, E, F
    [0, 1, 1, 1, 0, 1],  # E's connections to C, D, F
    [0, 0, 0, 1, 1, 0]   # F's connections to D, E
]

# Nodes corresponding to the matrix rows and columns
nodes = ['A', 'B', 'C', 'D', 'E', 'F']

# Add edges to the graph based on the adjacency matrix
for i, row in enumerate(adjacency_matrix):
    for j, val in enumerate(row):
        if val == 1:
            G.add_edge(nodes[i], nodes[j])

pos = nx.spring_layout(G)  # positions for all nodes

# Perform BFS and capture states
states = dfs_capture_states(G, 'A')

print(states)

# Create a figure and define the layout
fig, ax = plt.subplots(figsize=(12, 10))
plt.subplots_adjust(bottom=0.25)

# Function to draw a specific state of the BFS
def draw_state(state_index):
    ax.clear()  # Clear the current axes
    current_node, visited, queue = states[state_index]
    nx.draw_networkx(G, pos, ax=ax, node_color='lightgrey', node_size=1000, with_labels=True)  # Draw all nodes and edges
    nx.draw_networkx_nodes(G, pos, nodelist=visited, node_color='skyblue', node_size=1000, ax=ax)  # Highlight visited nodes
    nx.draw_networkx_nodes(G, pos, nodelist=queue, node_color='lightgreen', node_size=1000, ax=ax)  # Highlight nodes in queue
    nx.draw_networkx_nodes(G, pos, nodelist=current_node, node_color='yellow', node_size=1000, ax=ax)  # Highlight nodes in queue

    nx.draw_networkx_edges(G, pos, ax=ax, width=2)

    # Display the current queue
    queue_text = f'Current Node: {", ".join(current_node)} \nQueue: {queue}\nVisited: {visited}'
    ax.set_title(queue_text, fontsize=16, loc='left', y=1.05)

# Initial drawing
current_state = [0]  # Use a list for mutable integer to track the current state
draw_state(current_state[0])

# Button event handlers
def on_next(event):
    if current_state[0] < len(states) - 1:
        current_state[0] += 1
        draw_state(current_state[0])
        fig.canvas.draw_idle()

def on_previous(event):
    if current_state[0] > 0:
        current_state[0] -= 1
        draw_state(current_state[0])
        fig.canvas.draw_idle()

# Place buttons for navigation
axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bprev = Button(axprev, 'Previous')
bnext.on_clicked(on_next)
bprev.on_clicked(on_previous)

plt.show()
