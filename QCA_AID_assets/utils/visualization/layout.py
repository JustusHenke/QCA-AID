"""
Layout algorithms for network visualizations
"""

import numpy as np
import networkx as nx
from sklearn.manifold import MDS


def create_forceatlas_like_layout(G, iterations=100, gravity=0.01, scaling=10.0):
    """
    Create a ForceAtlas2-like layout using NetworkX and scikit-learn.
    
    Args:
        G: NetworkX Graph
        iterations: Number of iterations
        gravity: Strength of attraction to center
        scaling: Scaling factor for node distances
        
    Returns:
        Dictionary with node positions
    """
    print("Berechne ForceAtlas-ähnliches Layout...")
    
    # Handle empty graph
    if len(G.nodes()) == 0:
        print("Warnung: Graph ist leer, gebe leeres Layout zurück")
        return {}
    
    # Handle single node
    if len(G.nodes()) == 1:
        node = list(G.nodes())[0]
        return {node: (0, 0)}
    
    # Important: Create undirected copy of graph for distance calculation
    # This ensures distance matrix is symmetric
    G_undirected = G.to_undirected()
    
    # Special handling for hierarchical structures
    # Different weighting depending on node type
    edges = list(G_undirected.edges())
    for src, tgt in edges:
        src_type = G.nodes[src]['node_type']
        tgt_type = G.nodes[tgt]['node_type']
        
        # Same types closer together, different types further apart
        if src_type == tgt_type:
            if src_type == 'main':
                weight = 0.5  # Main categories somewhat closer
            elif src_type == 'sub':
                weight = 1.0  # Subcategories normal
            else:
                weight = 2.0  # Keywords further apart
        else:
            # Connections between different types
            if (src_type == 'main' and tgt_type == 'sub') or (src_type == 'sub' and tgt_type == 'main'):
                weight = 1.0  # Main-to-sub: normal
            else:
                weight = 2.0  # Other connections: further apart
        
        # Set weight directly in undirected graph
        G_undirected[src][tgt]['weight'] = weight
    
    # Use NetworkX's integrated method for shortest paths
    # This is more robust than csgraph for directed graphs
    try:
        # Calculate all shortest paths with undirected graph
        distances = dict(nx.all_pairs_shortest_path_length(G_undirected))
        
        # Convert to matrix
        nodes = list(G.nodes())
        n = len(nodes)
        distance_matrix = np.zeros((n, n))
        
        # Create lookup table for node indices
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Fill distance matrix - guaranteed symmetric
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i == j:
                    # Diagonal with zeros
                    distance_matrix[i, j] = 0
                else:
                    try:
                        # Use precomputed shortest paths
                        distance_matrix[i, j] = distances[node1][node2]
                    except KeyError:
                        # For unconnected nodes: maximum distance
                        distance_matrix[i, j] = n * 2
        
        # MDS requires symmetric matrix
        # This should already be guaranteed, but we check for safety
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Use MDS (Multidimensional Scaling) as replacement for ForceAtlas2
        mds = MDS(
            n_components=2,            # 2D layout
            dissimilarity='precomputed',  # We provide distance matrix
            random_state=42,           # For reproducibility
            n_init=1,                  # One initialization
            max_iter=iterations,       # Iterations
            normalized_stress=False,   # Disable normalized stress for metric MDS
            n_jobs=1                   # Single-threaded for stability
        )
                
        # Apply MDS to distance matrix
        pos_array = mds.fit_transform(distance_matrix)
        
        # Add scaling
        pos_array *= scaling
        
        # Weak attraction to center (gravitation)
        # This is a simplified version of gravitation in ForceAtlas2
        if gravity > 0:
            center = np.mean(pos_array, axis=0)
            for i in range(pos_array.shape[0]):
                direction = center - pos_array[i]
                distance = np.linalg.norm(direction)
                if distance > 0:  # Avoid division by zero
                    # The further away, the stronger the attraction
                    force = gravity * distance
                    pos_array[i] += direction / distance * force
        
        # Convert back to dictionary
        pos = {nodes[i]: (pos_array[i, 0], pos_array[i, 1]) for i in range(n)}
        
        # Optional further adjustments:
        # 1. Group nodes by type
        node_by_type = {'main': [], 'sub': [], 'keyword': []}
        for node in G.nodes():
            node_type = G.nodes[node]['node_type']
            node_by_type[node_type].append(node)
        
        # 2. Slight attraction within same types
        for node_type, nodes_of_type in node_by_type.items():
            if len(nodes_of_type) > 1:
                # Find centroid of this group
                centroid = np.mean([pos[node] for node in nodes_of_type], axis=0)
                
                # Attraction coefficient depending on type
                attraction = 0.1 if node_type == 'main' else 0.05 if node_type == 'sub' else 0.01
                
                # Move all nodes slightly towards their centroid
                for node in nodes_of_type:
                    curr_pos = np.array(pos[node])
                    direction = centroid - curr_pos
                    # Mix current position with slight attraction to centroid
                    new_pos = curr_pos + direction * attraction
                    pos[node] = (new_pos[0], new_pos[1])
        
        print("ForceAtlas-ähnliches Layout berechnet.")
        return pos
        
    except Exception as e:
        print(f"Fehler bei ForceAtlas-Layout: {str(e)}")
        print("Falle zurück auf Spring-Layout als Alternative...")
        
        # Handle empty graph in fallback
        if len(G.nodes()) == 0:
            print("Warnung: Graph ist leer im Fallback")
            return {}
        
        # Handle single node in fallback
        if len(G.nodes()) == 1:
            node = list(G.nodes())[0]
            return {node: (0, 0)}
        
        # Create initial positions for better convergence
        initial_pos = {}
        
        # Position main categories in center
        main_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'main']
        num_main = len(main_nodes)
        for i, node in enumerate(main_nodes):
            if num_main > 1:
                angle = 2 * np.pi * i / num_main
                initial_pos[node] = (0.2 * np.cos(angle), 0.2 * np.sin(angle))
            else:
                initial_pos[node] = (0, 0)
        
        # Position subcategories in middle ring
        sub_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'sub']
        num_sub = len(sub_nodes)
        for i, node in enumerate(sub_nodes):
            if num_sub > 1:
                angle = 2 * np.pi * i / num_sub
                initial_pos[node] = (0.5 * np.cos(angle), 0.5 * np.sin(angle))
            else:
                initial_pos[node] = (0.5, 0)
        
        # Position keywords in outer ring
        keyword_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'keyword']
        num_kw = len(keyword_nodes)
        for i, node in enumerate(keyword_nodes):
            if num_kw > 1:
                angle = 2 * np.pi * i / num_kw
                initial_pos[node] = (1.0 * np.cos(angle), 1.0 * np.sin(angle))
            else:
                initial_pos[node] = (1.0, 0)
        
        # Use spring layout as fallback with weighted edges
        # for better hierarchical representation
        pos = nx.spring_layout(
            G,
            pos=initial_pos,
            k=1.5/np.sqrt(len(G.nodes())),
            iterations=100,
            weight='weight',
            scale=scaling
        )
        
        return pos
