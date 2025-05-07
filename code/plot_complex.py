import networkx as nx
import itertools
import dionysus as d
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

# from Laplacian_Functions import compute_boundary_matrices

# Function from: https://github.com/iaciac/py-draw-simplicial-complex/blob/master/Draw%202d%20simplicial%20complex.ipynb
def draw_2d_simplicial_complex(simplices, pos=None, return_pos=False, ax = None):
    """
    Draw a simplicial complex up to dimension 2 from a list of simplices, as in [1].
        
        Args
        ----
        simplices: list of lists of integers
            List of simplices to draw. Sub-simplices are not needed (only maximal).
            For example, the 2-simplex [1,2,3] will automatically generate the three
            1-simplices [1,2],[2,3],[1,3] and the three 0-simplices [1],[2],[3].
            When a higher order simplex is entered only its sub-simplices
            up to D=2 will be drawn.
        
        pos: dict (default=None)
            If passed, this dictionary of positions d:(x,y) is used for placing the 0-simplices.
            The standard nx spring layour is used otherwise.
           
        ax: matplotlib.pyplot.axes (default=None)
        
        return_pos: dict (default=False)
            If True returns the dictionary of positions for the 0-simplices.
            
        References
        ----------    
        .. [1] I. Iacopini, G. Petri, A. Barrat & V. Latora (2019)
               "Simplicial Models of Social Contagion".
               Nature communications, 10(1), 2485.
    """

    
    #List of 0-simplices
    nodes =list(set(itertools.chain(*simplices)))
    
    #List of 1-simplices
    edges = list(set(itertools.chain(*[[tuple(sorted((i, j))) for i, j in itertools.combinations(simplex, 2)] for simplex in simplices])))

    #List of 2-simplices
    triangles = list(set(itertools.chain(*[[tuple(sorted((i, j, k))) for i, j, k in itertools.combinations(simplex, 3)] for simplex in simplices])))
    
    if ax is None: ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])      
    ax.set_ylim([-1.1, 1.1])
    ax.get_xaxis().set_ticks([])  
    ax.get_yaxis().set_ticks([])
    # ax.axis('off')
       
    if pos is None:
        # Creating a networkx Graph from the edgelist
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
            
        # Creating a dictionary for the position of the nodes
        pos = nx.spring_layout(G, seed=123, iterations=200, k=2/(len(G.nodes))**0.1)
        
    # Drawing the edges
    for i, j in edges:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        line = plt.Line2D([ x0, x1 ], [y0, y1 ],color = 'black', zorder = 1, lw=0.7)
        ax.add_line(line)
    
    # Filling in the triangles
    for i, j, k in triangles:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        (x2, y2) = pos[k]
        tri = plt.Polygon([ [ x0, y0 ], [ x1, y1 ], [ x2, y2 ] ],
                          edgecolor = 'black', facecolor = plt.cm.Blues(0.6),
                          zorder = 2, alpha=0.4, lw=0.5)
        ax.add_patch(tri)

    # Drawing the nodes 
    for i in nodes:
        (x, y) = pos[i]
        circ = plt.Circle([ x, y ], radius = 0.02, zorder = 3, lw=0.5,
                          edgecolor = 'Black', facecolor = u'#ff7f0e', label = str(i))
        ax.add_patch(circ)
        txt = ax.text(x+0.075, y+0.075, str(i), fontsize=12, ha='center', va='center', zorder = 4)

    if return_pos: return pos


def draw_filtration(f: d.Filtration):
    f.sort()
    # boundary_matrices, name_to_idx, simplices_at_time, relevant_times = compute_boundary_matrices(f, lambda x: 1)
    
    t_old = 0
    relevant_times = []
    for s in f:
        if s.data != t_old:
            relevant_times.append(t_old)
            t_old = s.data
    relevant_times.append(s.data)

    # Create a figure with subplots
    
    fig, axs = plt.subplots(len(relevant_times)//6+1, min(len(relevant_times),6), figsize=(5*min(len(relevant_times),6), 5*(len(relevant_times)//6+1)))
    # Iterate through the relevant times and plot each filtration step
    for i, time in enumerate(relevant_times):
        # Get the simplices at the current time
        cur_simplices = []
        for s in f:
            if s.data <= time:
                cur_simplices.append([s[j] for j in range(s.dimension()+1)])
        # Draw the simplicial complex
        if len(relevant_times) >= 6:
            cur_ax = axs[i//6, i%6]
        else:
            cur_ax = axs[i]
        draw_2d_simplicial_complex(cur_simplices, ax=cur_ax)

        cur_ax.set_title(f"Time: {time:.2f}")
    fig.tight_layout()

    plt.show()
