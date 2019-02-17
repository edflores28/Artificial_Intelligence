import sys
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from operator import itemgetter
from random import choice

def draw_map(name, planar_map, size, color_assignments=None):
    def as_dictionary(a_list):
        dct = {}
        for i, e in enumerate(a_list):
            dct[i] = e
        return dct
    G = nx.Graph()
    labels = as_dictionary(planar_map[ "nodes"])
    pos = as_dictionary(planar_map["coordinates"])
    # create a List of Nodes as indices to match the "edges" entry.
    nodes = [n for n in range(0, len(planar_map[ "nodes"]))]
    if color_assignments:
        colors = [c for n, c in color_assignments]
    else:
        colors = ['red' for c in range(0,len(planar_map[ "nodes"]))]
    G.add_nodes_from( nodes)
    G.add_edges_from( planar_map[ "edges"])
    plt.figure( figsize=size, dpi=600)
    nx.draw( G, node_color = colors, with_labels = True, labels = labels, pos = pos)
    plt.savefig(name + ".png")

def build_neighbors_dict(total_nodes, edges):
    '''
    This routine creates a neighbor dictionary for all nodes.

    Args:
        total_nodes - The total nodes in the map
        edges       - All the connections on the map
    Returns:
        A dictionary of node to neighbors
    '''
    # Build a dictionary that contains all the neighbors
    connected_to = [[x[1] for x in edges if x[0] == y] for y in range(total_nodes)]
    connected_from = [[x[0] for x in edges if x[1] == y] for y in range(total_nodes)]
    total = [x + y for x, y in zip(connected_from, connected_to)]
    return {key: total[key] for key in range(total_nodes)}

def valid_assignment(color, neighbors, assignments, named_node, trace):
    '''
    This routine determines if the node color assignment is valid

    Args:
        node - The node to check
        color - The color of the node
        neighbors - The neighbors of the node
        assignments - The current color assignments
        named_node - The name of the node
        trace - A flag for printing
    Returns:
        True if the color is not selected for a neighbor
        False otherwise
    '''
    neighbor_colors = [assignments[key] for key in neighbors]
    if trace:
        print(named_node, "assignment is", color, ". Neighboring colors are:", *neighbor_colors)
    # Determine if the color for this node has
    # been chosen for the neighbor nodes
    return color not in neighbor_colors

def forward_check(assignments, node, domains, neighbors, named_nodes, trace):
    '''
    This routine performs forward checking

    Args:
        assignments - The current color assignments
        node - The node to forward check
        neighbors - The neighbors
        trace - A flag for printin
        names_nodes - List of all node names
    Returns:
        A dictionary of updated domains
    '''
    if trace:
        print("Performing forward check removing", assignments[node], "from:")
        print(*[named_nodes[x] for x in neighbors])
    # Copy the current domains
    next_domains = deepcopy(domains)
    # Iterate over the neighbors of the node
    for neighbor in neighbors[node]:
        # If neighbor has not be assigned remove the
        # color that was assigned to the node
        if assignments[neighbor] is None:
            try:
                next_domains[neighbor].remove(assignments[node])
            except:
                pass
    # Return the next domains
    return next_domains

def degree_heuristic(neighbors, assignments, trace):
    '''
    This routine selects variables based on the
    degree heuristic. The node with most neighbors
    is selected

    Args:
        neighbors - The list of neighbors
        assignments - The current assignments
        trace - A flag for printin
    Returns:
        The selected node/variable
    '''
    if trace:
        print("\nSelecting a variable based on degree heuristic")
    # Determine which variables are not assigned, get their total neighbors, and
    # create a tuple of (node, total neighbors)    
    total = [(node, len(neighbors[node])) for node in assignments.keys() if assignments[node] is None]
    # Find the node with the most neighbors
    max_tuple = max(total,key=itemgetter(1))
    # Create a list with other nodes of the same value
    maxes = [x for x in total if x[1] == max_tuple[1]]
    # Pick a node out of the list
    return choice(maxes)[0]

def least_constraining_value(node, domains, neighbors, named_node, trace):
    '''
    This routines performs the LCV heuristic.

    Args:
        node - The node
        domains - The domains for all the nodes
        neighbors - the list of neighboard
        named_node - The name of the node
        trace - A flag for printin
    Returns:
        A list of values from starting from the LCV
    '''
    if trace:
        print("Determining", named_node, "least constraining values")
    # Create a counts dict based on the current domain
    # for the node
    counts = {key: 0 for key in domains[node]}
    # Iterate over the colors
    for key in counts.keys():
        # Iterate over each neighbor
        for neighbor in neighbors:
            # If the neighbor has the color then
            # increment the list
            if key in domains[neighbor]:
                counts[key] += 1
    # Return a list of colors by incrementing counts
    return sorted(key for (key, value) in counts.items())

def backtracking(assignments, neighbors, domains, next_domains, named_nodes, trace):
    '''
    The backtracking algorithm

    Args:
        assignments - the assignments
        neighbors - the dictionary of nodes to neighbors
        domains - the dictionary of domains for the nodes
        next_domains - the next domains
        named_nodes - the names of all the nodes
        trace - A flag for printin
    Returns:
        Assignments - all nodes assignents
        None - No assignments
    '''
    # Return when all the variables have been assigned
    # a solution has been found
    if not None in assignments.values():
        return assignments
    # Select a variable/node based using degree heuristic
    node = degree_heuristic(neighbors, assignments, trace)
    named_node = named_nodes[node]
    if trace:
        print(named_node, "was selected")
    # Iterate over each color that is available to the node base on the
    # least contraining value heuristic
    for color in least_constraining_value(node, domains, neighbors[node], named_node, trace):
        if trace:
            print("Trying color", color, "for", named_node)
        # Determine if the color for the node is a valid assignment 
        if valid_assignment(color, neighbors[node], assignments, named_node, trace):
            # Assign the color to the node
            assignments[node] = color
            # Perform forward checking and obtain an updated
            # domain set
            next_domains = forward_check(assignments, node, domains, neighbors, named_nodes, trace)
            # Recursive call to backtracking
            result = backtracking(assignments, neighbors, domains, next_domains, named_nodes, trace)
            # Only return the result when not None
            if result != None:
                return result
        if trace:
            print("No valid assignments for", named_nodes[node], "removing assignment\n")
        # Set the assignment for the node to None. This is the
        # backtracking portion. If this is reached then there
        # are no valid assignments or the recursive call
        # returned
        assignments[node] = None
    return None

def color_map( planar_map, colors, trace=False):
    """
    This function takes the planar_map and tries to assign colors to it.

    planar_map: Dict with keys "nodes", "edges", and "coordinates". "nodes" is a List of node names, "edges"
    is a List of Tuples. Each tuple is a pair of indices into "nodes" that describes an edge between those
    nodes. "coorinates" are x,y coordinates for drawing.

    colors: a List of color names such as ["yellow", "blue", "green"] or ["orange", "red", "yellow", "green"]
    these should be color names recognized by Matplotlib.

    If a coloring cannot be found, the function returns None. Otherwise, it returns an ordered list of Tuples,
    (node name, color name), with the same order as "nodes".
    """
    # Create a dict with initial value of None for all nodes
    assignments = {key: None for key in range(len(planar_map["nodes"]))}
    # Create a dict with all the colors for all nodes
    domains = {key: deepcopy(colors) for key in range(len(planar_map["nodes"]))}
    # Build the neighbors dictionary
    neighbors = build_neighbors_dict(len(assignments.keys()), planar_map["edges"])
    # Start the backtracking algorithm
    assignments = backtracking(assignments, neighbors, domains, None, planar_map["nodes"], trace)
    # Make sure the correct data is returned
    if assignments is None:
        return None
    else:
        return [(planar_map["nodes"][x], assignments[x]) for x in assignments.keys()]


connecticut = {"nodes": ["Fairfield", "Litchfield", "New Haven", "Hartford", "Middlesex", "Tolland", "New London", "Windham"],
               "edges": [(0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4), (3,5), (3,6), (4,6), (5,6), (5,7), (6,7)],
               "coordinates": [( 46, 52), ( 65,142), (104, 77), (123,142), (147, 85), (162,140), (197, 94), (217,146)]}

europe = {
    "nodes":  ["Iceland", "Ireland", "United Kingdom", "Portugal", "Spain",
                 "France", "Belgium", "Netherlands", "Luxembourg", "Germany",
                 "Denmark", "Norway", "Sweden", "Finland", "Estonia",
                 "Latvia", "Lithuania", "Poland", "Czech Republic", "Austria",
                 "Liechtenstein", "Switzerland", "Italy", "Malta", "Greece",
                 "Albania", "Macedonia", "Kosovo", "Montenegro", "Bosnia Herzegovina",
                 "Serbia", "Croatia", "Slovenia", "Hungary", "Slovakia",
                 "Belarus", "Ukraine", "Moldova", "Romania", "Bulgaria",
                 "Cyprus", "Turkey", "Georgia", "Armenia", "Azerbaijan",
                 "Russia" ], 
    "edges": [(0,1), (0,2), (1,2), (2,5), (2,6), (2,7), (2,11), (3,4),
                 (4,5), (4,22), (5,6), (5,8), (5,9), (5,21), (5,22),(6,7),
                 (6,8), (6,9), (7,9), (8,9), (9,10), (9,12), (9,17), (9,18),
                 (9,19), (9,21), (10,11), (10,12), (10,17), (11,12), (11,13), (11,45), 
                 (12,13), (12,14), (12,15), (12,17), (13,14), (13,45), (14,15),
                 (14,45), (15,16), (15,35), (15,45), (16,17), (16,35), (17,18),
                 (17,34), (17,35), (17,36), (18,19), (18,34), (19,20), (19,21), 
                 (19,22), (19,32), (19,33), (19,34), (20,21), (21,22), (22,23),
                 (22,24), (22,25), (22,28), (22,29), (22,31), (22,32), (24,25),
                 (24,26), (24,39), (24,40), (24,41), (25,26), (25,27), (25,28),
                 (26,27), (26,30), (26,39), (27,28), (27,30), (28,29), (28,30),
                 (29,30), (29,31), (30,31), (30,33), (30,38), (30,39), (31,32),
                 (31,33), (32,33), (33,34), (33,36), (33,38), (34,36), (35,36),
                 (35,45), (36,37), (36,38), (36,45), (37,38), (38,39), (39,41),
                 (40,41), (41,42), (41,43), (41,44), (42,43), (42,44), (42,45),
                 (43,44), (44,45)],
    "coordinates": [( 18,147), ( 48, 83), ( 64, 90), ( 47, 28), ( 63, 34),
                   ( 78, 55), ( 82, 74), ( 84, 80), ( 82, 69), (100, 78),
                   ( 94, 97), (110,162), (116,144), (143,149), (140,111),
                   (137,102), (136, 95), (122, 78), (110, 67), (112, 60),
                   ( 98, 59), ( 93, 55), (102, 35), (108, 14), (130, 22),
                   (125, 32), (128, 37), (127, 40), (122, 42), (118, 47),
                   (127, 48), (116, 53), (111, 54), (122, 57), (124, 65),
                   (146, 87), (158, 65), (148, 57), (138, 54), (137, 41),
                   (160, 13), (168, 29), (189, 39), (194, 32), (202, 33),
                   (191,118)]}


COLOR = 1

def test_coloring(planar_map, coloring):
    edges = planar_map["edges"]
    nodes = planar_map[ "nodes"]

    for start, end in edges:
        try:
            assert coloring[ start][COLOR] != coloring[ end][COLOR]
        except AssertionError:
            print("%s and %s are adjacent but have the same color." % (nodes[ start], nodes[ end]))


def assign_and_test_coloring(name, planar_map, colors, trace=False):
    print(f"Trying to assign {len(colors)} colors to {name}")
    coloring = color_map(planar_map, colors, trace=trace)
    if coloring:
        print(f"{len(colors)} colors assigned to {name}.")
        test_coloring(planar_map, coloring)
        draw_map(name, planar_map, (8,8), coloring)
    else:
        print(f"{name} cannot be colored with {len(colors)} colors.")

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    ## edit these to indicate what you implemented.
    print("Backtracking...", "Y")
    print("Forward Checking...", "Y")
    print("Minimum Remaining Values...", "N")
    print("Degree Heuristic...", "Y")
    print("Least Constraining Values...", "Y")
    print("")

    three_colors = ["red", "blue", "green"]
    four_colors = ["red", "blue", "green", "yellow"]

    # Easy Map
    assign_and_test_coloring("Connecticut_4", connecticut, four_colors, trace=debug)
    assign_and_test_coloring("Connecticut_3", connecticut, three_colors, trace=debug)
    # Difficult Map
    assign_and_test_coloring("Europe_4", europe, four_colors, trace=debug)
    assign_and_test_coloring("Europe_3", europe, three_colors, trace=debug)

    
