import sys
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from operator import itemgetter

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
    # Build a dictionary that contains all the neighbors
    connected_to = [[x[1] for x in edges if x[0] == y] for y in range(total_nodes)]
    connected_from = [[x[0] for x in edges if x[1] == y] for y in range(total_nodes)]
    total = [x + y for x, y in zip(connected_from, connected_to)]
    return {key: total[key] for key in range(total_nodes)}

def valid_assignment(node, color, neighbors, assignments):
    neighbor_colors = [assignments[key] for key in neighbors[node]]
    # Determine if the color for this node has
    # been chosen for the neighbor nodes
    return color not in neighbor_colors

def forward_check(assignments, node, domains, neighbors):
    # Copy the current domains
    next_domains = deepcopy(domains)
    # Iterate over the neighbord of the node
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

def degree_heuristic(neighbors, assignments):
    # Determine which variables are not assigned, get their total neighbors, and
    # create a tuple of (node, total neighbors)    
    total = [(node, len(neighbors[node])) for node in assignments.keys() if assignments[node] is None]
    # Return the node with the larger neighbor count
    return max(total,key=itemgetter(1))[0]

def print_dict(dict_stuff):
    for key in dict_stuff.keys():
        print("Node:", key, "Values:", dict_stuff[key])
    print("\n")

def backtracking(assignments, neighbors, domains, prev_domains):
    # Return when all the variables have been assigned
    # a solution has been found
    if not None in assignments.values():
        return assignments
    # Select a variable/node based using degree heuristic
    node = degree_heuristic(neighbors, assignments)
    # Iterate over each color that is available to the node
    for color in domains[node]:
        # Determine if the color for the node is a valid assignment 
        if valid_assignment(node, color, neighbors, assignments):
            # Assign the color to the node
            assignments[node] = color
            # Perform forward checking and obtain an updated
            # domain set
            next_domains = forward_check(assignments, node, domains, neighbors)
            # Recursive call to backtracking
            result = backtracking(assignments, neighbors, domains, next_domains)
            # Only return the result when not None
            if result != None:
                return result
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
    # Previous domains
    prev_domains = {key: [] for key in range(len(planar_map["nodes"]))}
    # Build the neighbors dictionary
    neighbors = build_neighbors_dict(len(assignments.keys()), planar_map["edges"])
    # Start the backtracking algorithm
    assignments = backtracking(assignments, neighbors, domains, prev_domains)
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
        draw_map(name, planar_map, (6,6), coloring)
    else:
        print(f"{name} cannot be colored with {len(colors)} colors.")

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    ## edit these to indicate what you implemented.
    print("Backtracking...", "?")
    print("Forward Checking...", "?")
    print("Minimum Remaining Values...", "?")
    print("Degree Heuristic...", "?")
    print("Least Constraining Values...", "?")
    print("")

    three_colors = ["red", "blue", "green"]
    four_colors = ["red", "blue", "green", "yellow"]

    # Easy Map
    assign_and_test_coloring("Connecticut", connecticut, four_colors, trace=debug)
    assign_and_test_coloring("Connecticut", connecticut, three_colors, trace=debug)
    # Difficult Map
    assign_and_test_coloring("Europe", europe, four_colors, trace=debug)
    assign_and_test_coloring("Europe", europe, three_colors, trace=debug)

    
