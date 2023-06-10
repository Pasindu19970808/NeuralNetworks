def topological_sort(o):
    indegrees = {}
    indegrees[str(id(o))] = 0
    nodes_with_no_incoming_edges = []
    outdegree_neighbors = {}
    topological_ordering = []

    def add_nodes(node):
        for child in node._prev:
            if str(id(child)) not in indegrees:
                indegrees[str(id(child))] = 0
            add_nodes(child)

    def assign_indegrees(node):
        indegrees[str(id(node))] += len(node._prev)
        for child in node._prev:
            assign_indegrees(child)

    def nodes_with_no_incoming(node):
        #children which are in the node._prev for multiple nodes will 
        #make it into this multiple times without the if statement
        if len(node._prev) == 0:
            if node not in nodes_with_no_incoming_edges:
                nodes_with_no_incoming_edges.append(node)
        else:
            for child in node._prev:
                nodes_with_no_incoming(child)

    def add_outdegree_neighbors(node):
        for child in node._prev:
            if str(id(child)) not in outdegree_neighbors:
                outdegree_neighbors[str(id(child))] = []
            outdegree_neighbors[str(id(child))].append(node)
            add_outdegree_neighbors(child)

    add_nodes(o)
    assign_indegrees(o)
    nodes_with_no_incoming(o)
    outdegree_neighbors[str(id(o))] = []
    add_outdegree_neighbors(o)
    while len(nodes_with_no_incoming_edges) > 0:
        node = nodes_with_no_incoming_edges.pop()
        topological_ordering.append(node)
        for neighbor in outdegree_neighbors[str(id(node))]:
            indegrees[str(id(neighbor))] -= 1
            if indegrees[str(id(neighbor))] == 0:
                if neighbor not in nodes_with_no_incoming_edges:
                    nodes_with_no_incoming_edges.append(neighbor)

    return indegrees,nodes_with_no_incoming_edges,outdegree_neighbors,topological_ordering