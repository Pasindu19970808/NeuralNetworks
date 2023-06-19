from pydantic import validate_arguments
from utils.modules import Value
import os
import graphviz
from graphviz import Digraph
import numpy as np

os.environ["PATH"] += os.pathsep + r'C:/Program Files/Graphviz/bin/'


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

def trace(root):
    nodes,edges = set(),set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
        for i in v._prev:
            edges.add((v,i,v._op))
            build(i)
    build(root)
    return nodes,edges

@validate_arguments(config = dict(arbitrary_types_allowed = True))
def build_graph(root:Value):
    nodes,edges = trace(root)
    dot = graphviz.Digraph('computation_graph',graph_attr={'rankdir' : 'LR'})
    for node in nodes:
        uid = str(id(node))
        dot.node(uid,label = f"{node.label} | Value = {np.round(node.data,4)} | Grad = {np.round(node.grad,4)}",shape = 'record')
        if node._op != '':
            dot.node(uid + f"_{node._op}",label = f"{node._op}")
            dot.edge(uid + f"_{node._op}",uid)
    for edge in edges:
    # required_data = obtain_label(edge)
    # dot.edge(required_data['child_name'],required_data['parent_name'])
        dot.edge(str(id(edge[1])),str(id(edge[0])) + f"_{edge[2]}")
    return dot