#from typing import List, Dict
import networkx as nx
import random
import math
import itertools
import ast

#######################################

def k_ary_fat_tree(k, directed=True):
    G = nx.Graph()

    core_switches = [('switch-c%d'%i, {'type':'switch', 'layer':'core'}) for i in range(1, (k**2)//4+1)] 
    G.add_nodes_from(core_switches)
    
    idx = 1
    hosts = []
    for pod in range(k):
        # adding agg switchs of a pod
        aggSwitches = [('switch-p%d-a%d'%(pod,i), {'type':'switch', 'layer':'aggregation'}) for i in range(1,1+k//2)]
        G.add_nodes_from(aggSwitches)
        # adding edge switchs of a pod
        edgeSwitches = [('switch-p%d-e%d'%(pod,i), {'type':'switch', 'layer':'edge'}) for i in range(1,1+ math.ceil(k/2))]
        G.add_nodes_from(edgeSwitches)
        # connection within the pod
        for s1, s2 in itertools.product(aggSwitches, edgeSwitches):
            G.add_edge(s1[0], s2[0])
        # insert hosts
        hosts = []
        for sw in (edgeSwitches):
            hosts = [('host-%d-p%d'%(idx+i, pod), {'type':'host', 'layer':'end'}) for i in range(k//2)]
            G.add_nodes_from(hosts)
            for host in hosts:
                G.add_edge(host[0], sw[0])
            idx+=k//2                
    # Connect core switches to aggregation switches
    i=0
    for sw, data in G.nodes(data=True):
        if data['layer']=='aggregation': 
            for _ in range(math.ceil(k/2)):
                core_switch = core_switches[i % ((k**2)//4)]
                G.add_edge(sw, core_switch[0])
                i+=1
    if directed:
        return G.to_directed()
    else:
        return G
    

def makeLineGraph(config, directed=True):
    """
    Creates a line topology with `numSwitchs` switches, and attaches `numHosts` hosts to the switches.

    Returns:
    - NetworkX graph object: A directed or undirected graph representing the line topology.
    """
    numHosts = config.getint('topology_config', 'numhosts')
    numSwitchs = config.getint('topology_config', 'numswitchs')
    
    # Create a line graph where switches are connected in a straight line
    graph = nx.path_graph(numSwitchs)
    
    # Rename nodes to indicate that they are switches
    mapping = {i: f'switch-c{i}' for i in range(numSwitchs)}
    nx.relabel_nodes(graph, mapping, copy=False)

    # Generate hosts and connect them to switches
    switches = list(graph.nodes())  # List of switches in the graph

    graph.add_edge(f'host-{0}', switches[0])
    graph.add_edge(f'host-{numHosts-1}', switches[-1])
    
    selected_switches = sorted(random.choices(range(numSwitchs), k=numHosts-2))
    for host in range(1, numHosts-1):
        hostName = f'host-{host}'
        graph.add_node(hostName)
        # Randomly select a switch to connect the host to
        switch = f'switch-c{selected_switches.pop(0)}'
        graph.add_edge(hostName, switch)
    
    if directed:
        return graph.to_directed()
    else:
        return graph

def makeRandomGraph(config, directed=True):
    
    numHosts = config.getint('topology_config', 'numhosts')
    numSwitchs = config.getint('topology_config', 'numswitchs')
    
    graphConfig = ast.literal_eval(config.get('topology_config', 'graphconfig'))

    typ = graphConfig['typ'].replace("'", '')

    # validate input
    if numSwitchs==1:
        typ ='er'
    if graphConfig['minnodedegreepersw'] >= numSwitchs:
        print('not enough switches to fulfill the condition on minnodedegreepersw, numSwitchs is increased to', graphConfig['minnodedegreepersw'], ' + 1 !!')
        numSwitchs = graphConfig['minnodedegreepersw'] + 1
        
    # generate graph
    try:
        if typ == 'w':
            #print('generating waxman graph...')
            graph = nx.waxman_graph(numSwitchs)
        if typ == 'ba':
            #print('generating barabasi_albert graph...')
            graph = nx.barabasi_albert_graph(numSwitchs, graphConfig['numEdgN'])
        if typ == 'er':
            #print('generating erdos_renyi graph...')
            graph = nx.erdos_renyi_graph(numSwitchs, graphConfig['prob'])

        # check and correct connection
        if graph and numSwitchs>1: # graph with more than one switch
            for node in graph.nodes():
                if graph.degree(node) < graphConfig['minnodedegreepersw']:
                    changeDegree(node,graph,graphConfig['minnodedegreepersw'])
        graph = graph.to_directed()
        ## rename switches
        mapping = {sw:'switch-'+str(sw) for sw in graph.nodes()}
        nx.relabel_nodes(graph, mapping, copy=False)
        
        ## generate hosts
        switches = list(graph.nodes())[:-graphConfig['numcoreswitches']]
        for host in range(numHosts):
            hostName = 'host-'+str(host)
            graph.add_node(hostName)
            # link it to a switch
            switch = random.choice(switches)
            graph.add_edge(hostName, switch)
            graph.add_edge(switch, hostName) # to make it directed
        if directed:
            return graph.to_directed()
        else:
            return graph
    except ValueError:
        print('Graph generation failed!!')
        return ValueError

def changeDegree (node, graph, newDegree):
    degree = graph.degree(node)
    neighbors = [n for n in graph.neighbors(node)]
    if newDegree > degree:
        newNeighbors = random.choices([i for i in graph.nodes() if i not in neighbors+[node]], k=newDegree-degree)
        for n in newNeighbors:
            graph.add_edge(node, n)
#        return graph
    if newDegree < degree:
        oldneighbors = random.choices(neighbors, k=degree-newDegree)
        for n in oldneighbors:
            graph.remove_edge(node, n)
    