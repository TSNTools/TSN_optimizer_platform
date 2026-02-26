##
# please don't remove this note
# this code is written by Hamza Chahed (hamza0chahed@gmail.com) in the context of AIDA project
# you can use it freely, enjoy!
##

import networkx as nx
from networkx.readwrite import json_graph
import random
import os
import json
import ast
import itertools
import configparser
import math
import numpy as np
import matplotlib.pyplot as plt

class  UsecaseGenerator():
    """ 
    randomly generate usecases using the provided parameters
    Output: all timed values in [ns], speed and bw in [mbps], datasize in [bytes]
    """
    index = 0
    __version__ = 1
    

    def __init__(self):
        pass
       
    def generateUC(self, configFile, *args, **kwargs):
        
        usecase = self.empty_uc(configFile, *args, **kwargs)
        # generate streams
        list(self.streamGenerator(usecase))
        usecase.index=self.index
        self.index+=1
        return usecase
    
    def ucGenerator(self, configFile, *args, **kwargs):
        '''incrementally generating use cases by adding one more stream at the time'''
        usecase = self.empty_uc(configFile, *args, **kwargs)
        streamGen = self.streamGenerator(usecase)
        for _ in streamGen:
            usecase.index = self.index
            self.index+=1
            yield usecase
##################################################################################
    def empty_uc(self, configFile, index=None, topology=None, newTopo=False):
         # initialise usecase
        usecase = Usecase(configFile, topology=topology)
        if index is not None:
            usecase.index = index
        
        # make usecase topology
        if (usecase.graph is None) or (newTopo):
            #usecase.config = self.get_config(fixPath, traffic_config=False)
            usecase.graph = self.makeTopology(usecase)
        # configure use case
        # usecase.taskembeddingparams['stride']['step'] = self.graphtypes['ft']['k']+1
        return usecase
    
##########################################################################################
#####                                Streams generation                              #####
##########################################################################################
    def streamGenerator(self, usecase):
        '''
        generate random streams as specified by the usecase configuration
        '''
        routing = ast.literal_eval(usecase.config.get('streams_config', 'routing'))
        fixPath = routing['fixPath']
        #fixPath = usecase.config.getboolean('streams_config', 'fixPath')
        numStreams = usecase.config.getint('streams_config', 'numofstreamsperinstance')
        
        for streamId in range(numStreams):
            stream = self.generateStream(streamId, usecase)#, talker, listeners)
            numplacesTalker = usecase.config.getint('streams_config', 'numplacesTalker')
            if fixPath and numplacesTalker==1:
                if routing['shortestPath']:
                    stream = self.shortestPath(stream, usecase)
                else: # random
                    stream = self.randomPath (stream, usecase) 
                
            usecase.streams.append(stream)
            yield stream
    
########################################################################
    def generateStream (self, streamId, usecase, talker=None, listeners=None):
        '''
        stream = streamId, interval(T)[ns], maxFramesPerInterval(Nf), maxFrameSize(l)[byte], MaxLatency(e2e)[ns]
        '''
         # read streams data
        period_range = range(*ast.literal_eval(str(usecase.config.get('streams_config', 'period_range'))).values())
        numberFrames_range = range(*ast.literal_eval(str(usecase.config.get('streams_config', 'numberframes_range'))).values())
        maxLatency_range = [float(i) for i in usecase.config.get("streams_config","maxlatency_range").split(',')]
        dataSize_range = range(*ast.literal_eval(str(usecase.config.get('streams_config', 'datasize_range'))).values())       
        #self.validate_input()
        
        maxLatency = int(random.choice(maxLatency_range)*1000000) # to make it in nanosecond
        numFramesPerCycle = random.choice(numberFrames_range)
        maxFrameSize = random.choice(dataSize_range)
        cycle_ns = int(random.choice(period_range)*1000) # to make it in nanosecond

        expiration = ast.literal_eval(usecase.config.get('streams_config', 'expiration'))
        if bool(expiration['finite_duration']):
            durations = range(int(expiration['start'])*60, int(expiration['stop'])*60) # in s
            random_duration_ns = random.choice(durations) * 1_000_000_000
            lifespanCycles = random_duration_ns // cycle_ns
        
        if talker is None:
            talker = self.talkerEmbedding(usecase)
            
        if listeners is None:
            listeners = self.listenerEmbedding(usecase, talker= talker[0]).tolist()

        
        stream =  {
            'name': 'stream-'+str(streamId),
            'type': 'isochronous',
            'interval':cycle_ns, #[ns]
            'maxFramesPerInterval':numFramesPerCycle,
            'maxFrameSize':maxFrameSize, #[bytes]
            'maxLatency':maxLatency, #[ns]
            'tlkrPssblPlcmnt': talker,
            'lstnsrPssblPlcmnt':listeners,
            'earliest-transmit-offset': 0,
            'latest-transmit-offset': cycle_ns//2,
            'jitter': 0
            }

        # Add lifespan if finite_duration
        if bool(expiration['finite_duration']):
            stream['lifespanCycles'] = int(lifespanCycles)
    
        return stream

    
########################################################################
    def talkerEmbedding(self, usecase):
        '''
        configure the probability of choosing a host as talker based on the needed options, and 
        choose the actual placements based on the configured probabilities
        '''    
        
        stride = ast.literal_eval(usecase.config.get('streams_config', 'stride'))
        staggered = ast.literal_eval(usecase.config.get('streams_config', 'staggered'))
        hotspot = ast.literal_eval(usecase.config.get('streams_config', 'hotspot'))
        
        numpossplace = usecase.config.getint('streams_config', 'numplacesTalker')
    
        allhosts = [node for node, data in usecase.graph.nodes(data=True) if data['type']=='host']
        allProbs = np.full(len(allhosts) , 1/len(allhosts))
        
        # choose talker
        if hotspot['active']:
            prevchosen = set(int(plc.split('-')[1])-1 for strm in usecase.streams for plc in strm['tlkrPssblPlcmnt'])
            allProbs[list(prevchosen)] *= hotspot['increase_factor']
            
        probabilities = allProbs/np.sum(allProbs) # to normalize
        
        '''does striding between chosen_placement and placements to be selected'''
        if stride['active'] and numpossplace>1:
            i = stride['step']
            
            chosen_placement = np.random.choice(allhosts, size=1, p=probabilities, replace=False)[0] # choose 1st placement of the talker
            x = int(chosen_placement.split('-')[1])-1
            for _ in range(numpossplace):
                x = (i+x)%len(allhosts)
                allProbs[x] *= stride['increase_factor']
            
            x = int(chosen_placement.split('-')[1])-1   
            tmp_hosts = allhosts[:x]+allhosts[x+1:]
            tmp_probabilities = allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]] / np.sum(allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]]) # to normalize
            return np.append(np.random.choice(tmp_hosts, size=numpossplace-1, p=tmp_probabilities, replace=False),chosen_placement).tolist()
        
            '''does staggering between chosen_placement and elements to be selected'''
        elif staggered['active'] and numpossplace>1: 
            edgep = staggered['edgep']
            podp = staggered['podp']
            
            chosen_placement = np.random.choice(allhosts, size=1, p=probabilities, replace=False)[0] # choose 1st placement of the talker
            pod = chosen_placement.split('-')[2]

            neighbors_in_pod = [node for node in allhosts if pod in node]        
            for host in neighbors_in_pod:
                x = int(host.split('-')[1])-1
                allProbs[x] = podp

            neighbors_in_edge = [node for node in usecase.graph.neighbors(list(usecase.graph.neighbors(chosen_placement))[0]) if (node in allhosts) and (node!=chosen_placement)]
            for host in neighbors_in_edge:
                x = int(host.split('-')[1])-1
                allProbs[x] = edgep

            x = int(chosen_placement.split('-')[1])-1
            if hotspot['active']:# in case hotspoting is combined with staggering
                allProbs[list(prevchosen)] *= hotspot['increase_factor']
               
            tmp_hosts = allhosts[:x]+allhosts[x+1:]
            tmp_probabilities = allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]] / np.sum(allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]]) # to normalize
            return np.append(np.random.choice(tmp_hosts, size=numpossplace-1, p=tmp_probabilities, replace=False),chosen_placement).tolist()
        
            '''choose randomly, uniformly'''
        else: 
            return np.random.choice(allhosts, size=numpossplace, p=probabilities, replace=False).tolist()
        
########################################################################
    def listenerEmbedding(self, usecase, talker, listeners = None):
        '''
        configure the probability of choosing a host based on the needed options, and 
        choose the actual placements based on the configured probabilities
        
        * the function is used to select listeners, the talker should be:
            - provided, 
            - filtered from hosts, and
            - removed from the returned list of elements if needed (don't forget to increase numpossplace to count for the given talker)
        ''' 
        if listeners is None:
            listeners = []
        
        stride = ast.literal_eval(usecase.config.get('streams_config', 'stride'))
        staggered = ast.literal_eval(usecase.config.get('streams_config', 'staggered'))
        hotspot = ast.literal_eval(usecase.config.get('streams_config', 'hotspot'))
        
        numpossplace = usecase.config.getint('streams_config', 'numplacesListener')
        numlistenersperstr = usecase.config.getint('streams_config', 'numlistenersperstr')
        
        allhosts = [node for node, data in usecase.graph.nodes(data=True) if data['type']=='host']
        allProbs = np.full(len(allhosts) , 1/len(allhosts))

        if hotspot['active']:
            prevchosen = [plc for strm in usecase.streams for plcs in strm['lstnsrPssblPlcmnt'] for plc in plcs]+listeners
            prevchosen = set(int(plc.split('-')[1])-1 for plc in prevchosen if plc in allhosts)
            allProbs[list(prevchosen)] *= hotspot['increase_factor']
            
            
        # if talker is None:
        #     probabilities = allProbs/np.sum(allProbs) # to normalize
        #     talker = np.random.choice(allhosts, size=1, p=probabilities, replace=False)[0]
            
            '''does striding between talker and elements to be selected'''
        if stride['active']:
            i = stride['step']
            
            x = int(talker.split('-')[1])-1
            for _ in range(numpossplace*numlistenersperstr):
                x = (i+x)%len(allhosts)
                allProbs[x] *= stride['increase_factor']
            x = int(talker.split('-')[1])-1   

            tmp_hosts = allhosts[:x]+allhosts[x+1:]
            tmp_probabilities = allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]] / np.sum(allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]]) # to normalize
            return np.random.choice(tmp_hosts, size=numpossplace*numlistenersperstr, p=tmp_probabilities, replace=False).reshape(numlistenersperstr, numpossplace)
        
            '''does staggering between talker and elements to be selected'''
        elif staggered['active']: 
            edgep = staggered['edgep']
            podp = staggered['podp']
            pod = talker.split('-')[2]

            neighbors_in_pod = [node for node in allhosts if pod in node]        
            for host in neighbors_in_pod:
                x = int(host.split('-')[1])-1
                allProbs[x] = podp

            neighbors_in_edge = [node for node in usecase.graph.neighbors(list(usecase.graph.neighbors(talker))[0]) if (node in allhosts) and (node!=talker)]
            for host in neighbors_in_edge:
                x = int(host.split('-')[1])-1
                allProbs[x] = edgep

            x = int(talker.split('-')[1])-1
            if hotspot['active']:# in case hotspoting is combined with staggering
                allProbs[list(prevchosen)] *= hotspot['increase_factor']
               
            tmp_hosts = allhosts[:x]+allhosts[x+1:]
            tmp_probabilities = allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]] / np.sum(allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]]) # to normalize
            return np.random.choice(tmp_hosts, size=numpossplace*numlistenersperstr, p=tmp_probabilities, replace=False).reshape(numlistenersperstr, numpossplace)
        
            '''choose randomly, uniformly'''
        else: 
            x = int(talker.split('-')[1])-1
            tmp_hosts = allhosts[:x]+allhosts[x+1:]
            tmp_probabilities = allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]] / np.sum(allProbs[[int(plc.split('-')[1])-1 for plc in  tmp_hosts]]) # to normalize
            return np.random.choice(tmp_hosts, size=numpossplace*numlistenersperstr, p=tmp_probabilities, replace=False).reshape(numlistenersperstr, numpossplace)
        
##########################################################################################
#####                                      Routing                                   #####
##########################################################################################
    def findAllPaths(self, usecase, talkers=None, listeners=None):
        allPaths = {} # dict of paths generators indexed by src-dest pair
        eligibleTalkers = [node for node in usecase.graph.nodes if node.startswith('host')]
        eligibleListeners = [node for node in usecase.graph.nodes if node.startswith('host')]
        # check list of talkers
        if talkers!=None:
            eligibleTalkers = [node for node in talkers if node in eligibleTalkers]
        # check list of listeners
        if listeners!=None:
            flat_listeners = [item for sublist in listeners for item in sublist]
            eligibleListeners = [node for node in flat_listeners if node in eligibleListeners]
         
        #for nd in itertools.combinations([node for node in self.graph.nodes() if node.startswith('host')], 2):
        for nd in itertools.product(*[eligibleTalkers,eligibleListeners]):
            pathsGen = nx.all_simple_paths(usecase.graph, nd[0], nd[1])#+ list(nx.all_simple_paths(self.graph, nd[1], nd[0]))       
            allPaths[nd] = pathsGen
        return allPaths 
        
#############################################################################    
    def shortestPath(self, stream, usecase):
        paths = []
        period_range = range(*ast.literal_eval(str(usecase.config.get('streams_config', 'period_range'))).values())
        shortestPaths = [nx.shortest_path(usecase.graph,pair[0],pair[1]) for pair in itertools.product(*[stream['tlkrPssblPlcmnt'],[item for sublist in stream['lstnsrPssblPlcmnt'] for item in sublist]])]
        
        for path in shortestPaths:
            for hop in range(len(path)-1):
                if usecase.graph.is_directed():
                    path[hop] = usecase.graph.get_edge_data(path[hop], path[hop+1])['sourcePort']
                else:
                    path[hop] = path[hop]+'.'+usecase.graph._adj[path[hop]][path[hop+1]][path[hop]]
                
            if self.pathConvenient(path, stream['maxLatency'], stream['maxFramesPerInterval'], stream['maxFrameSize'], usecase):
                paths.append(path)

            ##check convenience of period for the given (numFrames*dataSize), that the talker have enough time to transmit in every cycle
            cycle = stream['interval']
            while not self.periodConvenient(path, cycle,stream['maxFramesPerInterval'],stream['maxFrameSize'], usecase):
                cycle = max(stream['interval'], int(random.choice(period_range)*1000)) # to make it in nanosecond
            stream['interval'] = cycle
            stream['latest-transmit-offset'] = cycle//2
        stream['pathLists'] = paths
        return stream
        
#############################################################################
    def randomPath (self, stream, usecase):
        chosenListeners=[]
        paths = []

        flat_listeners = [item for sublist in stream['lstnsrPssblPlcmnt'] for item in sublist]
        allPathsList = [list(nx.all_simple_paths(usecase.graph, pair[0], pair[1])) 
                       for pair in itertools.product(*[stream['tlkrPssblPlcmnt'], flat_listeners])
                      ]

        # Shuffle the paths for each (talker, listener) pair to get randomness
        for paths_per_pair in allPathsList:
            random.shuffle(paths_per_pair)

        period_range = range(*ast.literal_eval(usecase.config.get('streams_config', 'period_range')).values())

        while allPathsList:
            idx = random.randrange(len(allPathsList))
            candidate_paths = allPathsList[idx]
            if not candidate_paths:
                allPathsList.pop(idx)
                continue
                
            # Pick and remove a random path from the shuffled sublist
            path = random.choice(candidate_paths)
            candidate_paths.remove(path)
            # path = candidate_paths.pop()
            if path[-1] in chosenListeners:
                continue  # Listener already assigned to a path; try another
              
            for hop in range(len(path)-1):
                if usecase.graph.is_directed():
                    path[hop] = usecase.graph.get_edge_data(path[hop], path[hop+1])['sourcePort']
                else:
                    path[hop] = path[hop]+'.'+usecase.graph._adj[path[hop]][path[hop+1]][path[hop]]
                
            if self.pathConvenient(path, stream['maxLatency'], stream['maxFramesPerInterval'], stream['maxFrameSize'], usecase):
                paths.append(path)
                chosenListeners.append(path[-1])

            if len(chosenListeners) >= len(stream['lstnsrPssblPlcmnt']):
                break

            ##check convenience of period for the given (numFrames*dataSize), that the talker have enough time to transmit in every cycle
            cycle = stream['interval']
            while not self.periodConvenient(path, cycle,stream['maxFramesPerInterval'],stream['maxFrameSize'], usecase):
                cycle = max(stream['interval'], int(random.choice(period_range)*1000)) # to make it in nanosecond
            stream['interval'] = cycle
            stream['latest-transmit-offset'] = cycle//2
            
        stream['pathLists'] = paths
        stream['lstnsrPssblPlcmnt'] = [chosenListeners]

        return stream
    
########################################################################
    def pathConvenient (self, pathList, maxLatency, maxFramesPerInterval, maxFrameSize, usecase):
        dataSize = maxFramesPerInterval*maxFrameSize*8 #size in bits
        minLatency = 0 # minimum possible latency
        for idx, hop in enumerate(pathList[:-1]):
            node = hop.split('.')[0]
            port = hop.split('.')[1]
            nextNode = pathList[idx+1].split('.')[0]
            #port = possibleEgress(node, G)[nextNode]
            transDelay = dataSize/usecase.graph.nodes[node]['ports'][port]['speed'] # need to consider the linkBw !!!!!
            try:
                minLatency += usecase.graph.get_edge_data(node,nextNode)['propDelay']+ transDelay + usecase.graph.nodes[node]['procDelay']
            except:
                pass
        return maxLatency > minLatency

########################################################################
    def periodConvenient (self, pathList, cycle, maxFramesPerInterval, maxFrameSize, usecase):
        accepted = True
        dataSize = maxFramesPerInterval*maxFrameSize*8 #size in bits
        i = 0
        while accepted:
            nextNode = pathList[i+1].split('.')[0]
            node = pathList[i].split('.')[0]
            port = pathList[i].split('.')[1]
            transDelay = dataSize/(usecase.graph.nodes[node]['ports'][port]['speed']/1000) #in [ns]  # need to consider the linkBw !!!!!
            if transDelay > cycle: 
                accepted = False
            elif i < len(pathList)-2:
                i+=1  
            else:
                break
        return accepted
    
##########################################################################################
#####                                  Generate topology                             #####
##########################################################################################
    # topology generation functions
    def makeTopology(self, usecase):
         # read default topology config
        portSpeed  = usecase.config.getint('topology_config', 'portspeed') # mbps
        propDelay = usecase.config.getint('topology_config', 'propdelay')
        procDelay = usecase.config.getint('topology_config', 'procdelay') # us
        linkBw = usecase.config.getint('topology_config', 'linkBw') # mbps
        queueCountPerPort = usecase.config.getint('topology_config', 'queuecountperport')

        graphConfig = ast.literal_eval(usecase.config.get('topology_config', 'graphconfig'))
        graphType = graphConfig['typ'].replace("'", '')
        
        ## generate graph
        if graphType == 'ft':
            #print('generating fat-tree graph...')
            graph = self.k_ary_fat_tree(graphConfig['k'])
        else:
            graph = self.makeRandomGraph(usecase)

        # add port information to graph
        for node, successors in graph._adj.items():
            portName = 1 
            graph.nodes[node]['ports'] = {}
            graph.nodes[node]['type'] = 'host'
            for succ in successors:
                if node not in graph._adj[succ]: # this is needed for directed graphs
                    graph._adj[succ][node]={}
                if succ not in graph._adj[node]: # this is needed for directed graphs
                    graph._adj[node][succ]={}
                    
                graph._adj[succ][node]['sourceNode']=succ
                graph._adj[succ][node]['destNode']=node
                graph._adj[succ][node]['destPort']=node+'.Port'+str(portName)
                graph._adj[node][succ]['sourcePort']=node+'.Port'+str(portName)
                graph._adj[succ][node]['propDelay']= propDelay
                graph._adj[succ][node]['linkBw']= linkBw
                graph.nodes[node]['ports']['Port'+str(portName)]= {'name':'Port'+str(portName) ,'speed':portSpeed, 'queueCount':queueCountPerPort}
                if portName > 1:
                    graph.nodes[node]['type'] = 'switch'
                    graph.nodes[node]['procDelay'] = procDelay*1000 # in [ns]

                portName+=1
        return graph
########################################################################    
    def k_ary_fat_tree(self, k, directed=True):
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

###############################################################################
    def makeRandomGraph(self, usecase, directed=True):
        
        numHosts = usecase.config.getint('topology_config', 'numhosts')
        numSwitchs = usecase.config.getint('topology_config', 'numswitchs')
        minnodedegreepersw = usecase.config.getint('topology_config', 'minnodedegreepersw') # minnumconnportpersw
        graphConfig = ast.literal_eval(usecase.config.get('topology_config', 'graphconfig'))

        typ = graphConfig['typ'].replace("'", '')

        # validate input
        if numSwitchs==1:
            typ ='er'
        if graphConfig['minnodedegreepersw'] >= numSwitchs:
            print('not enough switches to fulfill the condition on minnodedegreepersw, numSwitchs is increased to', minnodedegreepersw, ' + 1 !!')
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
                        self.changeDegree(node,graph,graphConfig['minnodedegreepersw'])
            graph = graph.to_directed()
            ## rename switches
            mapping = {sw:'switch-'+str(sw) for sw in graph.nodes()}
            nx.relabel_nodes(graph, mapping, copy=False)
            
            ## generate hosts
            switches = list(graph.nodes())[:-graphConfig['numCoreSwitches']]
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
    
    def changeDegree (self, node, graph, newDegree):
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
        
##########################################################################################
##########################################################################################
##########################################################################################
#####                                  Class use case                                #####
##########################################################################################
##########################################################################################
##########################################################################################

class  Usecase():
    def __init__(self, configFile, index= None, topology=None, streams=None):
        '''
        the charateristics of topology and streams are specified in the config file
        portSpeed [Mbps], linkbw [Mbps], propDelay [ns], procDelay [us], period[μs], 
        datasize[byte], maxLatency[ms]
        --> usecaseGenerator converts all timed values to ns

        TODO:            
            * we can fix talkers and listeners and remove paths, but we can not add paths or make talker/listeners floating: embedding*() needs to be improved:
            if numplacesTalker>len(tlkrPssblPlcmnt):
                tlkrPssblPlcmnt+random.sample([host for host in hosts if host not in tlkrPssblPlcmnt], numplacesTalker-len(tlkrPssblPlcmnt))  
            else: 
                tlkrPssblPlcmnt
        '''
        self.config = configparser.RawConfigParser(inline_comment_prefixes="#", allow_no_value=True)
        if configFile is not None:
            self.config.read(configFile)
        self.index = index
        self.graph = topology.to_directed() if topology is not None else topology
        self.streams = [] if streams is None else streams
    
    def removeRouting(self):
        routing = ast.literal_eval(self.config.get('streams_config', 'routing'))
        routing['fixPath'] = False
        self.config.set('streams_config', 'routing', str(routing))
        for stream in self.streams:
            try:
                del stream['pathLists']
            except:
                pass

    def addRouting(self, overwrite_config=False):
        numplacesTalker = self.config.getint('streams_config', 'numplacesTalker')
        if numplacesTalker!=1:
            return
            
        routing = ast.literal_eval(self.config.get('streams_config', 'routing'))
        if routing.get('fixPath', False) or overwrite_config:
            routing['fixPath'] = True
            self.config.set('streams_config', 'routing', str(routing))
            ucgen = UsecaseGenerator()
            for stream in self.streams:
                if routing.get('shortestPath', False):
                    stream = ucgen.shortestPath(stream, self)
                else: # random
                    stream = ucgen.randomPath(stream, self)
                

    def fixTalkers(self):
        self.config.set('streams_config','numplacestalker',1)
        for stream in self.streams:
            stream['tlkrPssblPlcmnt'] = [random.choice(stream['tlkrPssblPlcmnt'])]

    def fixListeners(self):
        self.config.set('streams_config','numplaceslistener',1)
        for stream in self.streams:
            stream['lstnsrPssblPlcmnt'] = [[random.choice(listener)] for listener in stream['lstnsrPssblPlcmnt']]
    
    def to_file(self, jsnPath=None, ucFileName=None):
        
        useCaseData = self.to_json()
        
        if jsnPath is None:
            jsnPath = './' 
        if ucFileName is None:
            ucFileName = self.get_config()+'.json'
        # export useCase
        with open(jsnPath+ucFileName, 'w') as outfile:
            json.dump(useCaseData, outfile, indent=4)
        
        print('A usecase is saved under the name', ucFileName,'under the folder', jsnPath)
        return ucFileName
               
    def to_json(self):
        usecase = self.topology2json()
        usecase['streams'] = []
        for stream in self.streams:
            usecase['streams'].append({key: value for key, value in stream.items()})
        usecase['index'] = self.index
        return usecase

    @property
    def hosts(self):
        return [node for node in self.graph.nodes if 'host' in node]

    @property
    def distance_to_hosts(self):
        """
        Returns: dict -> {host: {node: distance}}
        """
        all_distances = {}
    
        for host in self.hosts:
            if host not in self.graph:
                continue
            dist = nx.single_source_dijkstra_path_length(self.graph, host)
            all_distances[host] = dist
    
        return all_distances

    @property
    def switches(self):
        return [node for node in self.graph.nodes if 'switch' in node]
    
    @property
    def links(self):
        return [d for _,_,d in self.graph.edges(data=True)]
    
    @property
    def ports(self):
        ports = {}
        for _, _, d in self.graph.edges(data=True):
            ports[d['sourcePort']]={'sourceNode':d['sourceNode'],'destPort':d['destPort'], 'destNode':d['destNode']}
        for node, data in self.graph.nodes(data=True):
            for pname, pdata in data['ports'].items():   
                ports[node+'.'+pname].update(name=node+'.'+pname, speed=pdata['speed'], queueCount=pdata['queueCount'],nodeName=node)
        return ports
    
    def topology2json(self, source="sourcePort", target="destPort", name="name", key="key", linkSym="links",nodeSym="nodes"):
        '''
        this function adds ports attributes to the graph and export it to json
        !!! it changes the graph permanently !!!
        '''
        if len({source, target, key}) < 3:
            raise nx.NetworkXError("Attribute names are not unique.")
        data = {nodeSym: [
                    dict([('type',self.graph.nodes[n]['type']),(name, n),('ports',self.graph.nodes[n]['ports'])])
                    for n in self.graph if self.graph.nodes[n]['type']=='host'] + [
                    dict([('type',self.graph.nodes[n]['type']),(name, n),('procDelay', self.graph.nodes[n]['procDelay']),('ports',self.graph.nodes[n]['ports'])])
                    for n in self.graph if self.graph.nodes[n]['type']=='switch'],

                linkSym: [d for _,_,d in self.graph.edges(data=True)]
                #[dict([(source, d[source]), (target, d[target]), ('sourceNode', u), ('destNode', v), ('propDelay', d['propDelay']), ('linkBw', d['linkBw'])]) for u, v, d in self.graph.edges(data=True)]
        }
        return data
    
    def drawTopology(self): 
        
        plt.figure(1, figsize=(20, 8), dpi=60)

        positions = {}
        layers = {'core':3, 'aggregation':2, 'edge':1, 'end':0}
        layer_spacing = 1.0  # Adjust the spacing between layers
        xx = {'core':115, 'aggregation':75, 'edge':75, 'end':0}

        graphConfig = ast.literal_eval(self.config.get('topology_config', 'graphconfig'))
        graphType = graphConfig['typ'].replace("'", '')
        if graphType == 'ft':
            for node, data in self.graph.nodes(data=True):
                y = layers[data['layer']]
                x = xx[data['layer']]
                xx[data['layer']]+=20
                positions[node]=(x,y)
            pos = nx.multipartite_layout(self.graph, subset_key='layer', align='horizontal')
            # Draw the graph with adjusted spacing
            nx.draw(self.graph, pos=positions, with_labels=True, font_size=10)
        elif graphType == 'imported':
            try:
                for node in self.graph.nodes():
                    if '-c' in node:
                        layer = 'core'
                    elif '-a' in node:
                        layer = 'aggregation'
                    elif '-e' in node:
                        layer = 'edge'
                    else:
                        layer = 'end'
                    self.graph.nodes[node]['layer']=layer
                    y = layers[layer]
                    x = xx[layer]
                    xx[layer]+=20
                    positions[node]=(x,y)
                pos = nx.multipartite_layout(self.graph, subset_key='layer', align='horizontal')
                # Draw the graph with adjusted spacing
                nx.draw(self.graph, pos=positions, with_labels=True, font_size=10)
            except:
                nx.draw(self.graph, with_labels = True, font_size=10)
        else:
            nx.draw(self.graph, with_labels = True, font_size=10)
    
    @classmethod
    def from_json(cls, impUC):
        # Build graph
        G = nx.DiGraph()
        ## add nodes
        G.add_nodes_from([(node['name'], node) for node in impUC['nodes']])
        ## add edges
        G.add_edges_from([(link['sourceNode'], 
                           link['destNode'],
                           {'sourcePort':link['sourcePort'],
                            'destPort':link['destPort'],
                            'sourceNode':link['sourceNode'],
                            'destNode':link['destNode'],
                            'propDelay':link['propDelay'],
                            'linkBw':link['linkBw'],}) 
                          for link in impUC['links']])
    
        ucc = Usecase(configFile=None, index=impUC['index'], topology=G, streams=impUC['streams'])
        ucc.config = Usecase.guess_config(impUC)
    
        return ucc
        
    @classmethod
    def guess_config(cls, impUC):
        config = configparser.RawConfigParser(inline_comment_prefixes="#", allow_no_value=True)

        streams = impUC['streams']
        _streams ={}
        for d in streams:
            for key, value in d.items():
                try:
                    _streams[key].append(value)
                except:
                    _streams[key] = []
                    _streams[key].append(value)
        nodes = impUC['nodes']
        links = impUC['links']
        switches = [node for node in nodes if node['type']=='switch']
        randomNode = next(iter(nodes[0]['ports'].values()))
        
        config.add_section('topology_config')
        config.set('topology_config','numhosts',len(nodes)-len(switches))
        config.set('topology_config','numswitchs',len(switches))
        config.set('topology_config','portspeed',randomNode['speed'])
        config.set('topology_config','procdelay', switches[0]['procDelay']/1000)# make it in us
        config.set('topology_config','propdelay',links[0]['propDelay'])
        config.set('topology_config','linkBw', links[0]['linkBw'])
        config.set('topology_config','queuecountperport', randomNode['queueCount'])
        config.set('topology_config', 'graphconfig' ,"{'typ':'imported'}")

        config.add_section('streams_config')
        config.set('streams_config','numofstreamsperinstance', len(streams))
        #config.set('streams_config','maxstrpernode',)
        config.set('streams_config','period_range', {'start':min(_streams['interval'])//1000,'stop':max(_streams['interval'])//1000})# make it in us
        config.set('streams_config','numberframes_range', {'start':min(_streams['maxFramesPerInterval']),'stop':max(_streams['maxFramesPerInterval'])})
        config.set('streams_config','datasize_range', {'start':min(_streams['maxFrameSize']),'stop':min(_streams['maxFrameSize'])})
        config.set('streams_config','maxlatency_range', ','.join(set(str(val/1000000) for val in _streams['maxLatency']))) # make it in ms
        config.set('streams_config','numplacestalker',len(streams[0]['tlkrPssblPlcmnt']))
        config.set('streams_config','numplaceslistener',len(streams[0]['lstnsrPssblPlcmnt'][0]))
        config.set('streams_config','numlistenersperstr', max([len(l) for lol in _streams['lstnsrPssblPlcmnt'] for l in lol]))
        config.set('streams_config','routing',{'fixPath':'pathLists' in _streams})
        #config.set('streams_config','stride',)
        #config.set('streams_config','staggered',)
        #config.set('streams_config','hotspot',)
        return config

    def get_config(self):
        
        configstr = ''
        #topology config
        graphConfig = ast.literal_eval(self.config.get('topology_config', 'graphconfig'))
        if graphConfig["typ"] == 'ft': # typ
            configstr = 'ft'+'-'+str(graphConfig['k'])+'-'
        else: 
            for _, value in graphConfig.items():
                configstr+=str(value)+'-'
            configstr+=str(self.config.getint('topology_config', 'numhosts'))+'-'
            configstr+=str(self.config.getint('topology_config', 'numswitchs'))+'-'
        configstr += str(self.config.getint('topology_config', 'portspeed'))+'-'
        configstr += str(self.config.getint('topology_config', 'propdelay'))+'-'
        configstr += str(self.config.getint('topology_config', 'procdelay'))+'-'
        configstr += str(self.config.getint('topology_config', 'linkBw'))+'-'
        configstr += str(self.config.getint('topology_config', 'queuecountperport'))+'|'
        
        # traffic config
        configstr += str(self.config.getint('streams_config', 'numofstreamsperinstance')) + "-"
        configstr += str(self.config.getint('streams_config', 'numplacesTalker')) + "-" # numplacesTalker
        configstr += str(self.config.getint('streams_config', 'numplacesListener')) + "-" # numplacesListener 
        configstr += str(self.config.getint('streams_config', 'numlistenersperstr')) + "-" # numlistenersperstr 
        for param in ['routing','stride','staggered','hotspot']: # routing/taskConfig
            try:
                config = ast.literal_eval(str(self.config.get('streams_config', param)))
                for _, value in config.items():
                    configstr+=str(value)+'-'
            except:
                pass
        configstr += str(range(*ast.literal_eval(str(self.config.get('streams_config', 'period_range'))).values())) + "-" # period_range
        configstr += str(range(*ast.literal_eval(str(self.config.get('streams_config', 'numberframes_range'))).values())) + "-" # numberframes_range
        configstr += str(range(*ast.literal_eval(str(self.config.get('streams_config', 'dataSize_range'))).values())) + "-" # datasize_range
        configstr += str([float(i) for i in self.config.get("streams_config","maxlatency_range").split(',')]) + "|" # maxlatency_range
        # generation = ast.literal_eval(instance.config.get('streams_config', 'generation'))
        # configstr += str(generation['incremental'])+ "-"
        return configstr.rstrip("|")

    @staticmethod
    def generate_fattree_regions(k: int) -> list[list[str]]:
        """
        Generate non-overlapping, topology-aware regions for a k-ary Fat-Tree network.
    
        Regions:
          - Region 0: all core switches
          - Region i>0: aggregation + edge switches of pod i-1
    
        Naming conventions (consistent with k_ary_fat_tree):
          Core:          switch-c{1..(k^2)//4}
          Aggregation:   switch-p{pod}-a{1..k//2}
          Edge:          switch-p{pod}-e{1..ceil(k/2)}
        """
        import math
    
        regions = []
    
        # --- Core region ---
        n_core = (k ** 2) // 4
        core_switches = [f"switch-c{i}" for i in range(1, n_core + 1)]
        regions.append(core_switches)
    
        # --- One region per pod (aggregation + edge) ---
        for pod in range(k):
            pod_switches = []
            n_agg = k // 2
            n_edge = math.ceil(k / 2)
    
            pod_switches += [f"switch-p{pod}-a{i}" for i in range(1, n_agg + 1)]
            pod_switches += [f"switch-p{pod}-e{i}" for i in range(1, n_edge + 1)]
    
            regions.append(pod_switches)
    
        return regions

#################################################################################
def drawFattree(graph, **kwds):  # not really needed, potentially I will remove it
        
        plt.figure(1, figsize=(20, 8), dpi=60)
        positions = {}
        layers = {'core':3, 'aggregation':2, 'edge':1, 'end':0}
        layer_spacing = 1.0  # Adjust the spacing between layers
        xx = {'core':115, 'aggregation':75, 'edge':75, 'end':0}

        for node in graph.nodes():
            if '-c' in node:
                layer = 'core'
            elif '-a' in node:
                layer = 'aggregation'
            elif '-e' in node:
                layer = 'edge'
            else:
                layer = 'end'
            graph.nodes[node]['layer']=layer
            y = layers[layer]
            x = xx[layer]
            xx[layer]+=20
            positions[node]=(x,y)
        pos = nx.multipartite_layout(graph, subset_key='layer', align='horizontal')
        # Draw the graph with adjusted spacing
        nx.draw(graph, pos=positions, font_size=10, **kwds)
#################################################################################
import unittest

class TestUsecaseGenerator(unittest.TestCase):
    
    def test_add(self):
        self.assertEqual(1+2, 3)
