# UsecaseGenerator – Function Documentation

## 1. `k_ary_fat_tree(cls, k, directed=True)`
```python
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


@classmethod
    def k_ary_fat_tree(cls, k, directed=True):
        G = nx.Graph()

        core_switches = [('switch-c%d' % i, {'type': 'switch', 'layer': 'core'}) for i in range(1, (k**2)//4 + 1)] 
        G.add_nodes_from(core_switches)
        
        idx = 1
        hosts = []
        for pod in range(k):
            # adding agg switchs of a pod
            aggSwitches = [('switch-p%d-a%d' % (pod, i), {'type': 'switch', 'layer': 'aggregation'}) for i in range(1, 1 + k//2)]
            G.add_nodes_from(aggSwitches)

            # adding edge switchs of a pod
            edgeSwitches = [('switch-p%d-e%d' % (pod, i), {'type': 'switch', 'layer': 'edge'}) for i in range(1, 1 + math.ceil(k/2))]
            G.add_nodes_from(edgeSwitches)

            # connection within the pod
            for s1, s2 in itertools.product(aggSwitches, edgeSwitches):
                G.add_edge(s1[0], s2[0])

            # insert hosts
            hosts = []
            for sw in edgeSwitches:
                hosts = [('host-%d-p%d' % (idx + i, pod), {'type': 'host', 'layer': 'end'}) for i in range(k//2)]
                G.add_nodes_from(hosts)

                for host in hosts:
                    G.add_edge(host[0], sw[0])

                idx += k//2

        if directed:
            return G.to_directed()
        else:
            return G
```
### Parameters
- `k` *(int)*  
  Number of ports per switch. Determines the size and structure of the fat-tree topology.
- `directed` *(bool, default=True)*  
  Whether the resulting graph should be directed.

### What the Algorithm Does
- Generates a **k-ary fat-tree network topology**, commonly used in data centers.
- Steps:
  1. Creates **core switches** (`k^2 / 4` nodes).
  2. Iterates over `k` pods:
     - Adds **aggregation switches** (`k/2` per pod).
     - Adds **edge switches** (`k/2` per pod, rounded up).
     - Connects aggregation switches to edge switches (full bipartite connections).
  3. For each edge switch:
     - Adds `k/2` **hosts**.
     - Connects each host to its edge switch.

### Output
- A **NetworkX graph (`nx.Graph`)** representing the fat-tree topology.
- Nodes include:
  - Core switches
  - Aggregation switches
  - Edge switches
  - Hosts
- Edges represent physical/logical connections between components.


---

## 2. `makeRandomGraph(cls, usecase, directed=True)`
``` python 
@classmethod
    def makeRandomGraph(cls, usecase, directed=True):
        
        numHosts = usecase.config.getint('topology_config', 'numhosts')
        numSwitchs = usecase.config.getint('topology_config', 'numswitchs')
        minnodedegreepersw = usecase.config.getint('topology_config', 'minnodedegreepersw')
        graphConfig = ast.literal_eval(usecase.config.get('topology_config', 'graphconfig'))

        typ = graphConfig['typ'].replace("'", '')

        # validate input
        if numSwitchs == 1:
            typ = 'er'

        if graphConfig['minnodedegreepersw'] >= numSwitchs:
            print('not enough switches to fulfill the condition on minnodedegreepersw, numSwitchs is increased to', minnodedegreepersw, ' + 1 !!')
            numSwitchs = graphConfig['minnodedegreepersw'] + 1
           
        # generate graph
        try:
            if typ == 'w':
                graph = nx.waxman_graph(numSwitchs)

            if typ == 'ba':
                graph = nx.barabasi_albert_graph(numSwitchs, graphConfig['numEdgN'])

            if typ == 'er':
                graph = nx.erdos_renyi_graph(numSwitchs, graphConfig['prob'])
    
            # check and correct connection
            if graph and numSwitchs > 1:
                for node in graph.nodes():
                    if graph.degree(node) < graphConfig['minnodedegreepersw']:
                        cls.changeDegree(node, graph, graphConfig['minnodedegreepersw'])

            graph = graph.to_directed()

            # rename switches
            mapping = {sw: 'switch-' + str(sw) for sw in graph.nodes()}
            nx.relabel_nodes(graph, mapping, copy=False)
            
            # generate hosts
            switches = list(graph.nodes())[:-graphConfig['numCoreSwitches']]

            for host in range(numHosts):
                hostName = 'host-' + str(host)
                graph.add_node(hostName)

                # link it to a switch
                switch = random.choice(switches)
                graph.add_edge(hostName, switch)
                graph.add_edge(switch, hostName)

            if directed:
                return graph.to_directed()
            else:
                return graph

        except ValueError:
            print('Graph generation failed!!')
            return ValueError
```

### Parameters
- `usecase` *(object)*  
  Object containing configuration data (`usecase.config`) with:
  - `numhosts` → number of host nodes
  - `numswitchs` → number of switch nodes
  - `minnodedegreepersw` → minimum degree per switch
  - `graphconfig` → dictionary with:
    - `typ` → graph type (`'w'`, `'ba'`, `'er'`)
    - `numEdgN` → edges per node (for BA model)
    - `prob` → probability (for ER model)
    - `numCoreSwitches` → number of core switches
- `directed` *(bool, default=True)*  
  Whether the graph should be directed.

### What the Algorithm Does
- Generates a **random network topology** based on configuration.

#### Steps:
1. **Reads configuration** from `usecase`.
2. **Validates input**:
   - Forces `'er'` graph if only one switch exists.
   - Ensures minimum degree constraint is feasible.
3. **Generates base graph** using one of:
   - `'w'`: Waxman graph
   - `'ba'`: Barabási–Albert (scale-free) graph
   - `'er'`: Erdős–Rényi (random) graph
4. **Ensures minimum degree**:
   - Calls `changeDegree` on nodes with insufficient connections.
5. Converts graph to **directed**.
6. **Renames nodes** to `"switch-X"` format.
7. **Adds hosts**:
   - Creates `numHosts` host nodes.
   - Randomly connects each host to a switch.
8. Returns directed or undirected graph.

### Output
- A **NetworkX graph (`nx.Graph` or `nx.DiGraph`)**:
  - Switch nodes (renamed)
  - Host nodes
  - Directed edges if `directed=True`
- Returns `ValueError` if graph generation fails.


---

## 3. `changeDegree(cls, node, graph, newDegree)`
```python
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


class UsecaseGenerator():
    """ 
    randomly generate usecases using the provided parameters
    Output: all timed values in [ns], speed and bw in [mbps], datasize in [bytes]
    """
    index = 0
    __version__ = 1

    def __init__(self):
        pass


    @classmethod
    def changeDegree(cls, node, graph, newDegree):
        degree = graph.degree(node)
        neighbors = [n for n in graph.neighbors(node)]

        if newDegree > degree:
            newNeighbors = random.choices(
                [i for i in graph.nodes() if i not in neighbors + [node]],
                k=newDegree - degree
            )
            for n in newNeighbors:
                graph.add_edge(node, n)

        if newDegree < degree:
            oldneighbors = random.choices(neighbors, k=degree - newDegree)
            for n in oldneighbors:
                graph.remove_edge(node, n)
```

### Parameters
- `node` *(node identifier)*  
  The node whose degree should be modified.
- `graph` *(NetworkX graph)*  
  The graph containing the node.
- `newDegree` *(int)*  
  Desired degree (number of connections) for the node.

### What the Algorithm Does
- Adjusts the number of connections (degree) of a node.

#### Cases:
- **If `newDegree > current degree`**:
  - Randomly selects new nodes not already connected.
  - Adds edges until the desired degree is reached.

- **If `newDegree < current degree`**:
  - Randomly selects existing neighbors.
  - Removes edges until the degree is reduced.

### Output
- Modifies the **input graph in-place**.
- Does **not return anything (`None`)**.
- Ensures the node has approximately the desired degree.


---

## 4. `generateUC(cls, configFile, *args, **kwargs)`
```python
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

    @classmethod
    def generateUC(cls, configFile, *args, **kwargs):
        
        usecase = cls.empty_uc(configFile, *args, **kwargs)
        # generate streams
        list(cls.streamGenerator(usecase))
        usecase.index=cls.index
        cls.index+=1
        return usecase

```
### Parameters
- `configFile` (str / config object)  
  Path or reference to configuration file used to initialize the use case
- `*args`,`*kwargs`  
  Optional parameters passed to `empty_uc()` (e.g., topology, index, flags)

### What the Algortihm Does
- Generates a complete use case with streams  

#### Steps:
1. Calls `empty_uc()` to create a base use case object.
2. Calls `streamGenerator(usecase)`:  
- Iterates through it fully (frocing generation of all streams).
3. Assigns a unique `index` to use case.
4. Increments the class-level index counter.

### Output
- Returns a fully populated `Usecase` object:  
  - Contains:  
    - Topology (graph)
    - Generated streams
    - Assigned index


---

## 5. `ucGenerator(cls, configFile, *args, **kwargs)`
```python
@classmethod
    def ucGenerator(cls, configFile, *args, **kwargs):
        '''incrementally generating use cases by adding one more stream at the time'''
        usecase = cls.empty_uc(configFile, *args, **kwargs)
        streamGen = cls.streamGenerator(usecase)
        for _ in streamGen:
            usecase.index = cls.index
            cls.index+=1
            yield usecase
```

### Parameters
- `configFile`(str / config object)  
Configuration source for the use case.
- `*args`,`**kwargs`  
Passed to `empty_us()`

### What the algorithms does
- Generates use cases **incrementally (step-by-step)** as streams are added.

#### Steps
1. Create an empty use case via `empty_us()`
2. Initializes `streamGenerator(usecase)` (iterator).
3. Iterates through stream generation:
  - After each new stream:
    - Assigns a unique index
    - Yields the current state of the use case

Key idea: Instead of generating everything at once, it **yields immediate versions** of the use case as it grows. 

### Output
- A **generator(`yield`) of `Usecase` objects**:
  - Each yielded object represents:
    - The same use case
    - With progressively more streams added


---

## 6. empty_uc(cls, configFile, index=None, topology=None, newTopo=False)
```python
    @classmethod
    def empty_uc(cls, configFile, index=None, topology=None, newTopo=False):
         # initialise usecase
        usecase = Usecase(configFile, topology=topology)
        if index is not None:
            usecase.index = index
        
        # make usecase topology
        if (usecase.graph is None) or (newTopo):
            usecase.graph = cls.makeTopology(usecase)
        return usecase

```
### Parameters
- `configFile` (str / config object)  
Configuration for initializing the use case.
- `index` (int, optional)  
Optional index to assign to the use case.
- `topology` (graph, optional)  
Predefined topology to reuse.
- `newTopo` (bool, default=False)  
If True, forces creation of a new topology even if one exists.

### What the algorithm does
- Initializes a **base (empty) use case with topology**.

#### Steps:
1. Creates a `Usecase` object: Passes configFile and optional topology.
2. Assigns index if provided.
3. Checks topology:
  - If no graph exists OR newTopo=True:
    - Calls makeTopology(usecase) to generate one.
4. Returns the initialized use case.

### Output
- Returns a `Usecase` object:
  - Contains:
    - Configuration
    - Graph topology (new or reused)
    - Possibly an index
  - Does **not include streams yet** (empty base state)


---

## 7. `streamGenerator(cls, usecase)`
```python
@classmethod
    def streamGenerator(cls, usecase):
        '''
        generate random streams as specified by the usecase configuration
        '''
        routing = ast.literal_eval(usecase.config.get('streams_config', 'routing'))
        fixPath = routing['fixPath']
        #fixPath = usecase.config.getboolean('streams_config', 'fixPath')
        numStreams = usecase.config.getint('streams_config', 'numofstreamsperinstance')
        
        for streamId in range(numStreams):
            stream = cls.generateStream(streamId, usecase)
            numplacesTalker = usecase.config.getint('streams_config', 'numplacesTalker')
            if fixPath and numplacesTalker==1:
                if routing['shortestPath']:
                    stream = cls.shortestPath(stream, usecase)
                else: # random
                    stream = cls.randomPath(stream, usecase) 
                
            usecase.streams.append(stream)
            yield stream
```
### Parameters
- `usecase` *(object)*  
  Usecase object containing:
  - Configuration (`usecase.config`)
  - Topology (`usecase.graph`)
  - Existing streams list (`usecase.streams`)

---

### What the Algorithm Does
- Generates **multiple streams sequentially** based on configuration.

#### Steps:
1. Reads routing configuration:
   - `fixPath` → whether paths should be fixed
   - `shortestPath` → use shortest path or random path
2. Reads `numStreams` → number of streams to generate.
3. Iterates over stream IDs:
   - Calls `generateStream()` to create a stream.
   - If:
     - `fixPath = True` AND
     - only one talker placement:
       - Applies routing:
         - `shortestPath()` OR
         - `randomPath()`
4. Appends each stream to `usecase.streams`.
5. Yields each stream as it is created.

---

### Output
- A **generator of stream dictionaries**:
  - Each yielded item is a stream
- Side effect:
  - Streams are appended to `usecase.streams`

---

## 8. `generateStream(cls, streamId, usecase, talker=None, listeners=None)`
```python
@classmethod
    def generateStream (cls, streamId, usecase, talker=None, listeners=None):
        '''
        stream = streamId, interval(T)[ns], maxFramesPerInterval(Nf), maxFrameSize(l)[byte], MaxLatency(e2e)[ns]
        '''
         # read streams data
        period_range = range(*ast.literal_eval(str(usecase.config.get('streams_config', 'period_range'))).values())
        numberFrames_range = range(*ast.literal_eval(str(usecase.config.get('streams_config', 'numberframes_range'))).values())
        maxLatency_range = [float(i) for i in usecase.config.get("streams_config","maxlatency_range").split(',')]
        dataSize_range = range(*ast.literal_eval(str(usecase.config.get('streams_config', 'datasize_range'))).values())       
        
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
            talker = cls.talkerEmbedding(usecase)
            
        if listeners is None:
            listeners = cls.listenerEmbedding(usecase, talker= talker[0]).tolist()

        
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
```
### Parameters
- `streamId` *(int)*  
  Unique identifier for the stream.
- `usecase` *(object)*  
  Contains configuration and topology.
- `talker` *(list, optional)*  
  Predefined talker placement(s).
- `listeners` *(list, optional)*  
  Predefined listener placement(s).

---

### What the Algorithm Does
- Generates a **single stream with randomized parameters**.

#### Steps:
1. Reads parameter ranges from configuration:
   - `period_range`
   - `numberframes_range`
   - `maxlatency_range`
   - `datasize_range`
2. Randomly selects:
   - Cycle time (`interval`)
   - Frames per interval
   - Frame size
   - Max latency (converted to ns)
3. Handles expiration:
   - If `finite_duration=True`:
     - Random duration is chosen
     - Converted into `lifespanCycles`
4. Determines placements:
   - If `talker` not provided → calls `talkerEmbedding()`
   - If `listeners` not provided → calls `listenerEmbedding()`
5. Constructs stream dictionary with all parameters.
6. Adds optional `lifespanCycles` if finite duration is enabled.

---

### Output
- A **dictionary representing a stream** with fields:
  - `name`
  - `type`
  - `interval` (ns)
  - `maxFramesPerInterval`
  - `maxFrameSize` (bytes)
  - `maxLatency` (ns)
  - `tlkrPssblPlcmnt` (talker placements)
  - `lstnsrPssblPlcmnt` (listener placements)
  - Timing offsets and jitter
  - Optional: `lifespanCycles`

---

## 9. `talkerEmbedding(cls, usecase)`
```python
@classmethod
    def talkerEmbedding(cls, usecase):
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
```
### Parameters
- `usecase` *(object)*  
  Contains:
  - Graph (`usecase.graph`)
  - Existing streams
  - Configuration (`streams_config`)

---

### What the Algorithm Does
- Selects **host nodes to act as talkers** based on configurable probability strategies.

#### Steps:
1. Retrieves:
   - All host nodes from the graph
   - Number of placements (`numplacesTalker`)
2. Initializes uniform probability distribution over hosts.
3. Applies probability modifiers:

   **a. Hotspot (if active):**
   - Increases probability of previously used talkers.

   **b. Stride (if active and multiple placements):**
   - Selects one initial host.
   - Increases probability of hosts at fixed step intervals.
   - Selects remaining placements based on adjusted probabilities.

   **c. Staggered (if active and multiple placements):**
   - Selects one initial host.
   - Boosts:
     - Hosts in same pod (`podp`)
     - Hosts in same edge switch (`edgep`)
   - Optionally combines with hotspot adjustment.

4. If no special mode:
   - Selects hosts uniformly at random.

---

### Output
- A **list of selected host IDs**:
  - Represents possible talker placements
  - Size = `numplacesTalker`


---
## 10. `listenerEmbedding(cls, usecase, talker, listeners=None)`
```python
@classmethod
    def listenerEmbedding(cls, usecase, talker, listeners = None):
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
```
### Parameters
- `usecase` *(object)*  
  Contains:
  - Graph (`usecase.graph`)
  - Existing streams (`usecase.streams`)
  - Configuration (`streams_config`)
- `talker` *(str)*  
  Selected talker host (must be:
  - provided,
  - part of host nodes,
  - excluded from listener selection)
- `listeners` *(list, optional)*  
  Predefined listeners to consider in probability adjustments (default: empty list)

---

### What the Algorithm Does
- Selects **listener host nodes** based on configurable probability strategies.
- Ensures:
  - Talker is excluded from listener selection
  - Listener placements are structured per stream

#### Steps:
1. Retrieves configuration:
   - `stride`
   - `staggered`
   - `hotspot`
   - `numplacesListener` (placements per listener group)
   - `numlistenersperstr` (number of listener groups)

2. Collects all host nodes from the topology.

3. Initializes **uniform probability distribution** over hosts.

4. Applies probability modifiers:

   **a. Hotspot (if active):**
   - Increases probability of previously selected listener hosts
   - Includes:
     - Existing stream listeners
     - Provided `listeners` argument

---

#### Selection Strategies

**b. Stride (if active):**
- Uses talker as reference point
- Increases probability of hosts at fixed step intervals
- Selects:
  - `numplacesListener × numlistenersperstr` hosts
- Reshapes result into:
  - Matrix: `(numlistenersperstr, numplacesListener)`

---

**c. Staggered (if active):**
- Uses talker location to bias selection:
  - Same **pod** → probability = `podp`
  - Same **edge switch** → probability = `edgep`
- Optionally combines with hotspot
- Selects hosts and reshapes into matrix

---

**d. Default (uniform random):**
- Removes talker from candidate hosts
- Selects hosts uniformly
- Reshapes into matrix

---

### Output
- A **NumPy array** of shape: (numlistenersperstr, numplacesListener)
- Each row represents: A set of possible listener placements for one listener group
- Properties:
  - Talker is excluded
  - No replacement (unique selections)
  - Probability-driven selection depending on configuration


---

## 11. `findAllPaths(cls, usecase, talkers=None, listeners=None)`
```python
@classmethod
    def findAllPaths(cls, usecase, talkers=None, listeners=None):
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

        for nd in itertools.product(*[eligibleTalkers,eligibleListeners]):
            pathsGen = nx.all_simple_paths(usecase.graph, nd[0], nd[1])  
            allPaths[nd] = pathsGen
        return allPaths
```
### Parameters
- `usecase` *(object)*  
  Contains:
  - Graph (`usecase.graph`)
- `talkers` *(list, optional)*  
  Subset of talker host nodes to consider.
- `listeners` *(list of lists, optional)*  
  Listener placements (possibly nested lists).

---

### What the Algorithm Does
- Computes **all simple paths** between talker–listener pairs.

#### Steps:
1. Identifies all eligible hosts:
   - Filters nodes starting with `"host"` as potential talkers and listeners.
2. Applies optional filtering:
   - Restricts talkers if `talkers` is provided.
   - Flattens and filters listeners if `listeners` is provided.
3. Generates all combinations of:
   - `(talker, listener)` pairs using Cartesian product.
4. For each pair:
   - Uses `nx.all_simple_paths()` to compute all simple paths.
5. Stores path generators in a dictionary.

---

### Output
- A **dictionary**:
  - Key: `(talker, listener)` tuple
  - Value: generator of all simple paths between them

- Notes:
  - Paths are not materialized immediately (lazy generators)
  - Only host-to-host paths are considered

---

## 12. `shortestPath(cls, stream, usecase)`
```python
@classmethod
    def shortestPath(cls, stream, usecase):
        paths = []
        period_range = range(*ast.literal_eval(str(usecase.config.get('streams_config', 'period_range'))).values())
        shortestPaths = [nx.shortest_path(usecase.graph,pair[0],pair[1]) for pair in itertools.product(*[stream['tlkrPssblPlcmnt'],[item for sublist in stream['lstnsrPssblPlcmnt'] for item in sublist]])]
        
        for path in shortestPaths:
            for hop in range(len(path)-1):
                if usecase.graph.is_directed():
                    path[hop] = usecase.graph.get_edge_data(path[hop], path[hop+1])['sourcePort']
                else:
                    path[hop] = path[hop]+'.'+usecase.graph._adj[path[hop]][path[hop+1]][path[hop]]
                
            if cls.pathConvenient(path, stream['maxLatency'], stream['maxFramesPerInterval'], stream['maxFrameSize'], usecase):
                paths.append(path)

            ##check convenience of period for the given (numFrames*dataSize), that the talker have enough time to transmit in every cycle
            cycle = stream['interval']
            while not cls.periodConvenient(path, cycle,stream['maxFramesPerInterval'],stream['maxFrameSize'], usecase):
                cycle = max(stream['interval'], int(random.choice(period_range)*1000)) # to make it in nanosecond
            stream['interval'] = cycle
            stream['latest-transmit-offset'] = cycle//2
        stream['pathLists'] = paths
        return stream
```
### Parameters
- `stream` *(dict)*  
  Stream object containing:
  - `tlkrPssblPlcmnt` (talker placements)
  - `lstnsrPssblPlcmnt` (listener placements)
  - `interval`, `maxLatency`, `maxFramesPerInterval`, `maxFrameSize`
- `usecase` *(object)*  
  Contains:
  - Graph (`usecase.graph`)
  - Configuration (`streams_config`)

---

### What the Algorithm Does
- Computes and assigns **shortest paths** for a stream while ensuring timing constraints.

#### Steps:
1. Reads `period_range` from configuration.
2. Computes shortest paths for all:
   - Talker–listener combinations.
3. For each path:
   - Iterates over hops:
     - If graph is directed:
       - Replaces nodes with `sourcePort` values.
     - Else:
       - Encodes edge information into node string.
4. Validates path using:
   - `pathConvenient()`:
     - Checks latency and capacity constraints.
   - If valid:
     - Adds path to valid paths list.
5. Ensures transmission timing feasibility:
   - Repeatedly checks `periodConvenient()`
   - Adjusts `stream['interval']` if needed
6. Updates:
   - `latest-transmit-offset`

---

### Output
- Returns the **modified `stream` dictionary** with:
  - `pathLists`: list of valid shortest paths
  - Possibly updated:
    - `interval`
    - `latest-transmit-offset`

- Notes:
  - Only valid paths (meeting constraints) are kept
  - Timing constraints may modify stream parameters


---

## 13. `randomPath(cls, stream, usecase)`
```python
@classmethod
    def randomPath (cls, stream, usecase):
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
                
            if cls.pathConvenient(path, stream['maxLatency'], stream['maxFramesPerInterval'], stream['maxFrameSize'], usecase):
                paths.append(path)
                chosenListeners.append(path[-1])

            if len(chosenListeners) >= len(stream['lstnsrPssblPlcmnt']):
                break

            ##check convenience of period for the given (numFrames*dataSize), that the talker have enough time to transmit in every cycle
            cycle = stream['interval']
            while not cls.periodConvenient(path, cycle,stream['maxFramesPerInterval'],stream['maxFrameSize'], usecase):
                cycle = max(stream['interval'], int(random.choice(period_range)*1000)) # to make it in nanosecond
            stream['interval'] = cycle
            stream['latest-transmit-offset'] = cycle//2
            
        stream['pathLists'] = paths
        stream['lstnsrPssblPlcmnt'] = [chosenListeners]

        return stream
```
### Parameters
- `stream` *(dict)*  
  Stream object containing:
  - `tlkrPssblPlcmnt` (talker placements)
  - `lstnsrPssblPlcmnt` (listener placements)
  - Timing and size parameters (`interval`, `maxLatency`, etc.)
- `usecase` *(object)*  
  Contains:
  - Graph (`usecase.graph`)
  - Configuration (`streams_config`)

---

### What the Algorithm Does
- Selects **random valid paths** between talkers and listeners while enforcing constraints.

#### Steps:
1. Flattens listener placements into a single list.
2. Computes **all simple paths** for each talker–listener pair.
3. Shuffles paths per pair to introduce randomness.
4. Iteratively:
   - Randomly selects a path group
   - Picks a random path from that group
   - Skips if listener already assigned a path
5. Converts path representation:
   - Directed → replace with `sourcePort`
   - Undirected → encode as `"node.port"`
6. Validates path using `pathConvenient()`:
   - If valid:
     - Adds to selected paths
     - Marks listener as assigned
7. Stops when enough listeners are assigned.
8. Ensures timing feasibility:
   - Uses `periodConvenient()`
   - Adjusts `interval` if necessary
9. Updates stream:
   - `pathLists`
   - `lstnsrPssblPlcmnt` (only chosen listeners)

---

### Output
- Returns modified **`stream` dictionary**:
  - `pathLists`: randomly selected valid paths
  - `lstnsrPssblPlcmnt`: reduced to chosen listeners
  - Possibly updated:
    - `interval`
    - `latest-transmit-offset`

---

## 14. `pathConvenient(cls, pathList, maxLatency, maxFramesPerInterval, maxFrameSize, usecase)`
```python
@classmethod
    def pathConvenient (cls, pathList, maxLatency, maxFramesPerInterval, maxFrameSize, usecase):
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
```
### Parameters
- `pathList` *(list)*  
  Path represented as sequence of `"node.port"` or port identifiers.
- `maxLatency` *(int)*  
  Maximum allowed end-to-end latency (ns).
- `maxFramesPerInterval` *(int)*  
  Number of frames transmitted per cycle.
- `maxFrameSize` *(int)*  
  Size of each frame (bytes).
- `usecase` *(object)*  
  Contains graph with:
  - Node processing delay
  - Link propagation delay
  - Port speeds

---

### What the Algorithm Does
- Checks whether a path satisfies **latency constraints**.

#### Steps:
1. Computes total data size:
   - `dataSize = maxFramesPerInterval × maxFrameSize × 8` (bits)
2. Initializes minimum latency = 0.
3. Iterates through path hops:
   - Extracts:
     - Current node
     - Port
     - Next node
   - Computes transmission delay:
     - `dataSize / port speed`
   - Adds:
     - Propagation delay
     - Transmission delay
     - Processing delay
4. Compares total latency with `maxLatency`.

---

### Output
- Returns *(bool)*:
  - `True` → path is valid (latency constraint satisfied)
  - `False` → path exceeds allowed latency

---

## 15. `periodConvenient(cls, pathList, cycle, maxFramesPerInterval, maxFrameSize, usecase)`
```python
@classmethod
    def periodConvenient (cls, pathList, cycle, maxFramesPerInterval, maxFrameSize, usecase):
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
```
### Parameters
- `pathList` *(list)*  
  Path represented as `"node.port"` sequence.
- `cycle` *(int)*  
  Transmission cycle duration (ns).
- `maxFramesPerInterval` *(int)*  
  Number of frames per cycle.
- `maxFrameSize` *(int)*  
  Frame size (bytes).
- `usecase` *(object)*  
  Contains graph with port speeds.

---

### What the Algorithm Does
- Checks whether a path satisfies **timing (cycle) constraints**.

#### Steps:
1. Computes total data size:
   - `dataSize = maxFramesPerInterval × maxFrameSize × 8` (bits)
2. Iterates through path hops:
   - Extracts:
     - Node
     - Port
     - Next node
   - Computes transmission delay:
     - Adjusted for units → speed/1000
3. Verifies:
   - Transmission delay ≤ cycle time
4. Stops early if any hop violates constraint.

---

### Output
- Returns *(bool)*:
  - `True` → path fits within cycle timing constraints
  - `False` → transmission exceeds cycle duration


---

## 16. `makeTopology(cls, usecase)`
```python
# topology generation functions
    @classmethod
    def makeTopology(cls, usecase):
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
            graph = cls.k_ary_fat_tree(graphConfig['k'])
        else:
            graph = cls.makeRandomGraph(usecase)

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

                UsecaseGenerator.set_link_data(graph, 
                                               node, 
                                               succ, 
                                               portName, 
                                               propDelay, 
                                               linkBw, 
                                               portSpeed, 
                                               queueCountPerPort)
                    
                # graph._adj[succ][node]['sourceNode']=succ
                # graph._adj[succ][node]['destNode']=node
                # graph._adj[succ][node]['destPort']=node+'.Port'+str(portName)
                # graph._adj[node][succ]['sourcePort']=node+'.Port'+str(portName)
                # graph._adj[succ][node]['propDelay']= propDelay
                # graph._adj[succ][node]['linkBw']= linkBw
                # graph.nodes[node]['ports']['Port'+str(portName)]= {'name':'Port'+str(portName) ,'speed':portSpeed, 'queueCount':queueCountPerPort}
                if portName > 1:
                    graph.nodes[node]['type'] = 'switch'
                    graph.nodes[node]['procDelay'] = procDelay*1000 # in [ns]

                portName+=1
        return graph
```
### Parameters
- `usecase` *(object)*  
  Contains:
  - Configuration (`usecase.config`)
  - Used to determine topology type and parameters

---

### What the Algorithm Does
- Generates a **network topology graph with detailed link and port properties**.

#### Steps:
1. Reads topology configuration:
   - `portspeed` (Mbps)
   - `propdelay`
   - `procdelay` (µs → later converted to ns)
   - `linkBw` (Mbps)
   - `queuecountperport`
   - `graphconfig` (includes graph type)

2. Selects graph generation method:
   - If `typ = 'ft'` → calls `k_ary_fat_tree()`
   - Else → calls `makeRandomGraph()`

3. Iterates over all nodes and their neighbors:
   - Initializes:
     - Port numbering (`Port1`, `Port2`, …)
     - Node attributes (`ports`, `type`)
   - Ensures adjacency consistency (for directed graphs)

4. Calls `set_link_data()` for each connection:
   - Assigns link and port metadata

5. Determines node type:
   - If multiple ports → node becomes `switch`
   - Adds processing delay (`procDelay`) in ns

---

### Output
- Returns a **NetworkX graph**:
  - Nodes:
    - `host` or `switch`
    - Include port and processing attributes
  - Edges:
    - Include propagation delay, bandwidth, and port mappings

---

## 17. `set_link_data(cls, graph, src, dst, portName, propDelay, linkBw, portSpeed, queueCountPerPort)`
```python
@classmethod
    def set_link_data(cls, graph, src, dst, portName, propDelay, linkBw, portSpeed, queueCountPerPort):
        graph._adj[dst][src]['sourceNode']=dst
        graph._adj[dst][src]['destNode']=src
        graph._adj[dst][src]['destPort']=src+'.Port'+str(portName)
        graph._adj[src][dst]['sourcePort']=src+'.Port'+str(portName)
        graph._adj[dst][src]['propDelay']= propDelay
        graph._adj[dst][src]['linkBw']= linkBw
        graph.nodes[src]['ports']['Port'+str(portName)]= {'name':'Port'+str(portName) ,'speed':portSpeed, 'queueCount':queueCountPerPort}
```
### Parameters
- `graph` *(NetworkX graph)*  
  Graph being modified
- `src` *(node)*  
  Source node
- `dst` *(node)*  
  Destination node
- `portName` *(int)*  
  Port identifier for the source node
- `propDelay` *(int)*  
  Propagation delay of the link
- `linkBw` *(int)*  
  Link bandwidth (Mbps)
- `portSpeed` *(int)*  
  Port transmission speed (Mbps)
- `queueCountPerPort` *(int)*  
  Number of queues per port

---

### What the Algorithm Does
- Assigns **link-level and port-level metadata** to the graph.

#### Steps:
1. Sets edge attributes (reverse direction):
   - `sourceNode`
   - `destNode`
   - `destPort`
   - `propDelay`
   - `linkBw`
2. Sets forward edge attribute:
   - `sourcePort`
3. Updates node port dictionary:
   - Adds port with:
     - Name
     - Speed
     - Queue count

---

### Output
- **No return value (`None`)**
- Side effect:
  - Modifies graph in-place:
    - Adds edge metadata
    - Updates node port information


---

## 18. `k_ary_fat_tree(cls, k, directed=True)`
```python
@classmethod
    def k_ary_fat_tree(cls, k, directed=True):
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
```
### Parameters
- `k` *(int)*  
  Number of ports per switch (defines size of topology)
- `directed` *(bool, default=True)*  
  Whether the graph should be directed

---

### What the Algorithm Does
- Generates a **k-ary fat-tree topology with full hierarchical structure**.

#### Steps:
1. Creates **core switches**:
   - Total: `k² / 4`

2. For each pod (total `k` pods):
   - Adds:
     - `k/2` aggregation switches
     - `k/2` edge switches
   - Fully connects:
     - Aggregation ↔ Edge (bipartite connections)

3. Adds hosts:
   - For each edge switch:
     - Adds `k/2` hosts
     - Connects hosts to edge switch

4. Connects core layer:
   - Each aggregation switch connects to core switches
   - Ensures balanced distribution across cores

5. Converts graph to directed if required

---

### Output
- Returns a **NetworkX graph (`nx.Graph` or `nx.DiGraph`)**:
  - Nodes:
    - Core, aggregation, edge switches
    - Hosts
  - Edges:
    - Represent hierarchical fat-tree connections


---

## 19. `makeRandomGraph(cls, usecase, directed=True)`
```python
@classmethod
    def makeRandomGraph(cls, usecase, directed=True):
        
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
                        cls.changeDegree(node,graph,graphConfig['minnodedegreepersw'])
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
```
### Parameters
- `usecase` *(object)*  
  Contains:
  - Configuration (`usecase.config`) with:
    - `numhosts` → number of host nodes
    - `numswitchs` → number of switch nodes
    - `minnodedegreepersw` → minimum degree per switch
    - `graphconfig` → dictionary with:
      - `typ` → graph type (`'w'`, `'ba'`, `'er'`)
      - `numEdgN` → edges per node (BA model)
      - `prob` → probability (ER model)
      - `numCoreSwitches` → number of core switches
- `directed` *(bool, default=True)*  
  Whether the graph should be directed

---

### What the Algorithm Does
- Generates a **random network topology** and augments it with hosts.

#### Steps:
1. Reads configuration values from `usecase`.
2. Validates inputs:
   - If only one switch → forces Erdős–Rényi (`'er'`) graph
   - Ensures minimum degree constraint is feasible:
     - Increases number of switches if necessary
3. Generates base graph depending on type:
   - `'w'` → Waxman graph
   - `'ba'` → Barabási–Albert graph
   - `'er'` → Erdős–Rényi graph
4. Ensures minimum node degree:
   - Iterates over nodes
   - Calls `changeDegree()` if degree is too low
5. Converts graph to directed.
6. Renames nodes:
   - `"switch-X"` format
7. Adds host nodes:
   - Creates `numHosts` hosts
   - Randomly connects each host to a switch
   - Adds bidirectional edges for directed graph
8. Returns graph in desired format (directed/undirected)

---

### Output
- Returns a **NetworkX graph (`nx.Graph` or `nx.DiGraph`)**:
  - Switch nodes (renamed)
  - Host nodes
  - Directed edges (if enabled)
- Returns `ValueError` if generation fails

---

## 20. `changeDegree(cls, node, graph, newDegree)`
```python
@classmethod
    def changeDegree (cls, node, graph, newDegree):
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
```
### Parameters
- `node` *(node identifier)*  
  Node whose degree should be modified
- `graph` *(NetworkX graph)*  
  Graph containing the node
- `newDegree` *(int)*  
  Desired number of connections for the node

---

### What the Algorithm Does
- Adjusts the **degree (number of connections)** of a node.

#### Cases:

**1. Increase degree (`newDegree > current degree`):**
- Selects random nodes not already connected
- Adds edges until desired degree is reached

**2. Decrease degree (`newDegree < current degree`):**
- Selects random existing neighbors
- Removes edges until desired degree is reached

---

### Output
- **No return value (`None`)**
- Side effect:
  - Modifies the graph in-place
  - Ensures node degree approximates `newDegree`

# Usecase - Function Documentation
```python
# constants
MBPS_2_BPNS = 0.001
BYTE_2_BIT = 8
TOL = 1.1  # tolerance is 0.1 of the frame length

class  Usecase():
    def __init__(self, configFile, index= None, topology=None, streams=None, schedule=None):
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
        self.schedule = schedule
```
### Parameters
- `configFile` *(str)*  
  Path to configuration file defining topology and stream parameters
- `index` *(int, optional)*  
  Identifier for the use case
- `topology` *(NetworkX graph, optional)*  
  Predefined topology
- `streams` *(list, optional)*  
  Predefined list of streams
- `schedule` *(object, optional)*  
  Scheduling information

---

### What the Algorithm Does
- Initializes a **Usecase object** with configuration, topology, and streams.

#### Steps:
1. Loads configuration using `configparser`.
2. Assigns:
   - `index`
   - `schedule`
3. Initializes topology:
   - Converts to directed graph if provided
4. Initializes streams:
   - Empty list if none provided

---

### Output
- A **Usecase object** with:
  - Configuration loaded
  - Graph initialized (optional)
  - Streams list initialized
  - Schedule assigned

---

## 2. `hosts (property)`
```python
    @property
    def hosts(self):
        return [node for node in self.graph.nodes if 'host' in node]
```
### Parameters
- None (uses internal graph)

---

### What the Algorithm Does
- Extracts all **host nodes** from the topology.

#### Steps:
1. Iterates over graph nodes
2. Selects nodes containing `"host"` in their name

---

### Output
- Returns a **list of host node names**

---

## 3. `_frame_size(self, stream_idx)`
```python
    def _frame_size(self, stream_idx):
        """
        Compute the frame size in bits *with tolerance*.
        """
        stream = self.streams[stream_idx]
        return int(
            stream["maxFramesPerInterval"]
            * stream["maxFrameSize"]
            * BYTE_2_BIT
            * TOL # as a rate of the frame size
        )
```
### Parameters
- `stream_idx` *(int)*  
  Index of the stream in `self.streams`

---

### What the Algorithm Does
- Computes **frame size in bits with tolerance**.

#### Steps:
1. Retrieves stream parameters:
   - `maxFramesPerInterval`
   - `maxFrameSize`
2. Converts:
   - Bytes → bits
3. Applies tolerance factor (`TOL`)

---

### Output
- Returns *(int)*:
  - Frame size in bits (with tolerance)

---

## 4. `window_length(self, stream_idx, port_name, eps=1e-9)`
```python
    def window_length(self, stream_idx, port_name, eps = 1e-9):
        frame_size = self._frame_size(stream_idx)
        return frame_size / max(self.ports[port_name]['speed'] * MBPS_2_BPNS, eps)
```
### Parameters
- `stream_idx` *(int)*  
  Index of the stream
- `port_name` *(str)*  
  Port identifier
- `eps` *(float, default=1e-9)*  
  Small value to avoid division by zero

---

### What the Algorithm Does
- Computes **transmission window length** for a stream on a given port.

#### Steps:
1. Computes frame size using `_frame_size()`
2. Retrieves port speed
3. Converts speed:
   - Mbps → bits/ns
4. Calculates:
   - `window = frame_size / port_speed`

---

### Output
- Returns *(float)*:
  - Transmission window length (ns)

---

## 5. `distance_to_hosts (property)`
```python
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
```
### Parameters
- None (uses internal graph)

---

### What the Algorithm Does
- Computes **shortest path distances from each host to all nodes**.

#### Steps:
1. Iterates over all host nodes
2. For each host:
   - Runs Dijkstra shortest path algorithm
3. Stores distances per host

---

### Output
- Returns a **dictionary**: {host: {node: distance}}


---

## 6. `switches (property)`
```python
    @property
    def switches(self):
        return [node for node in self.graph.nodes if 'switch' in node]
```
### Parameters
- None

---

### What the Algorithm Does
- Extracts all **switch nodes** from the topology.

#### Steps:
1. Iterates over graph nodes
2. Selects nodes containing `"switch"`

---

### Output
- Returns a **list of switch node names**

---

## 7. `links (property)`
```python
    @property
    def links(self):
        return [d for _,_,d in self.graph.edges(data=True)]

```
### Parameters
- None

---

### What the Algorithm Does
- Extracts all **edge data (links)** from the graph.

#### Steps:
1. Iterates over graph edges
2. Collects edge attribute dictionaries

---

### Output
- Returns a **list of edge attribute dictionaries**

---

## 8. `ports (property)`
```python
    @property
    def ports(self):
        ports = {}
        for _, _, d in self.graph.edges(data=True):
            ports[d['sourcePort']]={'sourceNode':d['sourceNode'],'destPort':d['destPort'], 'destNode':d['destNode']}
        for node, data in self.graph.nodes(data=True):
            for pname, pdata in data['ports'].items():   
                ports[node+'.'+pname].update(name=node+'.'+pname, speed=pdata['speed'], queueCount=pdata['queueCount'],nodeName=node)
        return ports

```
### Parameters
- None

---

### What the Algorithm Does
- Builds a **comprehensive port mapping** combining:
- Edge data
- Node port configurations

#### Steps:
1. Iterates over edges:
 - Extracts:
   - `sourcePort`, `sourceNode`, `destNode`, `destPort`
2. Iterates over node port definitions:
 - Adds:
   - Port name
   - Speed
   - Queue count
   - Node name
3. Merges both into a unified dictionary

---

### Output
- Returns a **dictionary**:{
port_name: {
sourceNode,
destNode,
destPort,
speed,
queueCount,
nodeName,
...
}
}


---

## 9. `port_name_id_mapping (property)`
```python
    @property
    def port_name_id_mapping(self):
        """name -> 1-based port id mapping. includes '' -> 0."""
        port_name_to_id = {key: index for index, key in enumerate(self.ports.keys(), start=1)}
        port_name_to_id[''] = 0
        port_id_to_name = dict(enumerate(self.ports.keys(), start=1))
        port_id_to_name[0] = ''
        return port_name_to_id, port_id_to_name
```
### Parameters
- None

---

### What the Algorithm Does
- Creates **bidirectional mapping between port names and numeric IDs**.

#### Steps:
1. Enumerates all port names:
 - Assigns IDs starting from 1
2. Adds:
 - Empty string (`''`) mapped to `0`
3. Builds reverse mapping:
 - ID → port name

---

### Output
- Returns a **tuple of two dictionaries**:
1. `name → id`
2. `id → name`

## 10. `addSchedule(self, schedule)`
```python
    def addSchedule(self, schedule):
        _, port_id_to_name = self.port_name_id_mapping
        self.schedule = schedule
        schedule.update_graph(self.graph, port_id_to_name)
        schedule.update_streams(self.streams, self.ports, port_id_to_name)
        # update the usecase config
        # routing = ast.literal_eval(self.config.get('streams_config', 'routing'))
        # routing['fixPath'] = True
        # self.config.set('streams_config', 'routing', str(routing))
        # self.config.set('streams_config','numplacesTalker',1)
        # self.config.set('streams_config','numplaceslistener',1)
```
### Parameters
- `schedule` *(object)*  
  Scheduling object that supports:
  - `update_graph(graph, port_id_to_name)`
  - `update_streams(streams, ports, port_id_to_name)`

---

### What the Algorithm Does
- Attaches a **schedule to the use case** and propagates it to graph and streams.

#### Steps:
1. Retrieves port ID → name mapping.
2. Assigns schedule to `self.schedule`.
3. Updates graph:
   - Calls `schedule.update_graph()`
4. Updates streams:
   - Calls `schedule.update_streams()`

---

### Output
- **No return value (`None`)**
- Effects:
  - Schedule stored in use case
  - Graph and streams updated with scheduling data

---

## 11. `removeSchedule(self)`
```python
    def removeSchedule(self):
        self.schedule = None
        for _ , node_data in self.graph.nodes(data=True):
            for port, port_data in node_data['ports'].items():
                if 'schedule' in port_data:
                    del port_data['schedule']
                if 'basetime' in port_data:
                    del port_data['basetime']
                if 'scheduleCycle' in port_data:
                    del port_data['scheduleCycle']
                    
        for stream in self.streams:
            if 'offset' in stream:
                    del stream['offset']
            if 'accLatency' in stream:
                    del stream['accLatency']
```
### Parameters
- None

---

### What the Algorithm Does
- Removes **all scheduling information** from the use case.

#### Steps:
1. Sets `self.schedule = None`
2. Iterates over all node ports:
   - Removes:
     - `schedule`
     - `basetime`
     - `scheduleCycle`
3. Iterates over streams:
   - Removes:
     - `offset`
     - `accLatency`

---

### Output
- **No return value (`None`)**
- Effects:
  - Cleans all scheduling-related data from graph and streams

---

## 12. `addRouting(self, overwrite_config=False)`
```python
    def addRouting(self, overwrite_config=False):
        numplacesTalker = self.config.getint('streams_config', 'numplacesTalker')
        if numplacesTalker!=1:
            return
            
        routing = ast.literal_eval(self.config.get('streams_config', 'routing'))
        if routing.get('fixPath', False) or overwrite_config:
            routing['fixPath'] = True
            self.config.set('streams_config', 'routing', str(routing))
            for stream in self.streams:
                if routing.get('shortestPath', False):
                    stream = UsecaseGenerator.shortestPath(stream, self)
                else: # random
                    stream = UsecaseGenerator.randomPath(stream, self)
```
### Parameters
- `overwrite_config` *(bool, default=False)*  
  Forces routing configuration update

---

### What the Algorithm Does
- Applies **routing paths** to all streams.

#### Steps:
1. Checks:
   - Only applies if `numplacesTalker == 1`
2. Reads routing configuration
3. If:
   - `fixPath = True` OR `overwrite_config=True`:
     - Forces fixed routing
4. For each stream:
   - If `shortestPath=True`:
     - Calls `shortestPath()`
   - Else:
     - Calls `randomPath()`

---

### Output
- **No return value (`None`)**
- Effects:
  - Updates streams with:
    - `pathLists`
    - Possibly adjusted timing parameters

---

## 13. `removeRouting(self)`
```python
    def removeRouting(self):
        routing = ast.literal_eval(self.config.get('streams_config', 'routing'))
        routing['fixPath'] = False
        self.config.set('streams_config', 'routing', str(routing))
        for stream in self.streams:
            try:
                del stream['pathLists']
            except:
                pass
```
### Parameters
- None

---

### What the Algorithm Does
- Removes **routing paths** from all streams.

#### Steps:
1. Updates configuration:
   - Sets `fixPath = False`
2. Iterates over streams:
   - Deletes `pathLists` if present

---

### Output
- **No return value (`None`)**
- Effects:
  - Streams no longer contain routing information

---

## 14. `fixTalkers(self)`
```python
    def fixTalkers(self):
        self.config.set('streams_config','numplacestalker',1)
        for stream in self.streams:
            stream['tlkrPssblPlcmnt'] = [random.choice(stream['tlkrPssblPlcmnt'])]
```
### Parameters
- None

---

### What the Algorithm Does
- Restricts each stream to **one fixed talker**.

#### Steps:
1. Updates configuration:
   - Sets `numplacesTalker = 1`
2. For each stream:
   - Randomly selects one talker from possible placements

---

### Output
- **No return value (`None`)**
- Effects:
  - Each stream has exactly one talker

---

## 15. `fixListeners(self)`
```python
    def fixListeners(self):
        self.config.set('streams_config','numplaceslistener',1)
        for stream in self.streams:
            stream['lstnsrPssblPlcmnt'] = [[random.choice(listener)] for listener in stream['lstnsrPssblPlcmnt']]

```
### Parameters
- None

---

### What the Algorithm Does
- Restricts each stream to **one listener per group**.

#### Steps:
1. Updates configuration:
   - Sets `numplacesListener = 1`
2. For each stream:
   - Randomly selects one listener per listener group

---

### Output
- **No return value (`None`)**
- Effects:
  - Listener placements reduced to single choices

---

## 16. `remove_stream(self, stream_ids)`
```python
    def remove_stream(self, stream_ids):
        for stream_id in stream_ids:
            del self.streams[stream_id]
```
### Parameters
- `stream_ids` *(list)*  
  Indices of streams to remove

---

### What the Algorithm Does
- Removes specified streams from the use case.

#### Steps:
1. Iterates over provided indices
2. Deletes corresponding streams

---

### Output
- **No return value (`None`)**
- Effects:
  - Streams list is modified

---

## 17. `add_stream(self)`
```python
    def add_stream(self):
        self.streams.append(UsecaseGenerator.generateStream(len(self.streams), self))
```
### Parameters
- None

---

### What the Algorithm Does
- Adds a **new randomly generated stream**.

#### Steps:
1. Calls `UsecaseGenerator.generateStream()`
   - Uses current number of streams as ID
2. Appends new stream to list

---

### Output
- **No return value (`None`)**
- Effects:
  - New stream added to `self.streams`

---

## 18. `remove_link(self, src, dst)`
```python
    def remove_link(self, src, dst):
        try:
            self.graph.remove_edge(src, dst)
        except:
            pass
```
### Parameters
- `src` *(node)*  
  Source node
- `dst` *(node)*  
  Destination node

---

### What the Algorithm Does
- Removes a **link (edge)** from the graph.

#### Steps:
1. Attempts to remove edge `(src, dst)`
2. Ignores errors if edge does not exist

---

### Output
- **No return value (`None`)**
- Effects:
  - Graph topology modified

---

## 19. `add_link(self, src, dst)`
```python
    def add_link(self, src, dst):
        try:
            self.graph.add_edge(src, dst)

            portName = self.graph.out_degree(src)+1
            propDelay = self.config.getint('topology_config', 'propdelay')
            linkBw = self.config.getint('topology_config', 'linkBw') # mbps
            portSpeed  = self.config.getint('topology_config', 'portspeed') # mbps
            queueCountPerPort = self.config.getint('topology_config', 'queuecountperport')
            # update link data according to usecase config
            UsecaseGenerator.set_link_data(graph, node, succ, portName, 
                                           propDelay, 
                                           linkBw, 
                                           portSpeed, 
                                           queueCountPerPort)
        except:
            pass
```
### Parameters
- `src` *(node)*  
  Source node
- `dst` *(node)*  
  Destination node

---

### What the Algorithm Does
- Adds a **link (edge)** and assigns link properties.

#### Steps:
1. Adds edge `(src, dst)` to graph
2. Determines:
   - Port number
   - Link parameters from configuration
3. Calls `set_link_data()` to assign metadata

---

### Output
- **No return value (`None`)**
- Effects:
  - Graph updated with new edge and link attributes

## 20. `to_file(self, jsnPath=None, ucFileName=None)`
```python
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
```
### Parameters
- `jsnPath` *(str, optional)*  
  Directory path where the file will be saved (default: current directory)
- `ucFileName` *(str, optional)*  
  Name of the output JSON file

---

### What the Algorithm Does
- Serializes the use case and **writes it to a JSON file**.

#### Steps:
1. Calls `to_json()` to convert use case into dictionary format.
2. Sets default path if none provided.
3. Sets default filename if none provided:
   - Uses configuration name + `.json`
4. Writes JSON data to file using `json.dump()`.
5. Prints confirmation message.

---

### Output
- Returns *(str)*:
  - Name of the saved file
- Effects:
  - JSON file written to disk

---

## 21. `to_json(self)`
```python
    def to_json(self):
        usecase = self.topology2json()
        usecase['streams'] = []
        for stream in self.streams:
            usecase['streams'].append({key: value for key, value in stream.items() if key!="customGraph"})
        usecase['index'] = self.index
        return usecase
```
### Parameters
- None

---

### What the Algorithm Does
- Converts the use case into a **JSON-serializable dictionary**.

#### Steps:
1. Calls `topology2json()` to serialize graph.
2. Initializes empty `streams` list.
3. Iterates over streams:
   - Copies all fields except `"customGraph"`
4. Adds:
   - `streams`
   - `index`

---

### Output
- Returns a **dictionary**:{
nodes: [...],
links: [...],
streams: [...],
index: int
}


---

## 22. `topology2json(self, source="sourcePort", target="destPort", name="name", key="key", linkSym="links", nodeSym="nodes")`
```python
    def topology2json(self, source="sourcePort", target="destPort", name="name", key="key", linkSym="links",nodeSym="nodes"):
        '''
        this function adds ports attributes to the graph and export it to json
        !!! it changes the graph permanently !!!
        '''
        if len({source, target, key}) < 3:
            raise nx.NetworkXError("Attribute names are not unique.")
        data = {nodeSym: 
                [dict([
                    ('type',self.graph.nodes[n]['type']),
                    (name, n),
                    ('ports',self.graph.nodes[n]['ports'])
                ])for n in self.graph if self.graph.nodes[n]['type']=='host'] + 
                [dict([
                    ('type',self.graph.nodes[n]['type']),
                    (name, n),
                    ('procDelay', self.graph.nodes[n]['procDelay']),
                    ('ports',self.graph.nodes[n]['ports'])
                ])for n in self.graph if self.graph.nodes[n]['type']=='switch'],

                linkSym: [d for _,_,d in self.graph.edges(data=True)]
                #[dict([(source, d[source]), (target, d[target]), ('sourceNode', u), ('destNode', v), ('propDelay', d['propDelay']), ('linkBw', d['linkBw'])]) for u, v, d in self.graph.edges(data=True)]
        }
        return data
```
### Parameters
- `source` *(str, default="sourcePort")*  
Source port attribute name
- `target` *(str, default="destPort")*  
Destination port attribute name
- `name` *(str, default="name")*  
Node name attribute
- `key` *(str, default="key")*  
Edge key identifier
- `linkSym` *(str, default="links")*  
Key name for links in output
- `nodeSym` *(str, default="nodes")*  
Key name for nodes in output

---

### What the Algorithm Does
- Converts the **graph topology into JSON format**, including ports.

#### Steps:
1. Validates that attribute names are unique.
2. Builds node list:
 - Hosts:
   - `type`, `name`, `ports`
 - Switches:
   - `type`, `name`, `procDelay`, `ports`
3. Builds edge list:
 - Includes full edge attribute dictionaries
4. Combines into a single dictionary

⚠️ Note:
- This function **modifies the graph permanently**

---

### Output
- Returns a **dictionary**:{
nodes: [...],
links: [...]
}


---

## 23. `from_json(cls, impUC)`
```python
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
        # ompile the schedule object
        port_name_to_id, _ = self.port_name_id_mapping
        ucc.schedule = Schedule.from_usecase(ucc, port_name_to_id)
        return ucc
```
### Parameters
- `impUC` *(dict)*  
Imported use case JSON structure containing:
- `nodes`
- `links`
- `streams`
- `index`

---

### What the Algorithm Does
- Reconstructs a **Usecase object from JSON data**.

#### Steps:
1. Creates directed graph (`nx.DiGraph`).
2. Adds nodes:
 - Includes node attributes
3. Adds edges:
 - Includes:
   - Ports
   - Delays
   - Bandwidth
4. Instantiates `Usecase`:
 - Assigns graph, streams, and index
5. Infers configuration using `guess_config()`
6. Rebuilds schedule:
 - Uses `Schedule.from_usecase()`

---

### Output
- Returns a **Usecase object**:
- Graph reconstructed
- Streams restored
- Schedule recompiled

## 24. `guess_config(cls, impUC)`
```python
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
```
### Parameters
- `impUC` *(dict)*  
  Imported use case JSON containing:
  - `nodes`
  - `links`
  - `streams`
  - `index`

---

### What the Algorithm Does
- Infers a **configuration file** from an imported use case.

#### Steps:
1. Initializes a new `configparser` object.
2. Extracts and aggregates stream parameters:
   - Groups values per key across all streams
3. Extracts topology data:
   - Nodes, links, switches
4. Selects a sample port for default values.

5. Builds `topology_config`:
   - Number of hosts and switches
   - Port speed
   - Processing delay (converted to µs)
   - Propagation delay
   - Link bandwidth
   - Queue count
   - Graph type set to `"imported"`

6. Builds `streams_config`:
   - Number of streams
   - Ranges:
     - Period (converted to µs)
     - Frames per interval
     - Data size
     - Max latency (converted to ms)
   - Placement parameters:
     - Talkers
     - Listeners
     - Listener groups
   - Routing:
     - Enabled if `pathLists` exist

---

### Output
- Returns a **configparser object**:
  - Contains reconstructed `topology_config` and `streams_config`

---

## 25. `get_config(self)`
```python
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
```
### Parameters
- None

---

### What the Algorithm Does
- Generates a **string representation of the configuration**.

#### Steps:
1. Reads topology configuration:
   - If fat-tree:
     - Encodes `ft-k`
   - Else:
     - Encodes graph parameters and counts
2. Appends:
   - Port speed
   - Propagation delay
   - Processing delay
   - Link bandwidth
   - Queue count

3. Appends stream configuration:
   - Number of streams
   - Talker/listener placements
   - Listener groups
4. Appends routing and traffic modifiers:
   - `routing`, `stride`, `staggered`, `hotspot`
5. Appends parameter ranges:
   - Period
   - Frames
   - Data size
   - Latency

6. Cleans trailing separators

---

### Output
- Returns *(str)*:
  - Encoded configuration string
- Used for:
  - File naming
  - Identification of use cases

---

## 26. `generate_fattree_regions(k: int)`
```python
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
```
### Parameters
- `k` *(int)*  
  Number of ports per switch in the fat-tree

---

### What the Algorithm Does
- Generates **topology-aware regions** for a fat-tree network.

#### Steps:
1. Computes number of core switches:
   - `k² / 4`
2. Creates **core region**:
   - All core switches grouped together

3. Iterates over each pod (`k` pods):
   - Computes:
     - Aggregation switches (`k/2`)
     - Edge switches (`ceil(k/2)`)
   - Combines them into one region per pod

---

### Output
- Returns a **list of lists**:[
[core switches],
[pod 0 switches],
[pod 1 switches],
...
]

- Properties:
  - Regions are non-overlapping
  - Reflect fat-tree hierarchy

## 27. `drawTopology(self)`
```python
################
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
```
### Parameters
- None

---

### What the Algorithm Does
- Visualizes the **network topology graph** using layered layouts.

#### Steps:
1. Initializes a matplotlib figure.
2. Defines:
   - Layer mapping:
     - `core → 3`
     - `aggregation → 2`
     - `edge → 1`
     - `end → 0`
   - Initial x-coordinates per layer

3. Reads graph type from configuration.

---

#### Case 1: Fat-tree (`typ = 'ft'`)
1. Iterates over nodes:
   - Uses predefined `layer` attribute
   - Assigns `(x, y)` positions manually
2. Uses `nx.multipartite_layout()` for structure
3. Draws graph with labels

---

#### Case 2: Imported topology (`typ = 'imported'`)
1. Attempts to infer layers from node names:
   - `-c` → core
   - `-a` → aggregation
   - `-e` → edge
   - else → end
2. Assigns positions similarly to fat-tree
3. Draws graph
4. Fallback:
   - If error occurs → draws using default layout

---

#### Case 3: Other graph types
- Uses default `nx.draw()` layout

---

### Output
- **No return value (`None`)**
- Effects:
  - Displays a plotted graph using matplotlib

---

## 28. `drawFattree(graph, **kwds)`
```python
######################
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
```
### Parameters
- `graph` *(NetworkX graph)*  
  Graph to visualize
- `**kwds` *(optional)*  
  Additional keyword arguments for `nx.draw()`

---

### What the Algorithm Does
- Visualizes a **fat-tree topology** using layered layout.

#### Steps:
1. Initializes matplotlib figure.
2. Defines layer mapping and x-coordinates.
3. Iterates over nodes:
   - Infers layer from node name:
     - `-c`, `-a`, `-e`, or default
   - Assigns `(x, y)` positions
   - Stores layer in node attributes
4. Uses `nx.multipartite_layout()` for alignment
5. Draws graph with provided drawing options

---

### Output
- **No return value (`None`)**
- Effects:
  - Displays a plotted fat-tree graph

# TSN Scheduling Problem - Libraries constants and `require_instance(func)`

```python
import math
import time
import json
import ast
from collections import deque

import numpy as np
import networkx as nx

from usecase_generator_json import UsecaseGenerator, Usecase
from instancesLoader_json import InstancesLoader

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


# constants
BYTE_2_BIT = 8
MBPS_2_BPNS = 0.001
TOL = 1.1  # tolerance is 0.1 of the frame length

def require_instance(func):
    """Decorator: call self.no_instance_assigned() if no usecase is active."""
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "usecase", None):
            return self.no_instance_assigned()
        return func(self, *args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper
```
### Parameters
- `func` (callable) The method/function to be wrapped by the decorator

### What the algorithm does
- Acts as a decorator to enforce that an object has an active `usecase` before executing a method.
#### Steps
1. Defines an inner function `wrapper(self, *args, **kwargs)`.
2. Checks whether the object (self) has a usecase attribute:  
   - Uses getattr(self, "usecase", None) to safely access it.
3. If no `usecase` is assigned:
   - Calls `self.no_instance_assigned()` and returns its result.
4. If a `usecase` exists:
   - Executes the original function `func` with all arguments.
5. Preserves the original function name by assigning `wrapper.__name__ = func.__name__`.
6. Returns the `wrapper` function.

### Output 
- **Returns**: (callable) A wrapped version of the input function with precondition checking.
- **Effects**:
   - Prevents execution of methods when no `usecase` is assigned.
   - Delegates handling to `self.no_instance_assigned()` when the condition fails.

# InstanceContext
```python
class InstanceContext:
    """
    Context manager to temporarily set the current usecase (instance).
    It will call parent.build_metadata() once per usecase (by id).
    """
    def __init__(self, parent, instance):
        self.parent = parent
        self.instance = instance

    def __enter__(self):
        self.parent.usecase = self.instance

        # create set to track built usecases (store ids to avoid relying on hashability)
        if not hasattr(self.parent, "_metadata_built_for"):
            self.parent._metadata_built_for = set()

        uid = id(self.instance)
        if uid not in self.parent._metadata_built_for:
            self.parent.build_metadata()
            self.parent._metadata_built_for.add(uid)

        return self.parent

    def __exit__(self, exc_type, exc_value, traceback):
        self.parent.usecase = None
        # do NOT remove from _metadata_built_for; metadata is cached
```
### Parameters
- parent (object): The object that holds the active usecase and provides build_metadata().
- instance (object): The usecase instance to be temporarily assigned.

### What the Algorithm Does
- Implements a **context manager** that safely assigns a usecase to a parent object.
- Ensures that metadata for each usecase is built only once and then cached.

#### Steps:
1. **Initialization (`__init__`)**
   - Stores references to `parent` and `instance`.
2. **Entering the Context (__enter__)**
   - Assigns the given `instance` to `parent.usecase`.
   - Checks if `_metadata_built_for` exists on the parent:
     - If not, initializes it as an empty set.
   - Computes a unique identifier for the instance using `id(instance)`.
   - If this instance has not been processed before:
     - Calls `parent.build_metadata()`.
     - Stores the instance ID in `_metadata_built_for`.
   - Returns the `parent` object for use inside the context.
3. Exiting the Context (`__exit__`)
   - Resets `parent.usecase` to `None`.
   - Keeps `_metadata_built_for` unchanged to preserve cached metadata.

### Output
- Returns: (object) The `parent` object when entering the context.
- Effects:
   - Temporarily assigns `parent.usecase`.
   - Builds metadata once per unique instance.
   - Caches processed instances using their IDs.
   - Resets the usecase after exiting the context.

# TSNScheduling(problem_definition, ContextHandler=InstanceContext, **kwargs)
```python
class TSNScheduling():
    """
    Cleaned TSN scheduling problem class (step 1).
    - uses lists/dicts for streams/ports
    - metadata is built lazily and cached per-instance (usecase)
    """

    def __init__(self, problem_definition, ContextHandler=InstanceContext, **kwargs):
        # core helpers
        self.uc_gen = UsecaseGenerator()
        self.instances_loader = InstancesLoader()
        self.config_file = problem_definition

        # instance & context
        self.usecase = None
        self.ContextHandler = ContextHandler
        self.instances = []

        # metadata (initialized per-instance in build_metadata)
        self.config_matrix = None
```

### Parameters
- problem_definition (str / dict / object): Configuration or input definition used to construct TSN scheduling instances.
- ContextHandler (class, optional) Context manager used to handle instance switching. Defaults to InstanceContext.
- **kwargs (optional) Additional optional arguments (currently unused but allows extensibility).

### What the Algorithm Does
- Initializes a TSN (Time-Sensitive Networking) scheduling problem handler.
- Sets up core components required for generating and loading use cases.
- Prepares internal structures for lazy metadata construction per instance.

#### Steps:
1. Initialize Core Helpers
   - Creates a `UsecaseGenerator` instance for generating use cases.
   - Creates an `InstancesLoader` instance for loading predefined instances.
   - Stores the provided problem definition.
2. Initialize Context Handling
   - Sets the current `usecase` to `None`.
   - Assigns the context manager (`ContextHandler`) for controlled instance switching.
   - Initializes an empty list to store available instances.
3. Initialize Metadata Storage
   - Sets config_matrix to None.
   - Metadata will be built later using a dedicated method (build_metadata) and cached per instance.

### Output
- Returns: (None) Constructor initializes the object.
- Effects:
   - Instantiates helper objects (`UsecaseGenerator`, `InstancesLoader`).
   - Prepares internal state for managing multiple scheduling instances.
   - Defers metadata creation until explicitly needed (lazy initialization).

## 1. build_metadata(self)
```python
    # -------------------------
    # metadata
    # -------------------------
    def build_metadata(self):
        """
        Build config_matrix and attach customGraph to each stream dict.
        - Expects self.usecase to be set.
        - Uses make_default_stream_config(stream) which returns (config_column, custom_graph).
        """
        if not self.usecase:
            return self.no_instance_assigned()

        # allocate matrix
        self.config_matrix = np.zeros((self.n_ports, self.n_streams), dtype=int)

        # iterate streams in the raw usecase (list of dicts)
        for j, stream in enumerate(self.usecase.streams):
            config_column, custom_graph = self.make_default_stream_config(stream)

            # sanity check: config_column must match number of ports
            if len(config_column) != self.n_ports:
                raise ValueError(
                    f"config column length ({len(config_column)}) != n_ports ({self.n_ports})"
                )

            self.config_matrix[:, j] = config_column
            # attach custom graph to the stream dict so downstream code can use it
            stream["customGraph"] = custom_graph
            
        # After metadata is built, initialize Problem
        self.initialize_problem()
```
### Parameters
- self (TSNScheduling): Instance of the scheduling class containing the active usecase.

### What the algorithms does
- Builds metadata structures required for TSN scheduling.
- Constructs a configuration matrix mapping ports to streams.
- Generates and attaches a custom graph per stream.
- Initializes the problem after metadata preparation.

#### Steps:
1. Precondition Check
   - Verifies that `self.usecase` is set.
   - If not, calls `self.no_instance_assigned()`.
2. Initialize Configuration Matrix
   - Creates a zero matrix of shape `(n_ports, n_streams)`.
3. Process Each Stream
   - Iterates over all streams in `self.usecase.streams`.
   - Calls `make_default_stream_config(stream)`:
     - Returns:
       - `config_column`: port configuration vector
       - `custom_graph`: graph structure for the stream
4. Validate Configuration
   - Ensures `config_column` length matches `n_ports`.
   - Raises `ValueError` if mismatch occurs.
5. Populate Matrix
   - Assigns `config_column` to column `j` of `config_matrix`.
6. Attach Stream Metadata
   - Stores `custom_graph` inside the stream dictionary under "customGraph".
7. Finalize Initialization
   - Calls `self.initialize_problem()` after all metadata is built.

### Output
- Returns: (None)
- Effects:
   - Populates self.config_matrix.
   - Modifies each stream by adding "customGraph".
   - May raise ValueError on dimension mismatch.
   - Triggers further initialization via initialize_problem().

## 2. initialize_problem(self)
```python
    def initialize_problem(self):
        n_var = self.n_ports * self.n_streams
        fr, ftp = self.get_task()
        check_routing = not (fr and ftp)
        self.evaluator = TSNConstraintEvaluator(self, check_routing=check_routing, check_task_placement=not ftp)
```

### Parameters
- self (TSNScheduling) Instance of the scheduling class with initialized metadata.

### What the Algorithm Does
- Initializes the **constraint evaluation component** for the TSN scheduling problem.
- Determines which constraints should be enforced based on the active task configuration.
#### Steps:
1. Compute Number of Variables
   - Calculates total variables as:
     - `n_var = n_ports × n_streams`
2. Retrieve Task Configuration
   - Calls `self.get_task()`:
     - Returns two flags:
       - `fr` (frame routing)
       - `ftp` (frame/task placement)
3. Determine Constraint Flags
   - Sets `check_routing`:
     - Disabled only if both `fr` and `ftp` are enabled.
   - Sets task placement checking:
     - Enabled when `ftp` is False.
4. Initialize Evaluator
   - Creates a `TSNConstraintEvaluator` instance.
   - Passes:
     - The current scheduling object (`self`)
     - Routing constraint flag
     - Task placement constraint flag
### Output
- Returns: (None)
- Side effects:
   - Initializes `self.evaluator` with appropriate constraint checks.
   - Configures evaluation behavior based on task settings.

## 3. use_instance(self, instance)
```python
    def use_instance(self, instance):
        """Return a context manager that sets .usecase to instance inside a with-block."""
        return self.ContextHandler(self, instance)
```

### Parameters
- self (TSNScheduling) Instance of the scheduling class.
- instance (object) The usecase instance to activate within the context.

### What the Algorithm Does
- Provides a **helper method** to create a context manager for safely assigning a usecase.
- Delegates context handling to the configured `ContextHandler`.

#### Steps:
1. Receives a target instance.
2. Instantiates the configured ContextHandler with:
   - The current object (self)
   - The provided instance
3. Returns the context manager.

### Output
- Returns: (context manager) A context manager that sets self.usecase within a with block.
- Effects: None directly (effects occur when the context is entered).

## 4. no_instance_assigned(self)
```python
    def no_instance_assigned(self):
        raise ValueError("No current instance is set! Use a 'with problem.use_instance(instance):' block.")
```

### Parameters
- self (TSNScheduling) Instance of the scheduling class.

### What the Algorithm Does
- Enforces that a usecase must be active before certain operations can proceed.
- Raises an explicit error when no instance is assigned.

#### Steps:
1. Immediately raises a ValueError.
2. Provides a message indicating correct usage with a context manager.

### Output
- Returns: (None)
- Effects:
   - Raises `ValueError` when called.
   - Interrupts execution flow if no instance is set.

## 5. instantiate(self, export=False)
```python
def instantiate(self, export=False):
    usecase = self.generate_scheduling_usecase(self.config_file, export=export)
    self.instances.append(usecase)
    return usecase
```

### Parameters
- self (TSNScheduling) Instance of the scheduling system.
- export (bool, optional) If `True`, the generated usecase is exported to disk

### What the Algorithm Does
- Creates a single **TSN scheduling usecase** and stores it internally.
#### Steps:
1. Calls generate_scheduling_usecase() using self.config_file.
2. Appends the generated usecase to self.instances.
3. Returns the created usecase.

### Output
- Returns: (Usecase object) A newly generated scheduling usecase.
- Side effects:
   - Adds the usecase to self.instances.
   - Optionally writes the usecase to disk (if export is enabled).

## 6. import_scheduling_usecase(self, file_path, remove_routing=False, fix_task_placement=False)
```python
def import_scheduling_usecase(self, file_path, remove_routing=False, fix_task_placement=False):
    imp_uc = json.load(open(file_path))
    uc = Usecase.from_json(imp_uc)
    if remove_routing:
        uc.removeRouting()
    if fix_task_placement:
        uc.fixTalkers()
        uc.fixListeners()
    return uc
```

### Parameters
- file_path (str) Path to a JSON file containing a saved usecase.
- remove_routing (bool, optional) If `True`, removes routing information from the usecase.
- fix_task_placement (bool, optional) If `True`, fixes talker/listener assignments in the usecase.

### What the Algorithm Does
- Loads a TSN scheduling usecase from a JSON file.
- Optionally cleans or adjusts the imported data.
#### Steps:
1. Reads JSON file from file_path.
2. Converts JSON into a Usecase object using Usecase.from_json().
3. If remove_routing is enabled:
   - Calls uc.removeRouting().
4. If fix_task_placement is enabled:
   - Calls:
     - uc.fixTalkers()
     - uc.fixListeners()
5. Returns the processed usecase.

### Output
- Returns: (Usecase object) A reconstructed and optionally modified scheduling usecase.
- Effects:
   - Reads from file system.
   - May modify usecase structure depending on flags.

## 7. generate_scheduling_usecase(self, config_file, export=False, usecases_folder='./problem_instances/')
```python
def generate_scheduling_usecase(self, config_file, export=False, usecases_folder='./problem_instances/'):
    uc = self.uc_gen.generateUC(config_file)
    if export:
        uc.to_file(jsnPath=usecases_folder)
    return uc
```
### Parameters
- config_file (str / dict / object) Configuration used to generate the usecase.
- export (bool, optional) If True, saves generated usecase to disk.
- usecases_folder (str, optional) Directory where usecases are exported.
### What the Algorithm Does
- Generates a **single scheduling usecase** from a configuration file.
#### Steps:
1. Calls `self.uc_gen.generateUC(config_file)`.
2. If `export` is enabled:
   - Saves usecase using `uc.to_file()`.
3. Returns the generated usecase.
### Output
- Returns: (Usecase object) A newly generated TSN scheduling usecase.
- Effects: May write a file to disk if export is enabled.

## 8. generate_incremental_usecases(self, config_file, export=False, usecases_folder='./problem_instances/')
```python
def generate_incremental_usecases(self, config_file, export=False, usecases_folder='./problem_instances/'):
    ucs = list(self.uc_gen.ucGenerator(config_file))
    if export:
        for uc in ucs: 
            uc.to_file(jsnPath=usecases_folder)
    return ucs
```
### Parameters
- config_file (str / dict / object) Configuration used to generate multiple usecases.
- export (bool, optional) If True, exports all generated usecases.
- usecases_folder (str, optional) Directory where usecases are saved.
### What the Algorithm Does
- Generates **multiple incremental TSN usecases** from a single configuration.
#### Steps:
1. Calls generator `self.uc_gen.ucGenerator(config_file)`:
   - Produces an iterable of usecases.
2. Converts iterator into a list.
3. If `export` is enabled:
   - Iterates over all usecases.
   - Saves each one to disk using `to_file()`.
4. Returns the full list.
### Output
- Returns: (list of Usecase objects) A list of generated scheduling usecases.
- Effects:
   - May write multiple files to disk.
   - Generates multiple incremental problem instances.

## 9. get_config(self)
```python
@require_instance
def get_config(self):
    return self.usecase.get_config()
```
### Parameters
- self (TSNScheduling) Instance of the scheduling system with an active usecase.

### What the Algorithm Does
- Returns the configuration object of the currently active usecase.
- Protected by the `@require_instance` decorator (requires an active usecase).

#### Steps:
1. The decorator checks whether self.usecase is set.
   - If not, execution is blocked.
2. Calls self.usecase.get_config().
3. Returns the configuration data.

### Output
- Returns: (config object / dict-like structure) The configuration associated with the current usecase.
- Side effects:
   - None (read-only operation).
   - May raise error via decorator if no usecase is active.

## 10. get_task(self)
```python
def get_task(self):
    numplacesTalker = self.usecase.config.getint('streams_config', 'numplacesTalker')
    numplaceslistener = self.usecase.config.getint('streams_config', 'numplaceslistener')
    routing = ast.literal_eval(str(self.usecase.config.get('streams_config', 'routing')))
    return routing['fixPath'], numplacesTalker==numplaceslistener==1
```
### Parameters
- self (TSNScheduling) Instance containing an active usecase with configuration.
### What the Algorithm Does
- Extracts task-related configuration flags from the usecase config.
- Determines routing behavior and placement constraints.
#### Steps:
1. Read talker placement constraint
   - numplacesTalker is read from config:
     - 'streams_config' → 'numplacesTalker'
2. Read listener placement constraint
   - numplaceslistener is read similarly.
3. Parse routing configuration
   - Reads 'routing' field from config.
   - Converts string → Python object using:
     - ast.literal_eval(str(...))
4. Extract routing mode
   - Retrieves routing['fixPath'].
5. Determine placement constraint
   - Evaluates:
     - numplacesTalker == numplaceslistener == 1
   - Produces a boolean indicating strict 1-to-1 placement.
6. Return values
   - Returns a tuple:
     - (fixPath, strict_placement_flag)
### Output
- Returns: (tuple)
   - routing['fixPath'] → routing mode flag
   - numplacesTalker == numplaceslistener == 1 → boolean indicating strict placement constraint
- Effects
   - None (pure read operation).
   - Relies on valid configuration structure inside self.usecase.config.

## 11. n_ports(self)
```python
@property
@require_instance
def n_ports(self):
    return len(self.usecase.ports)
```

### Parameters
- self *(TSNScheduling)*  
  Instance with an active usecase.

### What the Algorithm Does
- Returns the number of ports in the current usecase.
- Requires an active usecase via @require_instance.

#### Steps:
1. Decorator ensures self.usecase is set.
2. Accesses self.usecase.ports.
3. Returns the length of the ports collection.


### Output
- Returns: int  
- Side effects: None


## 12. n_streams(self)
```python
@property
@require_instance
def n_streams(self):
    return len(self.usecase.streams)
```

### Parameters
- self *(TSNScheduling)*

### What the Algorithm Does
- Returns the number of streams in the current usecase.

### Output
- Returns: int


## 13. port_id_to_name(self)
```python
@property
@require_instance
def port_id_to_name(self):
    mapping = dict(enumerate(self.usecase.ports.keys(), start=1))
    mapping[0] = ''
    return mapping
```

### Parameters
- self *(TSNScheduling)*

### What the Algorithm Does
- Creates mapping from port IDs to names (1-based index).

### Output
- Returns: dict

## 14. port_name_to_id(self)
```python
@property
@require_instance
def port_name_to_id(self):
    mapping = {key: index for index, key in enumerate(self.usecase.ports.keys(), start=1)}
    mapping[''] = 0
    return mapping
```

### Parameters
- self *(TSNScheduling)*

### What the Algorithm Does
- Creates mapping from port names to 1-based IDs.
- Includes '' mapped to 0.

### Output
- Returns: dict


## 15. ports(self)
```python
@property
@require_instance
def ports(self):
    return {
        self.port_name_to_id[key]: {**value, 'destPort': self.port_name_to_id.get(value['destPort'], value['destPort'])}
        for key, value in self.usecase.ports.items()
    }
```

### Parameters
- self *(TSNScheduling)*

### What the Algorithm Does
- Returns ports keyed by port ID instead of name.
- Converts destPort names to IDs.

### Output
- Returns: dict


## 16. port_id_to_node(self)
```python
@property
@require_instance
def port_id_to_node(self):
    return {pId: self.ports[pId]['sourceNode'] for pId in self.ports}
```

### Parameters
- self *(TSNScheduling)*

### What the Algorithm Does
- Maps port IDs to their source nodes.

### Output
- Returns: dict


## 17. nodes(self)
```python
@property
@require_instance
def nodes(self):
    return {node[0]:(idx,node[1].get('procDelay', 0)) 
            for idx, node in enumerate(self.usecase.graph.nodes(data=True))}
```

### Parameters
- self *(TSNScheduling)*

### What the Algorithm Does
- Maps node names to (index, processing delay).

### Output
- Returns: dict

## 18. n_nodes(self)
```python
@property
@require_instance
def n_nodes(self):
    return len(self.usecase.graph.nodes)
```

### Parameters
- self *(TSNScheduling)*

### What the Algorithm Does
- Returns the number of nodes in the usecase graph.

#### Steps:
1. Ensures a usecase is active via decorator.
2. Accesses self.usecase.graph.nodes.
3. Returns the total count.

### Output
- Returns: int


## 19. node_to_port_id(self)
```python
@property
@require_instance
def node_to_port_id(self):
    egress_map = {}
    ingress_map = {}
    adjacent_egress_map = {}

    for src, dest, einfo in self.usecase.graph.edges(data=True):
        srcP = self.port_name_to_id[einfo.get('sourcePort')]
        destP = self.port_name_to_id[einfo.get('destPort')]

        egress_map.setdefault(src, []).append(srcP)
        ingress_map.setdefault(dest, []).append(destP)
        adjacent_egress_map.setdefault(dest, []).append(srcP)

    return egress_map, ingress_map, adjacent_egress_map
```

### Parameters
- self *(TSNScheduling)*

### What the Algorithm Does
- Builds mappings between nodes and their associated port IDs.

#### Steps:
1. Initializes three dictionaries:
   - egress_map
   - ingress_map
   - adjacent_egress_map
2. Iterates over graph edges with metadata.
3. Converts source and destination ports to IDs.
4. Updates:
   - egress_map: node → outgoing ports
   - ingress_map: node → incoming ports
   - adjacent_egress_map: node → neighboring outgoing ports
5. Returns all three mappings.

### Output
- Returns: tuple (egress_map, ingress_map, adjacent_egress_map)


## 20. next_prev_hops(self, hop)
```python
@property
@require_instance
def next_prev_hops(self, hop):
    hop_src = ports[hop]['sourceNode']
    hop_dest = ports[hop]['destNode']
    ports = self.ports

    next_hops = [pid for pid, info in ports.items() if info['sourceNode'] == hop_dest]
    prev_hops = [pid for pid, info in ports.items() if info['destNode'] == hop_src]
```

### Parameters
- self *(TSNScheduling)*
- hop *(int)*  
  The egress port ID representing the current hop.

### What the Algorithm Does
- Determines the next and previous hops based on the given port.

#### Steps:
1. Retrieves source and destination nodes for the given hop.
2. Iterates through all ports.
3. Finds:
   - next_hops: ports originating from the current destination node
   - prev_hops: ports ending at the current source node

### Output
- Returns: None (lists are computed but not returned)

- Effects:
  - None

## 21. macro_cycle(self)
```python
    @property
    @require_instance
    def macro_cycle(self):
        """LCM of stream intervals."""
        macro_cycle = 1
        for stream in self.usecase.streams:
            i = int(stream['interval'])
            macro_cycle = macro_cycle * i // math.gcd(macro_cycle, i)
        return macro_cycle
```
### Parameters
- self (TSNScheduling) The instance containing the use case and stream definitions.

### What the Algorithm Does
- Calculates the Macro Cycle of the TSN network, which is the Least Common Multiple (LCM) of all individual stream intervals.
- This represents the total time duration after which the entire network schedule repeats.

#### Steps:
1. Initializes macro_cycle to 1.
2. Iterates through every stream in the current use case.
3. Retrieves the interval attribute for each stream.
4. Updates the global macro_cycle by calculating the LCM between the current value and the new stream interval.

### Output
- Returns: int The calculated LCM of all stream intervals.

## 22. streams(self)
```python
    @property
    @require_instance
    def streams(self):
        """
        Return streams enriched with port ids where needed.
        This property will create a new list each call (shallow copies).
        """
        streams = []
        for s in self.usecase.streams:
            stream = s.copy()
            # replace talker/listener possible placements with 1-based port ids
            stream['tlkrPssblPlcmnt'] = [self.port_name_to_id[node + '.Port1'] for node in s['tlkrPssblPlcmnt']]
            stream['lstnsrPssblPlcmnt'] = [[self.port_name_to_id[node + '.Port1'] for node in nodes] for nodes in s['lstnsrPssblPlcmnt']]
            if 'pathLists' in stream:
                # convert hops to port ids but keep the same shape as original (omit last hop if that was a host owner)
                stream['pathLists'] = [[self.port_name_to_id[hop] for hop in path[:-1]] for path in s['pathLists']]
            streams.append(stream)
        return streams
```
### Parameters
- self (TSNScheduling) The instance containing stream data and the port_name_to_id mapping.

### What the Algorithm Does
- Enriches stream data by converting human-readable node and port names into numerical 1-based port IDs.
- Prepares stream paths for scheduling by trimming specific hops.

#### Steps:
1. Iterates through the list of raw streams.
2. Creates a shallow copy of each stream to avoid modifying the original data.
3. Maps Talker possible placement nodes to their corresponding port IDs.
4. Maps Listener possible placement nodes (nested lists) to their corresponding port IDs.
5. If pathLists exist:
   - Iterates through each path.
   - Converts every hop in the path to a port ID.
   - Omits the last hop in each path (typically the destination host).

### Output
- Returns: list[dict] A list of enriched stream dictionaries containing numerical port IDs instead of name strings.

## 23. links(self)
```python
    @property
    @require_instance
    def links(self):
        links = []
        for l in self.usecase.links:
            link = l.copy()
            link['sourcePort'] = self.port_name_to_id[l['sourcePort']]
            link['destPort'] = self.port_name_to_id[l['destPort']]
            links.append(link)
        return links
```
### Parameters
- self (TSNScheduling) The instance containing link data and the port_name_to_id mapping.

### What the Algorithm Does
- Standardizes link definitions by replacing port name strings with their internal port ID integers.
#### Steps:
1. Iterates through the collection of links in the use case.
2. Creates a copy of each link dictionary.
3. Performs a lookup in self.port_name_to_id for both the sourcePort and destPort.
4. Replaces the original string names with the retrieved numerical IDs.

### Output
- Returns: list[dict] 
A list of link dictionaries where sourcePort and destPort are represented by IDs.

## 24. _frame_size(cls, stream)
```python
    @classmethod
    def _frame_size(cls, stream):
        """
        Compute the frame size in bits *with tolerance*.
        """
        return (
            stream["maxFramesPerInterval"]
            * stream["maxFrameSize"]
            * BYTE_2_BIT
            * TOL
        )
```
### Parameters
- cls (class)
The class reference (method is a classmethod).

-stream (dict)
The stream dictionary containing frame configuration.

### What the Algorithm Does
- Calculates the total bandwidth requirement for a specific stream in bits, incorporating a safety margin (TOL).
- Converts the payload size from bytes to bits and accounts for multiple frames within a single interval.

#### Steps:
1. Multiplies the number of frames sent per interval by the maximum frame size.
2. Converts the byte-based size to bits using the BYTE_2_BIT constant.
3. Applies a tolerance multiplier (TOL) to provide overhead for headers or safety buffers.

### Output
- Returns: float The total calculated size of the stream's data per interval in bits.

## 25. hops_costs(self, frame_size)
```python
    @require_instance
    def hops_costs(self, frame_size):
        """
        Cost of forwarding a frame from one egress port.
        """
        hcost = np.zeros(self.n_ports)
        for pId, egport in self.ports.items():
            destNode = egport['destNode']
            sourceNode = egport['sourceNode']
            
            trans_delay = frame_size / (
                egport["speed"] * MBPS_2_BPNS
            )
            prop_delay = self.usecase.graph[sourceNode][destNode]["propDelay"]
            proc_delay = self.usecase.graph.nodes[destNode].get("procDelay",0)
    
            hcost[pId-1] = trans_delay + prop_delay + proc_delay
    
        return hcost
```
### Parameters
- self (TSNScheduling)The instance containing port information and network graph.
- frame_size (float/int)The size of the frame in bits (usually calculated by _frame_size).
### What the Algorithm Does
- Computes the latency cost associated with every egress port in the network for a frame of a specific size.
- Models the physical and logical delays encountered when moving data across a single hop.
#### Steps:
1. Initializes a zero-filled NumPy array for all ports.
2. Iterates through each port (egport) in the network:
   - Transmission Delay: Calculated as $\text{bits} / \text{link speed}$ (normalized to nanoseconds).
   - Propagation Delay: Extracted from the graph edge between source and destination.
   - Processing Delay: Extracted from the destination node's properties (defaults to 0).
3. Sums these three factors to determine the total hop cost.
4. Stores the cost in the array using 0-based indexing (pId-1).
### Output
- Returns: numpy.ndarray A vector where each index represents the cumulative delay (cost) of that specific port.

## 26. network_borders(self, graph)
```python
    @require_instance
    def network_borders(self, graph):
        """Return egress ports of switches that are connected to listeners (ports on switch->host edges)."""
        return [edge[2]['sourcePort'] for edge in graph.edges(data=True) if 'host' in edge[2]['destPort']]
```
### Parameters
- self (TSNScheduling) The instance containing the scheduling context.
- graph (NetworkX graph) The topology graph containing edge and port data.
### What the Algorithm Does
- Identifies the "edge" ports of the network fabric—specifically, switch ports that connect directly to end-host listeners.
- Filters the network topology to find points where data leaves the infrastructure.

#### Steps:
1. Iterates through all edges of the provided graph, including data attributes.
2. Checks the metadata of the destPort for each edge.
3. Filters for edges where the destination port belongs to a 'host' device.
4. Collects and returns the sourcePort ID for these specific edges.

### Output
- Returns: list[int] A list of port IDs representing the egress boundary of the switches.

## 27. configure_stream_ports(self, ports_in, ports_out)
```python
    @require_instance
    def configure_stream_ports(self, ports_in, ports_out):
        """
        After instantiating the problem, the streams can be independently configured using this function to force using some ports and excluding some other ports.
                                        ------------------------------     ------------
                                        | customGraph | config_column|      seedGraph
                             -----------------------------------------     ------------
                            always on   |    exist    |       1      |        exist
                             -----------------------------------------     ------------
                            always off/ |  removed    |              |       exist in        
                            concurent of|  from the   |       0      |      some seeds            
                            an always on|   graph     |              |      but not in            
                             -----------------------------------------      some others
                            free to use |    exist    |       0      |        seeds
                            ------------------------------------------     ------------ 
                            
        Build config_column and a pruned custom_graph for the stream based on ports_in / ports_out.
        - ports_in: dict with keys 'talker','listener','switch' listing egress-port names to force active
        - ports_out: dict with keys 'talker','listener','switch' listing ports to force inactive
        Returns (config_column (np.array len n_ports), custom_graph (networkx DiGraph)).
        """
        # ----- 1) always-active ports (talker/switch are egress ports; listener requires the incoming sourcePort)
        always_active_ports = []
        for typ, plist in ports_in.items():
            if typ != 'listener':
                always_active_ports.extend(plist)
            else:
                # for each listener egress-port name we need its incoming sourcePort in the global usecase graph
                for listener_port in plist:
                    node = listener_port.split('.')[0]
                    # take first incoming edge (should exist) and get its sourcePort
                    in_edges = list(self.usecase.graph.in_edges(node, data=True))
                    if in_edges:
                        src_port = in_edges[0][2]['sourcePort']
                        always_active_ports.append(src_port)
                    else:
                        # no incoming edge -> skip (defensive)
                        continue

        # map to 0-based indices for config matrix
        always_active_ids = [self.port_name_to_id[p] - 1 for p in always_active_ports if p in self.port_name_to_id]
        config_column = np.zeros(self.n_ports, dtype=int)
        if always_active_ids:
            config_column[always_active_ids] = 1

        # ----- 2) determine edges to remove from the global graph
        edges_to_remove = []

        # concurrent ports for forced-ins (these are edges to remove because they are "concurrent" with forced-ins)
        for typ, egports in ports_in.items():
            edges_to_remove.extend(self.get_concurrent_ports(egports, typ=typ))

        # ports explicitly forced out: remove edges touching those ports
        ports_out_listener = set(ports_out.get('listener', []))
        ports_out_talker = set(ports_out.get('talker', []))
        ports_out_switch = set(ports_out.get('switch', []))

        for u, v, data in self.usecase.graph.edges(data=True):
            if data.get('destPort') in ports_out_listener or data.get('sourcePort') in ports_out_talker or data.get('sourcePort') in ports_out_switch:
                edges_to_remove.append((u, v, data))

        # ensure we do not remove edges that are needed for always-active ports
        filtered_edges_to_remove = []
        for edge in edges_to_remove:
            src_port = edge[2].get('sourcePort')
            if src_port is None:
                continue
            src_id = self.port_name_to_id.get(src_port, None)
            if src_id is None:
                continue
            if (src_id - 1) in always_active_ids:
                # keep this edge (do not remove)
                continue
            filtered_edges_to_remove.append(edge)

        # ----- 3) build the custom graph (edge-subgraph then copy + prune isolated nodes)
        # build a set of edge keys (u,v) to keep
        keep_edges = [(u, v) for u, v, data in self.usecase.graph.edges(data=True) if (u, v, data) not in filtered_edges_to_remove]
        custom_graph = self.usecase.graph.edge_subgraph(keep_edges).copy()

        # prune isolated nodes
        isolated_nodes = [n for n, d in custom_graph.degree() if d < 1]
        if isolated_nodes:
            custom_graph.remove_nodes_from(isolated_nodes)

        return config_column, custom_graph
```

### Parameters
- self (TSNScheduling) The instance containing the use case graph and port mappings.
- ports_in (dict) A dictionary containing lists of port names to force Active (keys: 'talker', 'listener', 'switch').
- ports_out (dict) A dictionary containing lists of port names to force Inactive (keys: 'talker', 'listener', 'switch').

### What the Algorithm Does
- Customizes the network topology and configuration for a specific stream by "locking" certain ports as always-on or always-off.
- Prunes the network graph to ensure paths only use allowed infrastructure.

#### Steps:
1. Identify Forced-On Ports:
   - Direct mapping for talkers and switches.
   - For listeners, it identifies the incoming source port connected to the listener node.
2. Generate Config Mask: Creates a binary vector (config_column) where indices of forced-on ports are set to 1.
3. Identify Restricted Edges:
   - Collects "concurrent" ports that must be disabled to support the forced-in selections.
   - Adds ports explicitly listed in ports_out.
4. Safety Filter: Ensures that edges required for "always-active" ports are not removed from the restricted list.
5. Graph Pruning:
   - Generates a subgraph excluding the restricted edges.
   - Removes isolated nodes (nodes with no connections) to clean up the topology.

### Output
- Returns: (numpy.ndarray, networkx.DiGraph)
   - config_column: A 0/1 array representing forced-port constraints.
   - custom_graph: A specialized version of the network graph with restricted paths removed.

## 28. get_concurrent_ports(self, egports, typ=None)
```python
    @require_instance
    def get_concurrent_ports(self, egports, typ=None):
        """
        Return a list of edges (u,v,data) that are "concurrent" with the given egress ports.
        - typ in {'switch','talker','listener'}.
        """
        if not egports:
            return []

        nodes_names = [p.split('.')[0] for p in egports]

        if typ == 'switch':
            # other egress ports of the same switch (exclude the provided ones)
            return [
                edge for node in nodes_names
                for edge in self.usecase.graph.out_edges(node, data=True)
                if edge[2]['sourcePort'] not in egports
            ]
        elif typ == 'talker':
            # all egress ports of other hosts
            return [
                edge for node in self.usecase.graph.nodes
                if node not in nodes_names and node.startswith('host')
                for edge in self.usecase.graph.out_edges(node, data=True)
            ]
        elif typ == 'listener':
            # all ingress ports of other hosts
            return [
                edge for node in self.usecase.graph.nodes
                if node not in nodes_names and node.startswith('host')
                for edge in self.usecase.graph.in_edges(node, data=True)
            ]
        else:
            raise ValueError("check arguments of get_concurrent_ports(egports, typ)!!")
```
### Parameters
- self (TSNScheduling) The instance containing the network graph and use case data.
- egports (list[str]) A list of egress port names (e.g., ['Switch1.Port1', 'Switch1.Port2']).
- typ (str, optional) The category of the ports being processed. Must be one of 'switch', 'talker', or 'listener'.

### What the Algorithm Does
- Identifies "concurrent" (or competing) edges in the network based on a set of selected ports.
- In the context of TSN scheduling, this is often used to find edges that should be disabled or considered "interference" when a specific path is being forced or prioritized.

#### Steps:
1. Node Extraction: Splits the provided port strings to get the base node names (e.g., "Switch1.Port1" becomes "Switch1").
2. Type-Based Filtering:
   - Switch: Finds all other egress ports on the same switches that were not explicitly included in the egports list.
   - Talker: Finds all egress ports belonging to every other host in the network except the ones associated with the current egports.
   - Listener: Finds all ingress ports belonging to every other host in the network except the ones associated with the current egports.
3. Validation: Raises a ValueError if the provided typ is unrecognized, ensuring data integrity for the RAG system.

### Output
- Returns: list[tuple] A list of NetworkX edge tuples (u, v, data) representing the concurrent edges found.
- Effects: None. (This is a read-only query against the graph).

## 29. make_default_stream_config(self, stream)
```python
# default stream config builder (wrapper)
    @require_instance
    def make_default_stream_config(self, stream):
        """
        [generate a subgraph by removing all deactivated nodes and edges (i.e. that are not allowed to be used)]
        Build default (config_column, custom_graph) for a stream dict.
        stream is a dict with keys like 'tlkrPssblPlcmnt', 'lstnsrPssblPlcmnt', optionally 'pathLists'.
        """
        ports_in = {
            'talker': [stream['tlkrPssblPlcmnt'][0] + '.Port1'] if len(stream['tlkrPssblPlcmnt'])==1 else [],      # by default, talkers are free to use unless fixed
            'listener': [listener[0] for listener in stream['lstnsrPssblPlcmnt'] if len(listener) == 1],    # listeners free to use unless fixed
        }

        ports_out = {
            'talker': [host + '.Port1' for host in self.usecase.hosts if host not in stream['tlkrPssblPlcmnt']],
            'listener': [host + '.Port1' for host in self.usecase.hosts if host not in [h for listener in stream['lstnsrPssblPlcmnt'] for h in listener]],
            'switch': []
        }

        if 'pathLists' in stream:
            # all internal switch egress ports in provided paths are forced-in
            ports_in['switch'] = [egressport for path in stream['pathLists'] for egressport in path[1:-1]]
            # switches not in the path are out
            ports_out['switch'] = [port for port in self.usecase.ports if (port not in ports_in['switch']) and ('switch' in port)]

        return self.configure_stream_ports(ports_in, ports_out)
```
### Parameters
- self (TSNScheduling) 
- stream (dict) A stream definition containing placement constraints (tlkrPssblPlcmnt, lstnsrPssblPlcmnt) and optional predefined paths (pathLists).

### What the Algorithm Does
- Acts as a logic-heavy wrapper to prepare inclusion/exclusion lists for stream-specific network pruning.
- Automatically identifies which ports must be active and which must be disabled based on the stream's requirements.

#### Steps:
1. Define Forced-In Ports:
   - Adds talker or listener ports only if they have exactly one possible placement (i.e., they are "fixed").
2. Define Forced-Out Ports:
   - Identifies all host ports that are not part of the stream's possible placements and marks them for removal.
3. Handle Static Routing:
   - If the stream includes pathLists, it forces all internal switch egress ports in those paths to be active.
   - Conversely, it marks all other switch ports in the network as inactive for this stream.
4. Finalize: Passes these curated lists to configure_stream_ports.

### Output
- Returns: (numpy.ndarray, networkx.DiGraph) The configuration mask and pruned graph from configure_stream_ports.

## 30. make_stream_config(self, stream, always_in, always_out, overwrite=False, as_idx=True)
```python
# default stream config builder (wrapper)
    @require_instance
    def make_stream_config(self, stream, always_in, always_out, overwrite=False, as_idx=True):
        """
        Build (config_column, custom_graph) from explicit always_in / always_out lists.
        - always_in/out are either indices (0-based) or port names depending on as_idx flag.
        """
        if as_idx:
            # convert 0-based indices to 1-based id->name then to names
            always_in = [self.port_id_to_name[pidx + 1] for pidx in always_in]
            always_out = [self.port_id_to_name[pidx + 1] for pidx in always_out]

        ports_in = {
            'talker': [port for port in always_in if 'host' in port],
            'listener': [],
            'switch': [port for port in always_in if 'switch' in port and port not in self.network_borders(self.usecase.graph)]
        }

        ports_out = {
            'talker': [port for port in always_out if 'host' in port],
            'listener': [],
            'switch': [port for port in always_out if 'switch' in port]
        }

        if not overwrite:
            # respect original talker/listener placements from input usecase
            original_out_talkers = [host + '.Port1' for host in self.usecase.hosts if host not in stream['tlkrPssblPlcmnt']]
            ports_out['talker'] += original_out_talkers
            ports_out['listener'] += [host + '.Port1' for host in self.usecase.hosts if host not in [h for listener in stream['lstnsrPssblPlcmnt'] for h in listener]]
            ports_in['talker'] = [p for p in ports_in['talker'] if p not in original_out_talkers]

        return self.configure_stream_ports(ports_in, ports_out)
```
### Parameters
- self (TSNScheduling)
- stream (dict) The stream data dictionary.
- always_in (list) Ports to force active.
- always_out (list) Ports to force inactive.
- overwrite (bool, default=False) If False, merges the explicit lists with the original stream placement constraints.
- as_idx (bool, default=True) Determines if always_in/out are provided as 0-based indices or string names.

### What the Algorithm Does
- Provides a flexible way to manually override or augment a stream's network configuration using either numerical indices or port names.

#### Steps:
1. ID Resolution: Converts 0-based port indices to port name strings if as_idx is enabled.
2. Category Mapping: Sorts ports into talker and switch buckets. It specifically excludes switch ports that are on the "network border" from being forced into the switch category.
3. Constraint Merging:
   - If overwrite is disabled, it looks at the stream dictionary to identify ports that were originally disallowed.
   - It appends these original restrictions to the ports_out list and cleans up the ports_in list to prevent conflicts.
4. Execution: Hands off the final requirement dictionaries to configure_stream_ports.

### Output
- Returns: (numpy.ndarray, networkx.DiGraph) The resulting configuration vector and the customized topology graph.

## 31. hop_cost_eg2eg(self, cur_egp, next_egp, frame_size)

```python
    @require_instance
    def hop_cost_eg2eg(self, cur_egp, next_egp, frame_size):
        """
        Cost of forwarding a frame from one egress port to the egress port at next hop.

        Parameters
        ----------
        cur_egp : str
            Current egress port name.
        next_egp : str
            Next egress port name.
        frame_size : float
            Frame size in bits (with tolerance applied).

        Returns
        -------
        float
            Delay in nanoseconds.
        """
        trans_delay = frame_size / (
            self.usecase.ports[cur_egp]["speed"] * MBPS_2_BPNS
        )
        next_node = next_egp.split(".")[0]
        prop_delay = self.usecase.graph[cur_egp.split(".")[0]][next_node]["propDelay"]
        proc_delay = self.usecase.graph.nodes[next_node]["procDelay"]

        return trans_delay + prop_delay + proc_delay
```

### Parameters

* self *(TSNScheduling)*
* cur_egp *(str)*
  Current egress port name.
* next_egp *(str)*
  Next hop egress port name.
* frame_size *(float)*
  Frame size in bits.

### What the Algorithm Does

* Computes the **delay of forwarding a frame** between two egress ports.

#### Steps:

1. Compute transmission delay using port speed.
2. Extract next node from `next_egp`.
3. Read propagation delay from graph edge.
4. Read processing delay from node attributes.
5. Sum all delays.

### Output

* **Returns:** *(float)*
  Total delay in nanoseconds

## 32. bfs_forward(self, graph, start_port, stream)

```python
    @require_instance
    def bfs_forward(self, graph, start_port, stream):
        """
        Forward BFS from a talker port to compute minimum delay budgets.

        Parameters
        ----------
        graph : networkx.DiGraph
            The usecase graph.
        start_port : str
            Egress port name of the talker host.
        stream : dict
            Stream definition.

        Returns
        -------
        dict
            Mapping of next-egress-port -> {current-egress-port: min_delay}
        """
        frame_size = TSNScheduling._frame_size(stream)

        queue = deque([(start_port, 0)])  # (current egress port, cumulative delay)
        visited = set()
        ss_min = {}

        while queue:
            cep, delay = queue.popleft()

            for _, next_node, data in graph.out_edges(
                self.usecase.ports[cep]["destNode"], data=True
            ):
                nip = data["destPort"]
                nep = data["sourcePort"]

                if nip == cep:  # avoid going back
                    continue

                if (cep, nep) not in visited:
                    visited.add((cep, nep))
                    new_delay = delay + self.hop_cost_eg2eg(cep, nep, frame_size)

                    if nep not in ss_min:
                        ss_min[nep] = {}
                    ss_min[nep][cep] = new_delay

                    queue.append((nep, new_delay))

        return ss_min
```

### Parameters

* graph *(networkx.DiGraph)*
* start_port *(str)*
  Talker egress port
* stream *(dict)*

### What the Algorithm Does

* Performs **forward BFS traversal** to compute minimum delay budgets.

#### Steps:

1. Initialize queue with start port.
2. Traverse graph using BFS.
3. For each hop:

   * Compute delay using `hop_cost_eg2eg`.
4. Store minimum delay per transition.
5. Continue traversal until all reachable ports are processed.

### Output

* **Returns:** *(dict)*
  Mapping: `next_port → {current_port: min_delay}`

## 33. bfs_backward(self, graph, end_port, stream)

```python
    @require_instance
    def bfs_backward(self, graph, end_port, stream):
        """
        Backward BFS from a listener ingress port to compute maximum
        remaining delay budgets.

        Parameters
        ----------
        graph : networkx.DiGraph
            The usecase graph.
        end_port : str
            Ingress port name of the listener host.
        stream : dict
            Stream definition.

        Returns
        -------
        dict
            Mapping of current-egress-port -> {previous-egress-port: max_delay}
        """
        ml_ns = stream["maxLatency"]
        frame_size = TSNScheduling._frame_size(stream)

        queue = deque([(end_port, ml_ns)])  # (current egress port, remaining delay)
        visited = set()
        ss_max = {}

        while queue:
            cep, delay = queue.popleft()

            for _, _, data in graph.in_edges(cep.split(".")[0], data=True):
                nip = data["destPort"]
                nep = data["sourcePort"]

                if nip == cep:  # avoid going back
                    continue

                if (cep, nep) not in visited:
                    visited.add((cep, nep))
                    new_delay = delay - self.hop_cost_eg2eg(nep, cep, frame_size)

                    if cep not in ss_max:
                        ss_max[cep] = {}
                    ss_max[cep][nep] = new_delay

                    queue.append((nep, new_delay))

        return ss_max
```

### Parameters

* graph *(networkx.DiGraph)*
* end_port *(str)*
  Listener ingress port
* stream *(dict)*

### What the Algorithm Does

* Performs **backward BFS traversal** to compute maximum remaining delay budgets.

#### Steps:

1. Initialize queue with end port and max latency.
2. Traverse graph in reverse using incoming edges.
3. For each hop:

   * Subtract hop delay using `hop_cost_eg2eg`.
4. Store remaining delay budget.
5. Continue until all reachable paths are explored.

### Output

* **Returns:** *(dict)*
  Mapping: `current_port → {previous_port: max_delay}`

## 34. drawFattree(graph, ax=None, node_size=300, font_size=12, **kwds)

```python    @staticmethod
    def drawFattree(graph, ax=None, node_size=300, font_size=12, **kwds):
        """
        Draw a fat-tree style layout for the given graph.
    
        Parameters
        ----------
        graph : networkx.Graph
            The graph to draw.
        ax : matplotlib.axes.Axes or None
            Axis to draw into. If None, a new figure is created.
        node_size : int
            Size of nodes.
        font_size : int
            Size of labels.
        kwds : dict
            Extra keyword arguments passed to nx.draw.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(20, 8), dpi=60)
    
        positions = {}
        layers = {'core': 3, 'aggregation': 2, 'edge': 1, 'end': 0}
        xx = {'core': 115, 'aggregation': 75, 'edge': 75, 'end': 0}
    
        for node in graph.nodes():
            if '-c' in node:
                layer = 'core'
            elif '-a' in node:
                layer = 'aggregation'
            elif '-e' in node:
                layer = 'edge'
            else:
                layer = 'end'
            graph.nodes[node]['layer'] = layer
            y = layers[layer]
            x = xx[layer]
            xx[layer] += 20
            positions[node] = (x, y)
    
        pos = nx.multipartite_layout(graph, subset_key='layer', align='horizontal')
        nx.draw(graph, pos=positions, ax=ax, node_size=node_size, font_size=font_size, **kwds)
    
        return ax
```

### Parameters

* graph *(networkx.Graph)*
  Graph to visualize.

* ax *(matplotlib.axes.Axes, optional)*
  Axis to draw on. If None, a new figure is created.

* node_size *(int, optional)*
  Size of nodes.

* font_size *(int, optional)*
  Size of node labels.

* **kwds *(optional)*
  Additional keyword arguments for `nx.draw()`.

### What the Algorithm Does

* Draws a graph using a **fat-tree layered layout**.

#### Steps:

1. Creates a matplotlib axis if none is provided.
2. Defines layer hierarchy:

   * core, aggregation, edge, end.
3. Iterates over nodes:

   * Infers layer from node name.
   * Assigns (x, y) positions.
   * Stores layer in node attributes.
4. Uses multipartite layout for alignment.
5. Draws graph with custom positions.

### Output

* **Returns:** *(matplotlib.axes.Axes)*
  Axis containing the plotted graph

* **Side effects:**

  * Modifies node attributes (`layer`)
  * Draws visualization

## 35. plot_all_stream_custom_graphs(customGraphs, ncols=3, figsize=(18, 12), **draw_kwds)

```python
    @staticmethod
    def plot_all_stream_custom_graphs(customGraphs, ncols=3, figsize=(18, 12), **draw_kwds):
        """
        Plot all stream custom graphs side by side using the fat-tree layout.
    
        Parameters
        ----------
        usecase : object
            Must have `.streams`, where each stream is a dict with "customGraph".
        ncols : int
            Number of subplot columns.
        figsize : tuple
            Overall figure size.
        draw_kwds : dict
            Extra args passed to drawFattree (e.g. node_color, edge_color).
        """
        n_graphs = len(customGraphs)
        nrows = (n_graphs + ncols - 1) // ncols
    
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()
    
        for i, graph in enumerate(customGraphs):
            ax = axes[i]
            drawFattree(graph, ax=ax, **draw_kwds)
            ax.axis("off")
    
        # Hide any extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
    
        plt.tight_layout()
        return fig
```

### Parameters

* customGraphs *(list)*
  List of graphs to visualize.

* ncols *(int, optional)*
  Number of columns in subplot grid.

* figsize *(tuple, optional)*
  Size of the full figure.

* **draw_kwds *(optional)*
  Additional arguments passed to `drawFattree`.

### What the Algorithm Does

* Plots multiple graphs in a **grid layout** using fat-tree visualization.

#### Steps:

1. Computes number of rows based on number of graphs.
2. Creates subplot grid.
3. Iterates over graphs:

   * Draws each using `drawFattree`.
   * Disables axis display.
4. Hides unused subplots.
5. Adjusts layout for spacing.

### Output

* **Returns:** *(matplotlib.figure.Figure)*
  Figure containing all plotted graphs

* **Side effects:**

  * Displays multiple graph visualizations

# TSNConstraintEvaluator(problem, weights=None, check_routing=True, check_task_placement=True, scheduling_type=ScheduleEncodingType.PRIORITY)

```python
from enum import Enum
class ScheduleEncodingType(Enum):
    PRIORITY = 1
    START_TIME = 2
    #QUEUING_DELAY = 3

from scipy.sparse import csr_matrix
class TSNConstraintEvaluator:
    def __init__(self, problem, weights=None, check_routing=True, check_task_placement=True, 
                 scheduling_type: ScheduleEncodingType = ScheduleEncodingType.PRIORITY):
        # Default penalty weights
        default_weights = {
            # task placement
            "talker_not_fixed": 1,
            "listener_not_fixed": 1,
            "stray_talker": 1,
            "stray_listener": 1,
            "talker_listener_overlap": 1,
            # routing
            "backwarding": 1,
            "flow_conservation": 1,
            "looping": 1,
            "disconnected_path": 1,
            # scheduling
            "priority_conflicts": 1,
            "maxlatency_violations": 1,
            "stream_window_sequence":1,
            "interleaving":1,
            "macrocycle_overlap":1,
            "talker_offset":1
        }

        # Merge with user-supplied weights
        merged_weights = {**default_weights, **(weights or {})}

        # Determine which violation functions are active
        self.scheduling_type = scheduling_type
        self.active_checks = []
        relevant_weight_keys = set()

        if check_task_placement:
            self.active_checks.append(self._per_stream_host_role_violations)
            relevant_weight_keys.update([
                "talker_not_fixed",
                "listener_not_fixed",
                "stray_talker",
                "stray_listener",
                "talker_listener_overlap"
            ])

        if check_routing:
            self.active_checks.extend([
                self._per_stream_flow_conservation_violations,
                self._per_stream_flow_continuity_violations
            ])
            relevant_weight_keys.update([
                "backwarding",
                "flow_conservation",
                "looping",
                "disconnected_path"
            ])

        # Always check scheduling
        if self.scheduling_type.value == ScheduleEncodingType.PRIORITY.value:
            self.active_checks.extend([
                self._per_stream_priority_conflicts,
                self._per_stream_latency_violations_priority
            ])
            relevant_weight_keys.update([
                "maxlatency_violations",
                "priority_conflicts"
            ])
        if self.scheduling_type.value == ScheduleEncodingType.START_TIME.value:
            self.active_checks.extend([
                self._per_stream_window_sequence_violations,
                self._per_stream_macrocycle_overlap_violations,
                self._per_stream_talker_offset_violations,
                self._per_stream_latency_violations_start_time
            ])
            relevant_weight_keys.update([
                "maxlatency_violations",
                "stream_window_sequence",
                "interleaving",
                "macrocycle_overlap",
                "talker_offset"
            ])

        # Keep only relevant weights
        self.weights = {k: merged_weights[k] for k in relevant_weight_keys}
        self.n_constr = len(relevant_weight_keys)
        self.violations_keys = relevant_weight_keys
        
        #############################################################
        self._violation_evaluation_cache = {}
        self._metrics_evaluation_cache = {}
        ##############################################################
        
        n_nodes = problem.n_nodes
        n_ports = problem.n_ports
        n_streams = problem.n_streams

        self.n_streams =n_streams
        self.n_ports = n_ports

        self.macro_cycle = getattr(problem.usecase, "macro_cycle", 0)  # fallback if missing
        ports = problem.ports                        # dict keyed by port id (1-based)
        streams = problem.streams
        self.hosts = getattr(problem.usecase, "hosts", [])
        self.nodes = problem.nodes

        self.port_id_to_node = problem.port_id_to_node
        self.egress_map, self.ingress_map, self.adjacent_map = problem.node_to_port_id 

        ############### precompute ports/streams data ################
        self.port_speeds = np.zeros(n_ports, dtype=float)
        for p_idx in range(n_ports):
            pid = p_idx + 1
            pdata = ports[pid]
            self.port_speeds[p_idx] = pdata["speed"]
            
        self.stream_maxLatencies = [] 
        stream_intervals = []
        self.windLengths = np.zeros((n_ports, n_streams), dtype=float)
        self.all_hops_costs = np.zeros((n_ports, n_streams))
        stream_potential = []
        eps = 1e-9
        for s_idx, stream in enumerate(streams):
            stream_potential.append(len(stream["customGraph"].edges())) # total potential
            self.stream_maxLatencies.append(stream["maxLatency"])
            stream_intervals.append(stream["interval"])
            
            # precompute hops costs for latency evaluation
            frame_size = problem.usecase._frame_size(s_idx)
            self.all_hops_costs[:, s_idx] = problem.hops_costs(frame_size)
            # precompute windLengths
            for p_idx in range(n_ports):
                self.windLengths[p_idx, s_idx] = (frame_size / max(self.port_speeds[p_idx] * MBPS_2_BPNS, eps))

        self.stream_potential = stream_potential
        self.stream_intervals = np.asarray(stream_intervals, dtype=int)
        streams_pairwise_gcl = np.gcd(self.stream_intervals[:, None], self.stream_intervals[None, :])
        streams_pairwise_gcl[streams_pairwise_gcl == 0] = 1   # safety
        self.streams_pairwise_gcl = streams_pairwise_gcl # (S,S)
        self.streams_pairwise_gcl_inv = 1.0 / self.streams_pairwise_gcl[None, None, :, :]

        stream_pairwise_windowSum = (self.windLengths[:, :, None] + self.windLengths[:, None, :]) # (P,S,S)
        self.stream_pairwise_windowSum = stream_pairwise_windowSum
        # used to filter self-comparison for stream overlap
        self.pair_mask = ~np.eye(n_streams, dtype=bool)

        ################# nodes masks #####################
        node_index = {name: nid for name, (nid, _) in self.nodes.items()}
        
        node_egress_mask  = np.zeros((n_nodes, n_ports), dtype=int)
        node_ingress_mask  = np.zeros((n_nodes, n_ports), dtype=int)
        egress_to_node_mask  = np.zeros((n_nodes, n_ports), dtype=int)
        
        for name, n_idx in node_index.items():
            node_egress_mask[n_idx, np.array(self.egress_map[name])-1]=1
            node_ingress_mask[n_idx, np.array(self.ingress_map[name])-1]=1
            egress_to_node_mask[n_idx, np.array(self.adjacent_map[name])-1]=1
        
        host_mask = np.zeros((n_nodes), dtype=int)
        for host in self.hosts:
            host_mask[node_index[host]] = 1

        self.host_mask = host_mask
        self.switch_mask = 1-host_mask      

        self.egress_mask = node_egress_mask
        self.ingress_mask = node_ingress_mask
        self.to_node_mask = egress_to_node_mask

        ################# ports masks #####################
        adjacency_mask = np.zeros((n_ports, n_ports), dtype=int)
        next_hop_matrix = np.zeros((n_ports, n_ports), dtype=int)
        
        next_hop_list = [[] for _ in range(n_ports)]
        
        
        for pid in range(1, n_ports + 1):
            p_idx = pid - 1
            pdata = ports[pid]
            dst_node = pdata["destNode"]
            dst_port = pdata["destPort"]

            adjacency_mask[pid - 1, dst_port-1] = 1  
            
            next_ports = self.egress_map.get(dst_node, []) if dst_node not in self.hosts else []
            next_hop_list[p_idx] = list(dict.fromkeys(next_ports))

            next_eg_indices = [p-1 for p in next_ports]
            next_hop_matrix[p_idx, next_eg_indices] = 1
        
        self.adjacency_mask = adjacency_mask #
        self.next_hop_list = next_hop_list
        next_hop_matrix = csr_matrix(next_hop_matrix)

        self.next_hop_matrix = next_hop_matrix
        ################ CSR connectivity ################
        self._upstream_ports = np.repeat(np.arange(n_ports), np.diff(next_hop_matrix.indptr))
        self._downstream_ports = next_hop_matrix.indices
        
        ################# streams masks #####################
        talker_mask        = np.zeros((n_ports, n_streams), dtype=int)
        listener_mask      = np.zeros((n_ports, n_streams), dtype=int)
        listener_adj_mask  = np.zeros((n_ports, n_streams), dtype=int)
        
        self.stream_talker_ports = [] ##########
        self.stream_listener_adj_ports = [] ########## 
        
        for s_idx, stream in enumerate(problem.usecase.streams):
            
            # 1) Talker ports (1-based)
            t_nodes = stream.get("tlkrPssblPlcmnt", [])
            tps = []
            for t_node in t_nodes:
                t_ports = self.egress_map[t_node]
                tps +=t_ports
                talker_mask[np.array(t_ports) - 1, s_idx] = 1
            self.stream_talker_ports.append(np.array(tps, dtype=int) if tps else np.array([], dtype=int))

            # 2) Listener ports (1-based, stored as list of lists)
            lst_lists = stream.get("lstnsrPssblPlcmnt", [])
            lst_nodes = [l for sub in lst_lists for l in sub]
            lps = []
            for lst_node in lst_nodes:
                lst_ports = self.ingress_map[lst_node]
                listener_mask[np.array(lst_ports) - 1, s_idx] = 1
                
                lst_adj_ports = self.adjacent_map[lst_node]
                lps += lst_adj_ports
                listener_adj_mask[np.array(lst_adj_ports) - 1, s_idx] = 1
            self.stream_listener_adj_ports.append(np.array(lps, dtype=int) if lps else np.array([], dtype=int))

        # Save masks
        self.talker_mask = talker_mask # ports that can be talker
        self.listener_mask = listener_mask # ports that can be listener
        self.listener_adj_mask = listener_adj_mask # ports that reach to listener

        hosts_port_mask = node_egress_mask[host_mask==1].sum(0)
        hosts_adj_port_mask = egress_to_node_mask[host_mask==1].sum(0)
        self.stray_talker_mask = (1 - talker_mask) * hosts_port_mask[:,None]
        self.stray_listener_mask = (1 - listener_adj_mask) * hosts_adj_port_mask[:,None]
```

### Parameters

* problem *(TSNScheduling)*
  The scheduling problem instance containing topology, streams, and metadata.

* weights *(dict, optional)*
  Custom penalty weights for constraint violations.

* check_routing *(bool, optional)*
  Enable/disable routing-related constraints.

* check_task_placement *(bool, optional)*
  Enable/disable task placement constraints.

* scheduling_type *(ScheduleEncodingType, optional)*
  Defines scheduling model:

  * PRIORITY
  * START_TIME

### What the Algorithm Does

* Initializes a **constraint evaluation engine** for TSN scheduling.
* Selects active constraint checks.
* Precomputes **all required matrices, masks, and cost structures** for fast evaluation.

### Steps:

#### 1. Initialize Default Weights

* Defines penalties for:

  * Task placement violations
  * Routing violations
  * Scheduling violations

#### 2. Merge User Weights

* Overrides defaults with user-provided values.

#### 3. Select Active Constraint Checks

* Based on flags:

  * Task placement → host role checks
  * Routing → flow conservation & continuity
  * Scheduling:

    * PRIORITY → priority conflicts + latency
    * START_TIME → window, overlap, offset, latency

#### 4. Filter Relevant Weights

* Keeps only weights related to active checks.
* Stores:

  * `self.weights`
  * `self.n_constr`
  * `self.violations_keys`

#### 5. Initialize Caches

* `_violation_evaluation_cache`
* `_metrics_evaluation_cache`

#### 6. Extract Problem Dimensions

* Number of nodes, ports, streams
* Store topology-related data:

  * ports
  * streams
  * nodes
  * host list

#### 7. Precompute Port Data

* Build `port_speeds` array.

#### 8. Precompute Stream Data

* For each stream:

  * Store max latency
  * Store interval
  * Compute:

    * hop costs
    * transmission window lengths
* Compute:

  * pairwise GCD matrix
  * inverse GCD matrix
  * pairwise window sums
  * stream potential

#### 9. Build Node-Level Masks

* `egress_mask`: ports leaving nodes
* `ingress_mask`: ports entering nodes
* `to_node_mask`: adjacency mapping
* `host_mask` / `switch_mask`

#### 10. Build Port Connectivity Structures

* `adjacency_mask`: direct port-to-port connections
* `next_hop_list`: adjacency lists
* `next_hop_matrix`: sparse representation
* CSR upstream/downstream arrays

#### 11. Build Stream Masks

* `talker_mask`: valid talker ports
* `listener_mask`: valid listener ports
* `listener_adj_mask`: ports reaching listeners

#### 12. Precompute Stream-Specific Port Sets

* `stream_talker_ports`
* `stream_listener_adj_ports`

#### 13. Compute Stray Masks

* `stray_talker_mask`
* `stray_listener_mask`

### Output

* **Returns:** *(None)*

* **Effects:**

  * Initializes a large number of internal data structures:

    * Masks
    * Matrices
    * Cost arrays
    * Caches
  * Enables fast constraint evaluation for optimization algorithms.

## 1. _per_stream_host_role_violations(self, chromosome)

```python id="h8k2qp"
def _per_stream_host_role_violations(self, chromosome):
    act = chromosome > 0          # bool (no astype)

    # Boolean masking (no integer multiply)
    active_talkers   = act & self.talker_mask
    active_listeners = act & self.listener_adj_mask

    # Per-stream counts
    talker_count   = active_talkers.sum(axis=0)
    listener_count = active_listeners.sum(axis=0)

    # Stray per-stream (reduce directly)
    stray_talker   = (act & self.stray_talker_mask).sum(axis=0)
    stray_listener = (act & self.stray_listener_mask).sum(axis=0)

    # Matrix multiply (dominant cost)
    downstream = self.adjacency_mask @ active_listeners

    # Overlap per-stream
    overlap = (active_talkers & downstream).sum(axis=0)

    return np.vstack([
        np.abs(talker_count - 1),      # talker_not_fixed
        np.abs(listener_count - 1),    # listener_not_fixed
        overlap,                       # talker_listener_overlap
        stray_talker,                  # stray_talker
        stray_listener                 # stray_listener
    ])
```

### Parameters

* self *(TSNConstraintEvaluator)*
  Constraint evaluation instance

* chromosome *(np.ndarray)*
  Encoding of active ports per stream

### What the Algorithm Does

* Computes **per-stream host role violations**
* Evaluates correctness of:

  * talker assignment
  * listener assignment
  * role overlap
  * invalid activations

#### Steps

1. Activation mask
   * Converts chromosome into boolean activation vector
   * `act = chromosome > 0`

2. Valid role filtering
   * Extracts active talker ports using `talker_mask`
   * Extracts active listener ports using `listener_adj_mask`

3. Per-stream role counts
   * Computes number of active talkers per stream
   * Computes number of active listeners per stream

4. Stray activation detection
   * Computes stray talker activations outside valid regions
   * Computes stray listener activations outside valid regions

5. Network propagation
   * Applies adjacency matrix multiplication
   * Computes downstream influence of listener activations

6. Role overlap computation
   * Detects overlap between:
     * active talkers
     * downstream listener propagation

7. Violation aggregation
   * Stacks all violation metrics into a single matrix

### Output

* **Returns**

  * `np.ndarray` of shape `(5, n_streams)`

    * talker_not_fixed violations
    * listener_not_fixed violations
    * talker_listener_overlap violations
    * stray_talker violations
    * stray_listener violations

* **Effects**

  * No state modification
  * Pure computation (read-only evaluation)

## 2. UnionFind(elements)
```python
# connectivity helpers (Union-Find)
class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx
```

### Parameters

* elements *(iterable)*
  Collection of elements to initialize the disjoint sets

### What the Algorithm Does

* Implements a **Union-Find (Disjoint Set Union)** data structure
* Tracks connectivity between elements efficiently
* Supports:

  * path compression (`find`)
  * union of sets (`union`)

#### Steps

1. Initialization

   * Creates a parent dictionary mapping each element to itself

2. Find operation

   * Recursively finds root parent of an element
   * Applies path compression for efficiency

3. Union operation

   * Finds root representatives of both elements
   * Merges sets by assigning one root as parent of the other

### Output

* **Returns**

  * No direct return value

* **Effects**

  * Maintains and updates internal disjoint-set structure
  * Enables efficient connectivity queries and merges

Here is a structured description of your first function set, following the same style as your example:

---

## 3. group_connected_ports(chromosome, s_idx)

```python
    def group_connected_ports(self, chromosome, s_idx):
        """Group connected active ports using the adjacency precomputed in __init__."""
        active_ports = [p + 1 for p in np.nonzero(chromosome[:, s_idx])[0]]
        if not active_ports:
            return []
        pid_set = set(active_ports)
        uf = self.UnionFind(active_ports)
        for pid in active_ports:
            for nb in self.next_hop_list[pid - 1]:
                if nb in pid_set:
                    uf.union(pid, nb)
        groups = {}
        for pid in active_ports:
            root = uf.find(pid)
            groups.setdefault(root, set()).add(int(pid))
        return list(groups.values())
```

### Parameters

* chromosome *(2D array / matrix-like structure)*
  Binary representation of active/inactive ports across switches or stages

* s_idx *(int)*
  Index of the switch / stage being analyzed in the chromosome


### What the Algorithm Does

* Builds **connected components (groups)** of active ports based on adjacency relationships
* Uses a **Union-Find (Disjoint Set Union)** structure to efficiently merge connected ports
* Connectivity is determined using a precomputed adjacency list (`self.next_hop_list`)

#### Steps

1. Extract active ports

   * Finds indices where `chromosome[:, s_idx] == 1`
   * Converts them into 1-based port IDs (`+1` adjustment)

2. Early exit

   * If no active ports exist, return an empty list

3. Initialize Union-Find

   * Each active port starts as its own set

4. Merge connected ports

   * For each active port:

     * Check its neighbors in `self.next_hop_list`
     * If neighbor is also active → union the two sets

5. Build groups

   * Find root parent for each active port
   * Aggregate ports sharing the same root into sets

### Output

* **Returns**

  * `list[set[int]]`
    A list of groups, where each group is a set of connected active port IDs

* **Effects**

  * Does not modify input
  * Produces connected components of active ports based on adjacency structure


## 4. talker_listener_groups(groups, talker_ids, listener_ids)

```python
    def talker_listener_groups(self, groups, talker_ids, listener_ids):
        """True if any group contains at least one talker port and one listener port."""
        talker_group = None
        listener_group = None
        
        for group in groups:
            if group & set(talker_ids):
                talker_group = group
            if group & set(listener_ids):
                listener_group = group
        return talker_group, listener_group
```

### Parameters

* groups *(list of sets)*
  Connected port groups (output of `group_connected_ports`)

* talker_ids *(iterable of int)*
  Set/list of ports designated as talkers

* listener_ids *(iterable of int)*
  Set/list of ports designated as listeners

### What the Algorithm Does

* Identifies whether any connected group contains:

  * at least one **talker**
  * at least one **listener**
* Essentially checks for **communication feasibility within connected components**

#### Steps

1. Initialize placeholders

   * `talker_group = None`
   * `listener_group = None`

2. Scan all groups

   * For each group:

     * If it intersects with `talker_ids`, mark as talker-containing group
     * If it intersects with `listener_ids`, mark as listener-containing group

3. Return detected groups

### Output

* **Returns**

  * `(talker_group, listener_group)`

    * `talker_group`: first group found containing at least one talker (or `None`)
    * `listener_group`: first group found containing at least one listener (or `None`)

* **Effects**

  * Does not modify input
  * Identifies presence of talker/listener participation in connected components

## 5. group_penalty(group, k)

```python
    def group_penalty(self, group, k):
        if group is None:
            return k  # completely disconnected group
    
        active_nodes = {self.port_id_to_node[pid] for pid in group}
        
        n_nodes = len(active_nodes)
        n_edges = len(group)
    
        raw = k - n_nodes + (n_edges - n_nodes) / 2
        return int(max(0, raw))
```

### Parameters

* group *(set[int] or None)*
  A connected group of ports, or `None` if no valid group exists

* k *(int)*
  Reference or expected connectivity / grouping value used as a baseline penalty parameter

### What the Algorithm Does

* Computes a **penalty score** based on how well-connected a group is
* Penalizes:

  * missing nodes (disconnectivity)
  * imbalance between nodes and edges
* Converts structural quality into a bounded non-negative integer penalty

#### Steps

1. Handle missing group

   * If `group is None`, return full penalty `k`

2. Map ports to nodes

   * Convert each port ID to its corresponding node via `self.port_id_to_node`

3. Compute structure metrics

   * `n_nodes`: number of unique nodes in the group
   * `n_edges`: number of ports in the group

4. Compute raw penalty

   * Formula:

     * `raw = k - n_nodes + (n_edges - n_nodes) / 2`

5. Clamp result

   * Ensures penalty is not negative
   * Returns integer value

### Output

* **Returns**

  * `int`
    A non-negative penalty score representing group quality

* **Effects**

  * Does not modify input
  * Produces a heuristic score measuring structural imbalance and connectivity quality

## 6. _per_stream_flow_continuity_violations(chromosome)

```python
    def _per_stream_flow_continuity_violations(self, chromosome):
        n = self.n_streams
        v = np.zeros(n, dtype=int)  # 1D is faster to write into
    
        # Local bindings (avoids repeated attribute lookup)
        group_connected_ports = self.group_connected_ports
        talker_listener_groups = self.talker_listener_groups
        group_penalty = self.group_penalty
    
        stream_talker_ports = self.stream_talker_ports
        stream_listener_adj_ports = self.stream_listener_adj_ports
        stream_potential = self.stream_potential
    
        for s_idx in range(n):
            groups = group_connected_ports(chromosome, s_idx)
    
            talker_group, listener_group = talker_listener_groups(
                groups,
                stream_talker_ports[s_idx],
                stream_listener_adj_ports[s_idx],
            )
    
            if (talker_group is not None and listener_group is not None and talker_group is listener_group):
                continue
    
            k = stream_potential[s_idx]
            v[s_idx] = (group_penalty(talker_group, k) + group_penalty(listener_group, k))
    
        return v[np.newaxis, :]
```

### Parameters

* chromosome *(2D array-like, shape: P × S)*
  Binary or weighted representation of port activations per stream

  * P = number of ports
  * S = number of streams

### What the Algorithm Does

* Computes **flow continuity violations per stream**
* Evaluates whether:

  * talker and listener are properly connected in the same component
  * disconnected components incur penalties
* Uses **Union-Find-based grouping + penalty scoring**

#### Steps

1. Initialize output vector

   * `v[s_idx]` stores violation score per stream

2. Local function binding (optimization)

   * Caches frequently used methods and lookup tables:

     * `group_connected_ports`
     * `talker_listener_groups`
     * `group_penalty`
     * stream metadata arrays

3. Loop over streams

   For each stream `s_idx`:

   * Build connected groups of active ports
   * Identify:

     * talker group
     * listener group

4. Check perfect connectivity

   * If:

     * both groups exist AND
     * they are the same group
       → no violation (continue)

5. Otherwise compute penalty

   * Retrieve stream baseline `k = stream_potential[s_idx]`
   * Compute:

     * penalty for talker group
     * penalty for listener group
   * Sum both penalties into `v[s_idx]`

6. Return result

   * Adds extra dimension for consistency

### Output

* **Returns**

  * `np.ndarray` of shape `(1, S)`
    Each entry represents flow continuity violation score per stream

* **Effects**

  * Does not modify input
  * Produces per-stream violation metrics based on connectivity integrity

## 7. _per_stream_flow_conservation_violations(chromosome)

```python
    def _per_stream_flow_conservation_violations(self, chromosome):
        # Local bindings (faster attribute access)
        egress_mask = self.egress_mask
        ingress_mask = self.ingress_mask
        adjacency_mask = self.adjacency_mask
        switch_mask = self.switch_mask.astype(bool)
    
        # Binary activation
        act = (chromosome > 0).astype(int)  # (P, S)
    
        # Compute adjacency once
        adj_act = adjacency_mask @ act     # (P, S)
    
        # Broadcast masks
        active_egp = egress_mask[:, :, None] * act[None, :, :]      # (N,P,S)
        active_ing = ingress_mask[:, :, None] * adj_act[None, :, :] # (N,P,S)
    
        # Degrees (avoid building active_ieg_per_node)
        in_deg  = active_ing.sum(axis=1)   # (N,S)
        out_deg = active_egp.sum(axis=1)   # (N,S)
    
        # ---- Violations ----
    
        # backwarding (intersection count without explicit temp array)
        backwarding = (active_egp & active_ing).sum(axis=(0, 1))
    
        # Only switch nodes
        in_sw  = in_deg[switch_mask]
        out_sw = out_deg[switch_mask]
    
        flow_conservation = np.abs(in_sw - out_sw).sum(axis=0)
        looping = np.maximum(0, in_sw - 1).sum(axis=0)
    
        # Stack into (3, S)
        return np.vstack((
            backwarding,
            flow_conservation,
            looping
        ))
```

### Parameters

* chromosome *(2D array-like, shape: P × S)*
  Activation matrix of ports across streams



### What the Algorithm Does

* Measures **flow conservation violations per stream**
* Checks whether:

  * incoming flow matches outgoing flow
  * switches maintain valid routing constraints
* Detects:

  * backwarding
  * imbalance in flow
  * looping violations

#### Steps

1. Preprocess inputs

   * `act`: binary activation matrix (P × S)
   * `adj_act`: adjacency propagation of active flows

2. Apply masks

   * `active_egp`: outgoing edges per node
   * `active_ing`: incoming edges per node

3. Compute node degrees

   * `in_deg`: total incoming connections per node/stream
   * `out_deg`: total outgoing connections per node/stream

4. Compute violation components

* Backwarding

  * Counts overlapping active incoming/outgoing edges

* Flow conservation violation

  * Only considers switch nodes
  * Measures imbalance:

    * `|in - out|`

* Looping

  * Penalizes multiple incoming activations:

    * `max(0, in - 1)`

5. Stack results

* Combines all violations into matrix:

  * shape: `(3, S)`

  * Rows:

    1. backwarding
    2. flow conservation
    3. looping

### Output

* **Returns**

  * `np.ndarray` of shape `(3, S)`
    Each row corresponds to a different violation type per stream

* **Effects**

  * Does not modify input
  * Produces structured violation breakdown for flow correctness

## 8. _per_stream_priority_conflicts(chromosome)

```python
    def _per_stream_priority_conflicts(self, chromosome):
        # Flatten ports × streams
        flat = chromosome.reshape(-1, self.n_streams)  # shape (n_ports, n_streams)
        
        # Indices of active entries
        rows, cols = np.nonzero(flat > 0)
        
        # Corresponding values (priority labels)
        labels = flat[rows, cols]
        
        # Linearize (row, label) to detect duplicates
        lin = rows * (self.n_streams + 1) + labels  # +1 to avoid zero conflicts
        
        # Count occurrences
        counts = np.bincount(lin, minlength=flat.shape[0] * (self.n_streams + 1))
        
        # Duplicates = counts > 1
        mask_active = counts[lin] > 1
        
        # Allocate output
        v = np.zeros(self.n_streams, dtype=int)
        
        # Sum conflicts per stream
        np.add.at(v, cols[mask_active], 1)
        
        return v[np.newaxis, :]
```

### Parameters

* chromosome *(2D array-like, shape: P × S)*
  Matrix representing port-to-stream priority assignments or activations

### What the Algorithm Does

* Detects **priority conflicts per stream**
* A conflict occurs when:

  * the same port assigns multiple identical priority labels across streams
* Uses a **flattening + hashing technique** for fast duplicate detection

#### Steps

1. Reshape matrix

   * Flattens chromosome into `(ports × streams)` format

2. Extract active entries

   * Finds indices `(row, col)` where values are active (> 0)

3. Encode identity pairs

   * Creates linear index:

     * `lin = row * (n_streams + 1) + label`
   * Ensures unique mapping of (port, priority)

4. Count occurrences

   * Uses `np.bincount` to count duplicates efficiently

5. Detect conflicts

   * Marks entries where count > 1

6. Aggregate per stream

   * For each stream, counts how many conflicts occur

7. Return result

### Output

* **Returns**

  * `np.ndarray` of shape `(1, S)`
    Each entry represents number of priority conflicts per stream

* **Effects**

  * Does not modify input
  * Produces per-stream conflict counts based on duplicate priority assignments

## 9. _per_stream_latency_violations_priority(chromosome)

```python
    def _per_stream_latency_violations_start_time(self, chromosome):
        """
        chromosome[p, s] = transmission start time (0 = inactive)
        """
    
        active = chromosome > 0
    
        # Replace inactive entries so they don't affect min/max
        start_min = np.where(active, chromosome, np.inf)
        start_max = np.where(active, chromosome, -np.inf)
    
        # Per-stream earliest and latest transmission
        t_min = start_min.min(axis=0)
        t_max = start_max.max(axis=0)
    
        # Transmission window (use worst-case hop window)
        # windLengths already = transmission duration per port/stream
        max_window = np.where(active, self.windLengths, 0).max(axis=0)
    
        # Accumulated latency
        latency = (t_max - t_min) + max_window
    
        # Streams with no activity → no violation
        latency = np.where(np.isfinite(t_min), latency, 0)
    
        violations = (latency > self.stream_maxLatencies).astype(int)
    
        return violations[np.newaxis, :]
```

### Parameters

* chromosome *(2D array-like, shape: P × S)*
  Matrix where each entry represents a priority value for a port–stream pair

  * P = number of ports
  * S = number of streams

### What the Algorithm Does

* Computes **latency violations per stream based on priority-ordered execution**
* Simulates transmission accumulation according to priority ordering
* Evaluates whether total latency exceeds allowed stream limits

#### Steps

1. Identify active entries

   * `active_mask = chromosome > 0`
   * Inactive entries are ignored in calculations

2. Assign sorting priorities

   * Replace inactive entries with a large number (`big_num`)
   * Ensures inactive values move to the end when sorting

3. Sort by priority per port

   * Uses `np.argsort` to order streams within each port
   * Reorders:

     * hop costs (`self.all_hops_costs`)
     * active mask

4. Mask inactive entries

   * Zeroes out hops that are not active after sorting

5. Compute cumulative delay

   * `cumsum_delays = np.cumsum(sorted_hops, axis=1)`
   * Models sequential transmission delay accumulation per port

6. Map back to original stream order

   * Restores sorted cumulative delays into original chromosome layout

7. Compute total stream latency

   * Sums all contributions per stream
   * Considers only active entries

8. Detect violations

   * Compares against `self.stream_maxLatencies`
   * Marks streams exceeding limit

### Output

* **Returns**

  * `np.ndarray` of shape `(1, S)`
    Binary indicator per stream:

    * 1 = latency violation
    * 0 = valid

* **Effects**

  * Does not modify input
  * Produces per-stream latency constraint violations based on priority ordering

---

## 10. _per_stream_latency_violations_start_time(chromosome)

```python
    def _per_stream_latency_violations_start_time(self, chromosome):
        """
        chromosome[p, s] = transmission start time (0 = inactive)
        """
    
        active = chromosome > 0
    
        # Replace inactive entries so they don't affect min/max
        start_min = np.where(active, chromosome, np.inf)
        start_max = np.where(active, chromosome, -np.inf)
    
        # Per-stream earliest and latest transmission
        t_min = start_min.min(axis=0)
        t_max = start_max.max(axis=0)
    
        # Transmission window (use worst-case hop window)
        # windLengths already = transmission duration per port/stream
        max_window = np.where(active, self.windLengths, 0).max(axis=0)
    
        # Accumulated latency
        latency = (t_max - t_min) + max_window
    
        # Streams with no activity → no violation
        latency = np.where(np.isfinite(t_min), latency, 0)
    
        violations = (latency > self.stream_maxLatencies).astype(int)
    
        return violations[np.newaxis, :]
```

### Parameters

* chromosome *(2D array-like, shape: P × S)*
  Start-time matrix where:

  * value > 0 → active transmission start time
  * value = 0 → inactive

### What the Algorithm Does

* Computes **latency violations based on start-time spread**
* Measures time window between earliest and latest transmission
* Adds worst-case transmission duration
* Checks against maximum allowed latency per stream

#### Steps

1. Identify active transmissions

   * `active = chromosome > 0`

2. Mask inactive values

   * Replace inactive entries:

     * `+∞` for minimum calculation
     * `-∞` for maximum calculation

3. Compute per-stream timing bounds

   * `t_min`: earliest start time per stream
   * `t_max`: latest start time per stream

4. Compute transmission window

   * Uses worst-case hop duration per stream
   * `max_window = max(self.windLengths where active)`

5. Compute total latency

   * `latency = (t_max - t_min) + max_window`

6. Handle inactive streams

   * If no active transmissions → latency = 0

7. Detect violations

   * Compare latency against `self.stream_maxLatencies`

### Output

* **Returns**

  * `np.ndarray` of shape `(1, S)`
    Binary indicator per stream:

    * 1 = latency violation
    * 0 = valid

* **Effects**

  * Does not modify input
  * Computes timing-based latency constraint violations

---

## 11. _per_stream_talker_offset_violations(chromosome)

```python
    def _per_stream_talker_offset_violations(self, chromosome):
        """
        chromosome[p, s] = start time (0 = inactive)
        """
    
        active = chromosome > 0
    
        # Active talker entries
        talker_active = self.talker_mask & active
    
        # Extract talker start time
        talker_times = np.where(talker_active, chromosome, 0)
        talker_offset = talker_times.max(axis=0)
    
        intervals = self.stream_intervals
    
        violations = (
            (talker_offset >= intervals)
        ).astype(int)
    
        return violations[np.newaxis, :]
```

### Parameters

* chromosome *(2D array-like, shape: P × S)*
  Start-time matrix where:

  * value > 0 → active transmission start time
  * value = 0 → inactive

### What the Algorithm Does

* Detects **violations of allowed talker timing offsets per stream**
* Ensures talker transmissions occur within valid time intervals
* Checks whether maximum talker offset exceeds stream constraints

#### Steps

1. Identify active entries

   * `active = chromosome > 0`

2. Filter talker ports

   * `talker_active = self.talker_mask & active`

3. Extract talker timing

   * `talker_times = chromosome where active talkers exist`
   * Non-active entries set to 0

4. Compute maximum talker offset

   * `talker_offset = max(talker_times per stream)`

5. Compare against stream interval constraint

   * `intervals = self.stream_intervals`

6. Detect violations

   * If talker offset exceeds allowed interval → violation = 1

### Output

* **Returns**

  * `np.ndarray` of shape `(1, S)`
    Binary indicator per stream:

    * 1 = talker offset violation
    * 0 = valid

* **Effects**

  * Does not modify input
  * Produces per-stream timing compliance violations for talker scheduling

## 12. _per_stream_macrocycle_overlap_violations(chromosome)

```python
    def _per_stream_macrocycle_overlap_violations(self, chromosome):
        """
        Detect periodic window overlaps over the macro-cycle.
        Implements the periodic-interval collision test: 
            (phase1 - phase2) mod gcd(interval1, interval2) < window1 + window2; where phase = start % interval
        chromosome[p,s] = start time (0 = inactive)
        """
    
        active = chromosome > 0  # (P,S)

        # --- normalize start times into stream phase
        phase = chromosome % self.stream_intervals[None, :]
    
        # --- pairwise start-time differences (P,S,S) without abs
        diff = phase[:, :, None] - phase[:, None, :]
    
        # --- periodic phase difference using precomputed gcd
        delta = np.mod(diff, self.streams_pairwise_gcl)
    
        # --- overlap condition
        overlap = delta < self.stream_pairwise_windowSum
    
        # remove self comparisons
        overlap &= self.pair_mask
    
        # require both streams active on same port
        both_active = active[:, :, None] & active[:, None, :]
        overlap &= both_active
    
        # --- count violations per stream
        # (sum over ports and partner streams)
        v = overlap.sum(axis=(0, 2))
    
        return v[np.newaxis, :]
```

### Parameters

* self *(object instance of the class)*
  The `self` parameter represents the current instance of the class and provides access to:

  * configuration constants (e.g., `self.stream_intervals`)
  * precomputed matrices (e.g., `self.streams_pairwise_gcl`)
  * helper masks (e.g., `self.pair_mask`)
  * derived scheduling constraints used in macro-cycle analysis

  In this function, `self` is essential because all periodic timing rules and pairwise stream relationships are stored in the object state rather than passed as arguments.

* chromosome *(2D array-like, shape: P × S)*
  Start-time matrix where:

  * P = number of ports
  * S = number of streams
  * value = transmission start time (0 means inactive)

### What the Algorithm Does

* Detects **periodic macro-cycle overlap violations between streams**
* Checks whether streams collide within repeating time windows
* Uses modular arithmetic to model periodic scheduling behavior
* Ensures that streams respect cycle-based separation constraints

#### Steps

1. Identify active transmissions

   * `active = chromosome > 0`

2. Convert start times into periodic phase

   * `phase = chromosome % self.stream_intervals`
   * Maps each stream into its position inside its repeating cycle

3. Compute pairwise phase differences

   * `diff = phase[:, :, None] - phase[:, None, :]`
   * Produces (P, S, S) matrix comparing all stream pairs per port

4. Apply periodic wrap-around using GCD structure

   * `delta = np.mod(diff, self.streams_pairwise_gcl)`
   * Ensures differences are evaluated within cycle compatibility windows

5. Detect overlap condition

   * `overlap = delta < self.stream_pairwise_windowSum`
   * Marks pairs that violate allowed separation thresholds

6. Remove invalid comparisons

   * `overlap &= self.pair_mask`
   * Eliminates self-comparisons and invalid stream pairs

7. Require co-activity on same port

   * `both_active = active[:, :, None] & active[:, None, :]`
   * Ensures overlap only counts when both streams are active at same port

8. Count violations

   * Sum over ports and stream pairs:

     * `v = overlap.sum(axis=(0, 2))`

### Output

* **Returns**

  * `np.ndarray` of shape `(1, S)`
    Each entry is the number of macrocycle overlap violations per stream

* **Effects**

  * Does not modify input
  * Produces periodic scheduling conflict metrics across streams

---

## 13. _per_stream_window_sequence_violations(chromosome)

```python
    def _per_stream_window_sequence_violations(self, chromosome):
        """
        chromosome[p, s] = start time (0 = inactive)
        
        Enforces two scheduling constraints:
        -------------------------------------------------------
        1) INTRA-STREAM SEQUENCING (causality)
    
            Constraint per connection (u → d): chromosome[d, s] >= chromosome[u, s] + hop_cost[u, s]
    
        Prevents a frame leaving a downstream port before it
        arrives from upstream. Violations are marked at the downstream port (d).
        -------------------------------------------------------
        2) INTER-STREAM FIFO (slot stealing / interleaving)
    
        For two streams s1,s2 on the same port:
            (t11 - t21)*(t12 - t22) >= 0
        meaning arrival order must equal departure order.
        Prevents one stream stealing another stream’s slot.
        """
        n_ports, n_streams = self.n_ports, self.n_streams
        upstream_ports = self._upstream_ports
        downstream_ports = self._downstream_ports
    
        if upstream_ports.size == 0:
            return np.zeros((n_streams,), dtype=int)
        
        upstream_start = chromosome[upstream_ports]          # (connections, S)
        downstream_start = chromosome[downstream_ports]      # (connections, S)
    
        upstream_active = upstream_start > 0
        downstream_active = downstream_start > 0
        # Only check connections where stream exists at both ends
        valid = upstream_active & downstream_active
        # Required arrival time at downstream
        enqueuing_start = upstream_start + self.all_hops_costs[upstream_ports] # per connection

        # -------------------------------------------------
        # PART 1 — INTRA-STREAM SEQUENCING
        # -------------------------------------------------
        connection_violations = valid & (downstream_start < enqueuing_start)
        # Accumulate violations at downstream ports
        violations = np.zeros((n_ports, n_streams), dtype=bool)
        np.maximum.at(
            violations,
            (downstream_ports[:, None], np.arange(n_streams)),
            connection_violations
        )
        # Sum violations per stream
        seq_v  = violations.sum(axis=0)
        # -------------------------------------------------
        # PART 2 — INTER-STREAM FIFO (slot stealing)
        # -------------------------------------------------
        active = chromosome > 0
        
        eg_queue_arrival = np.zeros((n_ports, n_streams), dtype=float)           
        np.maximum.at(
            eg_queue_arrival,
            (downstream_ports[:, None], np.arange(n_streams)),
            enqueuing_start
        )

        arr_diff = eg_queue_arrival[:, :, None] - eg_queue_arrival[:, None, :]
        dep_diff = chromosome[:, :, None] - chromosome[:, None, :]
        
        interleave = (arr_diff * dep_diff) < 0
        interleave &= self.pair_mask
        both_active = active[:, :, None] & active[:, None, :]
        interleave &= both_active
        
        fifo_v = interleave.sum(axis=(0,2))

        return np.vstack((seq_v[np.newaxis, :], fifo_v[np.newaxis, :]))
```

### Parameters

* self *(object instance of the class)*
  Provides access to:

  * network topology (`self._upstream_ports`, `self._downstream_ports`)
  * hop timing costs (`self.all_hops_costs`)
  * configuration constants (`self.n_ports`, `self.n_streams`)
  * pairwise masks (`self.pair_mask`)

  Here, `self` is critical because sequencing rules depend on precomputed port connectivity and timing data stored inside the object.

* chromosome *(2D array-like, shape: P × S)*
  Start-time matrix where:

  * P = ports
  * S = streams
  * value > 0 indicates active transmission start time

### What the Algorithm Does

* Evaluates **two classes of scheduling violations per stream**
* Ensures correct temporal ordering of transmissions:

  1. intra-stream causality (data must not arrive before it is sent)
  2. inter-stream FIFO fairness (no slot stealing between streams)

---

### Steps

#### Setup

1. Extract topology information

   * `upstream_ports`
   * `downstream_ports`

2. Handle empty topology

   * If no connections exist → return zero violations

---

### PART 1: Intra-stream sequencing violations

3. Extract timing at endpoints

   * `upstream_start = chromosome[upstream_ports]`
   * `downstream_start = chromosome[downstream_ports]`

4. Check activation validity

   * Both upstream and downstream must be active

5. Compute required arrival time

   * `enqueuing_start = upstream_start + hop_cost`

6. Detect causality violations

   * Violation occurs if:

     * downstream starts before arrival

7. Accumulate violations per port

   * Uses `np.maximum.at` to assign violations to downstream ports

8. Aggregate per stream

   * `seq_v = violations.sum(axis=0)`

---

### PART 2: Inter-stream FIFO violations

9. Build arrival matrix

   * `eg_queue_arrival` stores effective arrival times per port/stream

10. Compute pairwise comparisons

* `arr_diff`: difference in arrival times between streams
* `dep_diff`: difference in departure times

11. Detect ordering inversion

* `interleave = (arr_diff * dep_diff) < 0`
* Indicates streams violate FIFO ordering consistency

12. Filter invalid comparisons

* Apply:

  * `self.pair_mask`
  * `both_active`

13. Count FIFO violations

* `fifo_v = interleave.sum(axis=(0,2))`

---

### Output

* **Returns**

  * `np.ndarray` of shape `(2, S)`

    * Row 0 → intra-stream sequencing violations
    * Row 1 → inter-stream FIFO violations

* **Effects**

  * Does not modify input
  * Produces structured temporal correctness evaluation for scheduling feasibility

## 14. per_stream_all_violations(chromosome)

```python
    def per_stream_all_violations(self, chromosome):
        key = chromosome.tobytes()
        # get violations
        cached = self._violation_evaluation_cache.get(key)
        if cached is not None:
            per_stream_violations = cached
        else:
            # List to hold violation arrays
            all_v = []

            # Collect results as (1, S) arrays and append to list
            for check_function in self.active_checks:
                all_v.append(check_function(chromosome))               

            per_stream_violations = np.vstack(all_v)

            self._violation_evaluation_cache[key] = per_stream_violations
        # Concatenate all violation arrays along the first axis (rows)
        return per_stream_violations
```

### Parameters

* self *(object instance of the class)*
  Provides access to:

  * `self.active_checks` → list of violation-check functions
  * `self._violation_evaluation_cache` → memoization storage
  * internal configuration for all stream constraint rules

  Here, `self` acts as the central controller that defines which violation rules are active and stores cached results to avoid recomputation.

* chromosome *(2D array-like, shape: P × S)*
  Port-stream activation matrix:

  * P = ports
  * S = streams
  * non-zero values indicate active configuration states

### What the Algorithm Does

* Computes **all per-stream violation metrics in a unified pipeline**
* Uses caching to avoid recomputing results for identical chromosomes
* Aggregates results from multiple constraint-checking functions

#### Steps

1. Hash chromosome

   * Converts chromosome into byte representation:

     * `key = chromosome.tobytes()`
   * Used for cache lookup

2. Check cache

   * If results exist in `_violation_evaluation_cache`, reuse them

3. If not cached:

   * Initialize empty list `all_v`

4. Run all violation checks

   * Iterates through `self.active_checks`
   * Each function returns `(1, S)` violation array
   * Appends results to list

5. Stack results

   * `per_stream_violations = np.vstack(all_v)`
   * Produces unified violation matrix

6. Store in cache

   * Saves result for future reuse

### Output

* **Returns**

  * `np.ndarray` of shape `(K, S)`

    * K = number of active violation checks
    * S = number of streams
  * Each row corresponds to a different violation type

* **Effects**

  * Does not modify input chromosome
  * Populates evaluation cache for performance optimization

---

## 15. feasible_streams(chromosome)

```python
    def feasible_streams(self, chromosome):
        violations_array = self.per_stream_all_violations(chromosome)
        infeasible_streams = np.any(violations_array > 0, axis=0)
        return ~infeasible_streams
```

### Parameters

* self *(object instance of the class)*
  Provides access to:

  * violation evaluation system via `per_stream_all_violations`
  * internal constraint logic defining feasibility

* chromosome *(2D array-like, shape: P × S)*
  Configuration matrix defining active stream behavior

### What the Algorithm Does

* Determines which streams are **fully feasible (violation-free)**
* Aggregates all violation types per stream
* Flags any stream with at least one violation

#### Steps

1. Compute all violations

   * `violations_array = self.per_stream_all_violations(chromosome)`
   * Shape: `(K, S)`

2. Detect infeasible streams

   * `infeasible_streams = np.any(violations_array > 0, axis=0)`
   * A stream is infeasible if any violation type is non-zero

3. Return feasibility mask

   * Logical negation:

     * `~infeasible_streams`

### Output

* **Returns**

  * `np.ndarray` of shape `(S,)`
    Boolean array:

    * `True` → stream is feasible
    * `False` → stream has at least one violation

* **Effects**

  * Does not modify input
  * Provides per-stream feasibility classification

---

## 16. compute_metrics(chromosome, alpha=0.9, eps=1e-9)

```python
    def compute_metrics(self, chromosome, alpha=0.9, eps=1e-9):
        """Vectorized performance metrics (congestion, imbalance, path length)."""
        key = chromosome.tobytes()
        cached = self._metrics_evaluation_cache.get(key)
        if cached is not None:
            metrics = cached
        else:
            load_per_port = np.sum((chromosome > 0).astype(float) * self.windLengths, axis=1)
            util = load_per_port / np.maximum(self.macro_cycle * self.port_speeds, eps)
            
            avg_util = float(np.clip(np.nanmean(util), 0, 1))
            std_util = float(np.clip(np.nanstd(util), 0, 1))
            congestion_score = float(np.clip(alpha * avg_util + (1 - alpha) * std_util, 0, 1))
            
            mean_load = float(np.mean(load_per_port))
            if mean_load <= eps:
                load_imbalance = 1.0
            else:
                imbalance = float(np.std(load_per_port) / mean_load)
                load_imbalance = float(min(imbalance, 1.0) if np.isfinite(imbalance) else 1.0)
            
            active_per_stream = np.count_nonzero(chromosome > 0, axis=0)
            avg_path_len = float(np.mean(active_per_stream))
            path_len_score = float(np.clip(avg_path_len / (self.n_ports + eps), 0, 1))

            metrics = np.array([congestion_score, avg_util, std_util, load_imbalance, path_len_score, avg_path_len])
            self._metrics_evaluation_cache[key] = metrics
        # Return the metrics as an array
        return metrics
```

### Parameters

* self *(object instance of the class)*
  Provides access to:

  * `self.windLengths` → per-port transmission durations
  * `self.macro_cycle` → global scheduling cycle length
  * `self.port_speeds` → port capacity/speed limits
  * `self.n_ports` → number of ports
  * `_metrics_evaluation_cache` → cached metric results

  Here, `self` defines the system-wide performance model used to evaluate congestion, imbalance, and path characteristics.

* chromosome *(2D array-like, shape: P × S)*
  Port-stream activation matrix used to derive system utilization

* alpha *(float, default=0.9)*
  Weighting factor balancing:

  * average utilization
  * utilization variability

* eps *(float, default=1e-9)*
  Small constant to prevent division by zero

---

### What the Algorithm Does

* Computes **global performance metrics of the network configuration**
* Measures:

  * congestion
  * load imbalance
  * path length characteristics
* Uses vectorized computation for efficiency
* Employs caching to avoid recomputation

---

### Steps

1. Check cache

   * Uses `chromosome.tobytes()` as key
   * Returns cached metrics if available

---

### Load computation

2. Compute per-port load

   * `load_per_port = sum(active ports × windLengths)`
   * Represents total transmission cost per port

3. Compute utilization

   * `util = load_per_port / (macro_cycle × port_speeds)`
   * Normalized usage of each port

---

### Congestion metrics

4. Average utilization

   * `avg_util = mean(util)` clipped to `[0,1]`

5. Utilization variability

   * `std_util = std(util)` clipped to `[0,1]`

6. Combined congestion score

   * Weighted mix:

     * `congestion_score = alpha * avg_util + (1 - alpha) * std_util`

---

### Load imbalance

7. Mean load

   * `mean_load = mean(load_per_port)`

8. If near-zero load

   * imbalance = 1.0 (max penalty)

9. Otherwise:

   * `imbalance = std(load) / mean(load)`
   * Clipped to `[0,1]`

---

### Path metrics

10. Active connections per stream

* `active_per_stream = count_nonzero(chromosome > 0, axis=0)`

11. Average path length

* `avg_path_len = mean(active_per_stream)`

12. Path score normalization

* `path_len_score = avg_path_len / n_ports`

---

### Final metric vector

13. Stack metrics into array:

* `[congestion_score,
   avg_util,
   std_util,
   load_imbalance,
   path_len_score,
   avg_path_len]`

14. Cache result

---

### Output

* **Returns**

  * `np.ndarray` of shape `(6,)`
    Contains:

    1. congestion_score
    2. avg_util
    3. std_util
    4. load_imbalance
    5. path_len_score
    6. avg_path_len

* **Effects**

  * Does not modify input
  * Stores computed metrics in cache for reuse
## 17. evaluate(chromosome)

```python
    def evaluate(self, chromosome):
        key = chromosome.tobytes()
        # get violations
        per_stream_violations = self.per_stream_all_violations(chromosome)

        # get metrics
        metrics = self.compute_metrics(chromosome)
        
        return per_stream_violations.sum(axis=1), metrics
```

### Parameters

* self *(object instance of the class)*
  Provides access to the full evaluation system, including:

  * `self.per_stream_all_violations` → computes constraint violations per stream
  * `self.compute_metrics` → computes global performance metrics
  * internal caches for optimization (used indirectly by called methods)

  In this function, `self` acts as the orchestrator that connects constraint evaluation and performance measurement into a single evaluation pipeline.

* chromosome *(2D array-like, shape: P × S)*
  Configuration matrix representing:

  * P = ports
  * S = streams
  * non-zero entries indicate active scheduling/connection decisions

### What the Algorithm Does

* Performs a **full evaluation of a chromosome**
* Combines:

  * per-stream constraint violations
  * global network performance metrics
* Produces a compact summary used for optimization (e.g., evolutionary algorithm fitness evaluation)

#### Steps

1. Compute per-stream violations

   * Calls:

     * `self.per_stream_all_violations(chromosome)`
   * Output shape: `(K, S)`

     * K = number of violation types
     * S = number of streams

2. Aggregate violations

   * `per_stream_violations.sum(axis=1)`
   * Collapses all violation types into a single scalar per stream
   * Result shape: `(S,)`

3. Compute performance metrics

   * Calls:

     * `self.compute_metrics(chromosome)`
   * Returns a vector of global system performance indicators

4. Return combined evaluation

   * Returns a tuple:

     * per-stream violation scores
     * system-level metrics

### Output

* **Returns**

  * `(np.ndarray, np.ndarray)`

  1. **Violation vector**

     * Shape: `(S,)`
     * Each entry = total violation score per stream

  2. **Metrics vector**

     * Shape: `(6,)`
     * Global performance indicators:

       * congestion, utilization, imbalance, path length, etc.

* **Effects**

  * Does not modify input chromosome
  * Acts as the main evaluation function combining feasibility + performance scoring
  * Relies heavily on internal caching via called methods for efficiency

## 18. suggest_corrections(Y)

```python
    def suggest_corrections(self, Y):

        adj_port_mask= self.to_node_mask[None, :, :, None]
        egress_mask = self.egress_mask[None, :, :, None]
        switch_mask = self.switch_mask[None, :, None]
        talker_mask   = self.talker_mask[None,:,:]
        listener_adj_mask = self.listener_adj_mask[None,:,:]
        adjacency_mask = self.adjacency_mask
        stream_intervals = self.stream_intervals[None, None, :]

        act = Y > 0
        adj_act = (adjacency_mask @ act)
        expanded_act = act[:, None, :, :]  # (I,N, P,S)
        expanded_adj_act = adj_act[:, None, :, :]

        active_egp = egress_mask & expanded_act
        active_adj_egp = adj_port_mask & expanded_act
        
        out_deg = active_egp.sum(axis=2) 
        in_deg = active_adj_egp.sum(axis=2)

        # switches state
        active_switch_egp = out_deg * switch_mask
        active_switch_adj_egp = in_deg * switch_mask

        active_hosts_egp = out_deg * (1-switch_mask)
        active_hosts_adj_egp = in_deg * (1-switch_mask)

        # hosts (talkers and listenres) state
        active_talker_egp = active_egp.sum(axis=1) * talker_mask
        active_listener_adj_ing = active_adj_egp.sum(axis=1) * listener_adj_mask
        
        n_talkers = active_talker_egp.sum(axis=1) 
        n_listeners = active_listener_adj_ing.sum(axis=1)

        forwarding_hosts = ((active_hosts_egp == 1) & (active_hosts_adj_egp == 1))[:,:,None,:]

        # switches grouping
        overloaded_switches = (active_switch_adj_egp > 1)[:,:,None,:]
        sourced_switches = ((active_switch_adj_egp == 0) & (active_switch_egp > 0))[:,:,None,:] 
        sinked_switches = ((active_switch_adj_egp > 0) & (active_switch_egp==0))[:,:,None,:]

        # hosts grouping
        too_many_talkers = n_talkers > 1
        missing_talker = n_talkers < 1
        too_many_listeners = n_listeners > 1
        missing_listener = n_listeners < 1

        # removal pressure
        remove_bias = (
            2 * too_many_talkers[:, None, :] * active_talker_egp # remove extra talkers
            + 2 * too_many_listeners[:, None, :] * active_listener_adj_ing # remove extra listeners
            + 5 * ((active_adj_egp + active_egp) & forwarding_hosts).sum(axis=1) # Encourage moving talker and listener elsewhere
            + (active_adj_egp & overloaded_switches).sum(axis=1) # remove looping
            + (active_egp & sourced_switches).sum(axis=1) # remove blankstart switches
            + (active_adj_egp & sinked_switches).sum(axis=1) # remove deadend switches
            + (expanded_adj_act & expanded_act).sum(axis=1) # remove backwarding
        )

        # addition pressure
        add_bias = (
             missing_talker[:, None, :] * talker_mask # activate talker
            +missing_listener[:, None, :] * listener_adj_mask # activate listener
            +(egress_mask & sinked_switches).sum(axis=1) # extend deadend switches
            + (adj_port_mask & sourced_switches).sum(axis=1) # extend blankstart switches
        )
        
        add_bias*=(1-self.stray_talker_mask)*(1-self.stray_listener_mask) # avoid useless activations

        # priority pressure
        if self.scheduling_type.value == ScheduleEncodingType.PRIORITY.value:
            counts = (Y[:, :, :, None] == Y[:, :, None, :]).sum(axis=2)
            conflict_mask = (counts > 1)[:, None, :, :] #& act
            gene_value_bias = conflict_mask.astype(int) #change conflicting priority
        elif self.scheduling_type.value == ScheduleEncodingType.START_TIME.value:
            n_chromosomes, n_ports, n_streams = Y.shape
        
            upstream_ports = self._upstream_ports
            downstream_ports = self._downstream_ports
            all_hops_costs = self.all_hops_costs

            stream_pairwise_windowSum = self.stream_pairwise_windowSum[None]
            streams_pairwise_gcl = self.streams_pairwise_gcl[None, None, :, :]
            pair_mask = self.pair_mask[None, None]
        
            # Gather per-connection start times
            upstream_start = Y[:, upstream_ports, :]       # (I, connections, S)
            downstream_start = Y[:, downstream_ports, :]   # (I, connections, S)
    
            upstream_active = upstream_start > 0
            downstream_active = downstream_start > 0
            valid = upstream_active & downstream_active
    
            # Required arrival time at downstream (broadcast correctly)
            enqueuing_start = upstream_start + all_hops_costs[upstream_ports][None, :, :]
    
            connection_bias = np.where(valid & (downstream_start < enqueuing_start),
                               enqueuing_start - downstream_start,
                               0)
    
            sequencing_bias_downstream = np.zeros_like(Y, dtype=int) # (I, P, S)
            sequencing_bias_upstream = np.zeros_like(Y, dtype=int) # (I, P, S)
            eg_queue_arrival = np.zeros((n_chromosomes, n_ports, n_streams), dtype=float) # !!!!
            for i in range(n_chromosomes):
                np.maximum.at(
                    sequencing_bias_downstream[i],
                    (downstream_ports[:, None], np.arange(n_streams)),
                    connection_bias[i]
                )
                np.maximum.at(
                    sequencing_bias_upstream[i],
                    (upstream_ports[:, None], np.arange(n_streams)),
                    connection_bias[i]
                )
            # ---  interleaving bias ---
                np.maximum.at(                                             # !!!!
                    eg_queue_arrival[i],
                    (downstream_ports[:, None], np.arange(n_streams)),
                    enqueuing_start[i]
                )

            arr_diff = eg_queue_arrival[:, :, :, None] - eg_queue_arrival[:, :, None, :]
            dep_diff = Y[:, :, :, None] - Y[:, :, None, :]
            
            interleave = (arr_diff * dep_diff) < 0
            interleave &= pair_mask
            act = Y > 0
            both_active = act[:, :, :, None] & act[:, :, None, :]  # require both streams active
            interleave &= both_active
            
            sign_upstream = np.sign(arr_diff)
            sign_downstream = np.sign(dep_diff)

            pressure_down = sign_upstream * np.abs(dep_diff) * interleave
            pressure_up = sign_downstream * np.abs(arr_diff) * interleave
            
            downstream_bias = np.sum(pressure_down, axis=3)
            
            upstream_bias_tmp = np.sum(pressure_up, axis=3)
            upstream_bias = np.zeros_like(Y)
            for i in range(n_chromosomes):
                np.add.at(
                    upstream_bias[i],
                    (upstream_ports[:, None], np.arange(n_streams)),
                    upstream_bias_tmp[i, upstream_ports, :]
                )
            interleaving_pressure = upstream_bias + downstream_bias
            # ---  talker offset bias ---
            # Compute per-stream latest talker start
            talker_times = np.where(active_talker_egp > 0, Y, 0)               # (I, P, S)
            talker_offset = talker_times.max(axis=1, keepdims=True)            # (I, 1, S)
            # violation = talker too late
            talker_bias = np.maximum(0, talker_offset - stream_intervals)  # (I, 1, S)
            
            # ---  maxlatency bias ---
            t_max = np.where(act, Y, -np.iinfo(Y.dtype).max).max(axis=1, keepdims=True)
            latency_limit = talker_offset + np.asarray(self.stream_maxLatencies)[None,None,:]
            end_time = Y + self.windLengths[None,:,:]
            is_latest_active = (Y == t_max) & act
            latency_bias = (latency_limit - end_time) * (end_time > latency_limit)  * is_latest_active

            # --- macrocycle_bias overlap bias ---
            phase = Y % stream_intervals
            diff = phase[:, :, :, None] - phase[:, :, None, :]
            delta = diff - np.floor(diff * self.streams_pairwise_gcl_inv) * streams_pairwise_gcl
            
            # diff = Y[:, :, :, None] - Y[:, :, None, :] # (I,P,S,S) # pairwise start differences
            # delta = diff - streams_pairwise_gcl * np.floor(diff / streams_pairwise_gcl)
            overlap = delta < stream_pairwise_windowSum # overlap condition
            overlap &= pair_mask                  # remove self comparisons
            overlap &= both_active
            # signed cyclic direction
            half_g = streams_pairwise_gcl / 2
            signed_delta = np.where(delta <= half_g, delta, delta - streams_pairwise_gcl)
            # overlap magnitude
            overlap_depth = stream_pairwise_windowSum - delta
            overlap_depth = np.maximum(overlap_depth, 0)
            overlap_depth *= overlap
            
            gene_value_bias =  np.stack([
                sequencing_bias_downstream # (sequencing) move forward early downstream windows at switches
                , - sequencing_bias_upstream # (sequencing) move backward late upstream windows
                , - talker_bias  * talker_mask # (talker offset) talker out of 1st period
                , + (np.sign(signed_delta) * overlap_depth).sum(axis=3) # (macrocycle overlap) overlapping slots repel each other
                , + np.maximum(upstream_bias, 0) # (interleaving) move forward owner at upstream
                , + np.minimum(upstream_bias, 0) # (interleaving) move backward thief at uptream 
                , + np.maximum(downstream_bias, 0) # (interleaving) move forward thief at downstream
                , + np.minimum(downstream_bias, 0) # (interleaving) move backward owner at downstream
                , + latency_bias # (maxlatency) move latest window bakward
            ], axis=1).astype(int)

        else:
            gene_value_bias = expanded_act
                    
        #####################################
        remove_bias *= act
        add_bias    *= (~act) & (adj_act == 0) # avoid backward connection and talker/listener overlap
        gene_value_bias  = gene_value_bias * expanded_act

        return remove_bias, add_bias, gene_value_bias    
```

### Parameters

* self *(object instance of the class)*
  Central configuration and topology container providing:

  * network structure (ports, switches, adjacency)
  * masks (egress, ingress, talker/listener, etc.)
  * scheduling configuration (priority or start-time encoding rules)
  * precomputed matrices for routing, timing, and constraint checking

  In this function, `self` acts as a **constraint engine context**, supplying all structural rules used to generate correction signals for invalid chromosomes.

* Y *(3D array-like, shape: I × P × S)*
  Candidate population or solution batch where:

  * I = number of individuals (chromosomes)
  * P = ports
  * S = streams
  * values represent either:

    * activation (binary case), or
    * priority / start-time (depending on encoding type)

---

### What the Algorithm Does

* Generates **correction pressures (repair signals)** for invalid solutions
* Instead of returning feasibility, it produces:

  * removal pressure (what to delete)
  * addition pressure (what to activate)
  * gene-level correction bias (how to modify values)
* Supports both:

  * PRIORITY encoding
  * START_TIME encoding
  * fallback binary-like encoding

---

### Step 1: Initialization and Mask Setup

1. Load structural masks from `self`

   * `adj_port_mask` → adjacency-to-node mapping
   * `egress_mask` → outgoing port constraints
   * `switch_mask` → identifies switch nodes
   * `talker_mask`, `listener_adj_mask` → endpoint roles
   * `adjacency_mask` → connectivity matrix
   * `stream_intervals` → timing constraints

---

### Step 2: Activation and propagation

2. Compute activation state

   * `act = Y > 0`

3. Propagate adjacency

   * `adj_act = adjacency_mask @ act`

4. Expand dimensions for broadcasting

   * `expanded_act = (I, 1, P, S)`
   * `expanded_adj_act = (I, 1, P, S)`

---

### Step 3: Degree computation

5. Compute outgoing and incoming degrees

   * `out_deg = active_egp.sum(axis=2)`
   * `in_deg = active_adj_egp.sum(axis=2)`

---

### Step 4: Node role classification

6. Split into switches and hosts

   * Switches:

     * `active_switch_egp`, `active_switch_adj_egp`
   * Hosts:

     * `active_hosts_egp`, `active_hosts_adj_egp`

---

### Step 5: Endpoint analysis

7. Identify talker/listener structure

   * `active_talker_egp`
   * `active_listener_adj_ing`

8. Count per stream

   * `n_talkers`, `n_listeners`

---

### Step 6: Structural grouping signals

9. Identify structural conditions

   * `forwarding_hosts`
   * `overloaded_switches`
   * `sourced_switches`
   * `sinked_switches`

---

### Step 7: Host imbalance detection

10. Detect host-level issues

* too many / missing talkers
* too many / missing listeners

---

### Step 8: Removal pressure (repair toward deletion)

11. Compute `remove_bias`

Includes penalties for:

* redundant talkers/listeners
* incorrect forwarding behavior
* looping switches
* dead-end or blank-start switches
* backward connections

Output meaning:

* higher value → stronger suggestion to remove that gene

---

### Step 9: Addition pressure (repair toward activation)

12. Compute `add_bias`

Encourages:

* missing talkers/listeners
* extending dead-end paths
* fixing blank-start switches

13. Filter unsafe activations

* prevents:

  * stray activations
  * backward connections
  * invalid talker/listener overlap

---

### Step 10: Encoding-specific correction logic

14. Branch on scheduling type (`self.scheduling_type`)

---

#### CASE A: PRIORITY encoding

15. Detect duplicate priorities

* `counts = (Y == Y).sum(...)`
* `conflict_mask = counts > 1`

16. Generate correction signal

* `gene_value_bias = conflict_mask`

---

#### CASE B: START_TIME encoding

##### Step 1: topology extraction

17. Load upstream/downstream structure

* `upstream_ports`
* `downstream_ports`
* `all_hops_costs`

---

##### Step 2: sequencing violations

18. Compute start-time propagation

* `enqueuing_start = upstream_start + hop_cost`

19. Detect causality violations

* downstream starts before arrival

20. Accumulate corrections

* `sequencing_bias_downstream`
* `sequencing_bias_upstream`

---

##### Step 3: interleaving violations

21. Compute arrival ordering differences

* `arr_diff`
* `dep_diff`

22. Detect FIFO violations

* `interleave = (arr_diff * dep_diff) < 0`

23. Compute directional pressure

* `pressure_up`
* `pressure_down`

24. Aggregate interleaving bias

* `interleaving_pressure`

---

##### Step 4: talker timing correction

25. Compute talker offset

* latest talker activation per stream

26. Detect late talkers

* `talker_bias`

---

##### Step 5: latency correction

27. Compute max latency violation

* compares end time vs allowed limit

28. Generate `latency_bias`

---

##### Step 6: macrocycle overlap correction

29. Compute phase

* `phase = Y % stream_intervals`

30. Detect cyclic overlap

* `delta` (modular difference)

31. Apply overlap mask

* `overlap`

32. Compute signed correction direction

* `signed_delta`

33. Compute overlap severity

* `overlap_depth`

---

##### Step 7: final gene correction tensor

34. Stack all correction signals

Includes:

* sequencing corrections
* talker timing corrections
* macrocycle overlap repulsion
* interleaving corrections
* latency corrections

---

#### CASE C: fallback (default)

35. Use simple activation-based bias

* `gene_value_bias = expanded_act`

---

### Step 11: Final filtering

36. Remove invalid corrections

* `remove_bias *= act`
* `add_bias *= (~act) & (adj_act == 0)`
* `gene_value_bias *= expanded_act`

---

### Output

* **Returns**

  * `(remove_bias, add_bias, gene_value_bias)`

  1. **remove_bias**

     * shape: (I, P, S)
     * indicates elements to deactivate

  2. **add_bias**

     * shape: (I, P, S)
     * indicates elements to activate

  3. **gene_value_bias**

     * shape: (I, ?, P, S) or (I, P, S)
     * encoding-dependent correction signals:

       * priority corrections OR
       * timing/scheduling adjustments

---

### Effects

* Does not modify input `Y`
* Produces **repair guidance signals** for evolutionary or optimization-based correction
* Encodes both:

  * structural fixes (topology-level)
  * temporal fixes (scheduling-level)
  * encoding-specific fixes (priority vs start-time)
