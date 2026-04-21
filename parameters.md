# Parameters

## topology_config

- **numhosts** = the end stations that can send or receive traffic  
- **numswitchs** = the nodes forwarding the traffic between the end stations  
- **portspeed** = the speed of traffic transmission at the egress ports of the switches and end stations  
- **procdelay** = the time needed by the switches to process a traffic frame  
- **propdelay** = the time needed by the traffic to propagate in the links from an egress port of a switch to an ingress port of the next switch  
- **linkBw** = the bandwidth of the link  
- **queuecountperport** = the number of queues available in the egress ports  
- **graphConfig** = dictionary defining the type and parameters of the network graph. There are three mutually exclusive possibilities:  
  1. Example 1 – Fat-tree  
    - `typ = 'ft'`  
    - `k` = parameter of the fat-tree  
  2. Example 2 – Erdos–Renyi graph  
    - `typ = 'er'`  
    - `prob` = probability of connection between nodes  
    - `numcoreswitches` = number of core switches  
    - `Minnodedegreepersw` = minimum number of connections per switch
  3. Example 3 – Barabasi–Albert graph
    - `typ = 'ba'`
    - `numEdgN` = number of edge switches
    - `numcoreswitches` = number of core switches
    - `minnodedegreepersw` = minimum number of connections per switch

---

## streams_config

- **numofstreamsperinstance** = number of streams  
- **maxstrpernode** = maximum number of streams that can be sourced from a single node  
- **period_range** = cycle of the streams — dictionary keyed `{ "start": …, "stop": …, "step": … }`  
- **numberframes_range** = number of frames per stream within a cycle — dictionary keyed `{ "start": …, "stop": … }`  
- **datasize_range** = size of a frame — dictionary keyed `{ "start": …, "stop": … }`  
- **maxlatency_range** = maximum latency constraint of a stream — list of comma-separated floats  
- **numplacestalker** = number of end stations where the talker (stream sender) can possibly be placed  
- **numplaceslistener** = number of end stations where the listener (stream receiver) can possibly be placed  
- **numlistenersperstr** = maximum number of listeners that can be placed in the same node  
- **routing** = Dictionary defining the routing configuration.  
  - **fixPath** = Boolean indicating pre-routing. If `True` and `numplacestalker = 1`, streams will be pre-routed.  
  - **shortestPath** = Boolean defining routing method. If `True` → Dijkstra shortest path else `False` → random path  
- **generation** = Defines experiment scenario generation.  
  - **incrementation** = Boolean defining whether streams used in an experiment are included in the next experiment  
    - If `True`:  
      - experiment `0` has 1 stream  
      - experiment `i+1` contains streams from experiment `i` plus **1 new stream**
- **expiration** = Defines the life span of streams.
  - **finite_duration** = Boolean indicating whether the stream is finite or continuous during an experiment  
  - **start** = minimum number of cycles a finite stream can live  
  - **stop** = maximum number of cycles a finite stream can live