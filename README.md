# TSN Optimization Platform Architecture

The platform models and optimizes TSN (Time-Sensitive Networking) scenarios through a clear separation of concerns across domain modeling, transformation, and optimization representation.

---

## Core Domain Model

At the center is the `Usecase`, which represents a complete TSN instance. It includes:

- A port-aware network topology (graph-based representation)
- Traffic streams with placement and routing constraints
- Scheduling state and derived network properties

The `Usecase` acts as the **single source of truth**, responsible for managing the lifecycle of a TSN scenario, including:

- Stream addition/removal
- Link updates
- Schedule application and removal
- Serialization/deserialization

---

## Construction Layer

Two dedicated handlers are responsible for building and maintaining the domain instance.

### TopologyHandler
Responsible for:
- Constructing the network graph
- Managing TSN-specific metadata (ports, hosts, switches)
- Adding/removing links

### TrafficHandler
Responsible for:
- Generating traffic streams
- Embedding streams into the topology
- Managing routing and placement constraints

---

## Optimization Representation Layer

### OptimizationModelHandler

This layer transforms a `Usecase` into an optimization-ready representation. It does **not solve the problem**, but prepares all structures required for optimization algorithms.

It builds and manages:

- Configuration matrices (port × stream decision space)
- Stream-specific pruned graphs (custom graphs)
- Search space representations
- Latency and slack tensors
- Constraint-aware encodings

This abstraction allows optimization methods to operate on a clean, numerical representation of the problem without interacting directly with the domain model.

---

## Optimization Layer (External)

Optimization algorithms operate independently from the core system. Examples include:

- Genetic Algorithms
- Integer Linear Programming (ILP)
- Reinforcement Learning agents
- Heuristic search methods

These solvers consume the `OptimizationModelHandler` output and produce schedules.

---

## Execution Flow

1. A `Usecase` is created or loaded
2. Topology and traffic are generated via their handlers
3. `OptimizationModelHandler` builds the optimization model
4. A solver computes a schedule
5. The schedule is applied back to the `Usecase`

---

## Architecture Overview
Usecase
│
├── TopologyHandler
├── TrafficHandler
│
└── OptimizationModelHandler (previous tsnSchedulingProblem)
        ├── config matrices
        ├── slack / search space
        └── custom graphs

External Solvers
        → operate on optimization model
        → output schedules


---

## Key Design Principle

The system separates:

- Domain modeling
- Optimization representation
- Optimization solving

This enables modularity, extensibility, and independent development of scheduling algorithms.
