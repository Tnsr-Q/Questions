# Questions Monorepo

This monorepo hosts two major projects for building connected computational infrastructure: **JAX'D** and **Darwin's Gate**.

---

## 🔬 JAX'D

**JAX-based Differential Equations and Scientific Computing**

A Python-based scientific computing framework leveraging JAX for high-performance numerical computations, focusing on differential equations, Regge calculus, and quantum field theory applications.

### Technology Stack
- **Language**: Python 3.x
- **Core Framework**: JAX (Google's machine learning framework)
- **Dependencies**: NumPy, SciPy, PyTorch, DeepSpeed

### Project Structure
```
JAX'D/
├── src/                      # Source code
│   ├── hessian_jax.py       # Hessian computations with JAX
│   ├── regge_*.py           # Regge calculus solvers (bootstrap, vmap, pmap, shard_map)
│   ├── rge_solver.py        # Renormalization group equation solver
│   ├── spectral_*.py        # Spectral density and flow computations
│   ├── bootstrap_solver.py  # Bootstrap method implementations
│   ├── optimizer.py         # Optimization routines
│   ├── callbacks/           # Training/computation callbacks
│   ├── discovery/           # Discovery algorithms
│   ├── mesh/                # Mesh generation and handling
│   ├── spectral/            # Spectral methods
│   ├── tolerance/           # Tolerance checking
│   └── truth/               # Ground truth computations
├── tests/                   # Test suite
├── configs/                 # Configuration files
├── scripts/                 # Utility scripts
└── docker/                  # Docker configurations
```

### Key Features
- Hessian eigenvalue computations using Lanczos methods
- Regge calculus solvers with various parallelization strategies (vmap, pmap, shard_map)
- Renormalization group equation solvers
- Spectral flow and density analysis
- Bootstrap methods for quantum field theory
- BRST symmetry checking
- Unified topology computations

### Installation
```bash
pip install -r requirements.txt
```

### Use Cases
- Quantum field theory research
- Differential geometry computations
- Spectral analysis
- Reinforcement learning for mathematical conjectures
- High-performance scientific computing

---

## 🚪 Darwin's Gate

**Distributed Systems Infrastructure with eBPF**

A Go-based distributed systems platform featuring eBPF-powered network observability, intelligent routing, and swarm coordination capabilities. Implements a gateway architecture for managing complex distributed deployments.

### Technology Stack
- **Language**: Go 1.23+
- **Core Technologies**: eBPF, Protocol Buffers, Rust (embedded components)
- **Architecture**: Microservices with gRPC/Connect

### Project Structure
```
Darwins-Gate/
├── cmd/                                    # Command-line applications
│   ├── gatewayd/                          # Main gateway daemon
│   ├── ebpf-loaderd/                      # eBPF program loader daemon
│   ├── ebpf-maps/                         # eBPF map management
│   ├── mutation-firewalld/                # Dynamic firewall mutation service
│   └── overseer-coordinatord/             # Swarm coordination overseer
├── internal/                              # Internal packages
│   ├── swarmbridge/                       # Swarm communication bridge
│   └── ...                                # Other internal components
├── bpf/                                   # eBPF programs and maps
├── proto/                                 # Protocol Buffer definitions
├── gen/                                   # Generated code (protobuf, etc.)
├── swarm/                                 # Swarm intelligence implementation
├── web/                                   # Web interfaces
├── rust/                                  # Rust components
├── deploy/                                # Deployment configurations
├── docs/                                  # Documentation
├── test/                                  # Test suites
└── tools/                                 # Development tools
```

### Key Components

#### Daemons
- **gatewayd**: Main gateway service for routing and coordination
- **ebpf-loaderd**: Manages eBPF program lifecycle
- **mutation-firewalld**: Dynamic firewall rule engine
- **overseer-coordinatord**: Coordinates swarm behavior

#### Features
- eBPF-based network observability and filtering
- Dynamic traffic routing and mutation
- Protocol Buffer-based service definitions
- Swarm coordination and intelligence
- Multi-language support (Go, Rust)
- Web-based management interface

### Go Workspace Setup
Darwin's Gate uses Go modules. To work with this project in a Go workspace:

```bash
# Initialize workspace (from monorepo root)
go work init ./Darwins-Gate

# Verify workspace
go work use -r ./Darwins-Gate
```

### Building
```bash
cd Darwins-Gate
make build  # See makefile.txt for build commands
```

### Use Cases
- Distributed system observability
- Intelligent network routing
- Service mesh implementations
- Swarm-based coordination
- Dynamic security policy enforcement

---

## 🔗 Connected Infrastructure

This monorepo is designed to build **both environments** and create the **connected infrastructure** for JAX'D and Darwin's Gate, enabling:

- **JAX'D** to provide computational backends for Darwin's Gate decision-making
- **Darwin's Gate** to provide distributed infrastructure for JAX'D workloads
- Shared data pipelines and protocol definitions
- Unified deployment and orchestration
- Cross-project observability and monitoring

### Monorepo Structure

```
Questions/
├── JAX'D/                   # Python/JAX scientific computing
├── Darwins-Gate/            # Go/eBPF distributed systems
├── assets/                  # Shared assets
├── requirements.txt         # Python dependencies (JAX'D)
└── README.md               # This file
```

### Go Workspace Configuration

For working with Darwin's Gate Go modules across the monorepo:

```bash
# Create workspace from root
go work init

# Add Darwin's Gate
go work use ./Darwins-Gate

# If Darwin's Gate has nested modules, add them
go work use -r ./Darwins-Gate
```

The `go.work` file will track all Go modules in the monorepo, making it easy to work on Darwin's Gate components with proper module resolution.

---

## 🚀 Getting Started

### Prerequisites
- **For JAX'D**: Python 3.8+, CUDA (optional, for GPU acceleration)
- **For Darwin's Gate**: Go 1.23+, Linux kernel with eBPF support

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Questions
   ```

2. **Set up JAX'D**
   ```bash
   pip install -r requirements.txt
   cd JAX\'D
   python -m pytest tests/
   ```

3. **Set up Darwin's Gate**
   ```bash
   cd Darwins-Gate
   go work init .
   go build ./cmd/gatewayd
   ```

---

## 📦 Dependencies

### JAX'D Dependencies
See `requirements.txt` for complete Python package list including:
- JAX (CPU/GPU)
- NumPy, SciPy
- PyTorch, PyTorch Lightning
- DeepSpeed
- TensorBoard

### Darwin's Gate Dependencies
Managed via Go modules. See individual `go.mod` files in Darwin's Gate subdirectories.

---

## 🤝 Contributing

When contributing to this monorepo:
1. Maintain clear separation between JAX'D and Darwin's Gate codebases
2. Use appropriate language conventions (Python for JAX'D, Go for Darwin's Gate)
3. Update this README when adding new major components
4. Ensure Go workspace configuration remains valid

---

## 📄 License

See individual LICENSE files in project directories.
