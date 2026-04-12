# 🚪 Darwin's Gate

**Distributed Systems Infrastructure with eBPF**

A Go-based distributed systems platform featuring eBPF-powered network observability, intelligent routing, and swarm coordination capabilities. Implements a gateway architecture for managing complex distributed deployments.

## Technology Stack

- **Language**: Go 1.23+
- **Core Technologies**: eBPF, Protocol Buffers, Rust (embedded components)
- **Architecture**: Microservices with gRPC/Connect

## Project Structure

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

## Key Components

### Daemons

- **gatewayd**: Main gateway service for routing and coordination
- **ebpf-loaderd**: Manages eBPF program lifecycle
- **mutation-firewalld**: Dynamic firewall rule engine
- **overseer-coordinatord**: Coordinates swarm behavior

### Features

- eBPF-based network observability and filtering
- Dynamic traffic routing and mutation
- Protocol Buffer-based service definitions
- Swarm coordination and intelligence
- Multi-language support (Go, Rust)
- Web-based management interface

## Go Workspace Setup

Darwin's Gate uses Go modules. To work with this project in a Go workspace:

```bash
# Initialize workspace (from monorepo root)
go work init ./Darwins-Gate

# Verify workspace
go work use -r ./Darwins-Gate
```

## Building

```bash
cd Darwins-Gate
make build  # See makefile.txt for build commands
```

## Use Cases

- Distributed system observability
- Intelligent network routing
- Service mesh implementations
- Swarm-based coordination
- Dynamic security policy enforcement
