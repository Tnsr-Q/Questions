####internal/ebpf/objects/generate.go

Assuming you already have router_tc.c, this is the generator line:

package objects

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang -cflags "-O2 -g -Wall -Werror -I../../../bpf/include" RouterTC ../../../bpf/tc/router_tc.c -- -target bpf
What this expects from bpf2go

Your generated object struct should look roughly like this:

RoutePackets for the program
RoutingMap for the map
StatsMap for the counters map

That corresponds to:

SEC("tc") int route_packets(...)
routing_map
stats_map

from the C side.

If your generated names differ slightly, adjust these lines in loader.go:

r.Objs.RoutePackets
r.Objs.RoutingMap
r.Objs.StatsMap
Build / run flow
Generate objects
go generate ./internal/ebpf/objects
Build loader
go build ./cmd/ebpf-loaderd
Run loader
sudo ./ebpf-loaderd -iface eth0 -dir egress

That should:

create /sys/fs/bpf/tensorq
pin:
/sys/fs/bpf/tensorq/routing_map
/sys/fs/bpf/tensorq/stats_map
ensure clsact
attach the BPF filter
A few things I would immediately tighten
1. Pin-path restart behavior

This version does:

_ = os.Remove(routingMapPin)
_ = os.Remove(statsMapPin)

That is practical for a dev bootstrap, but a more production-safe version would:

try LoadPinnedMap first
reuse if compatible
only replace on explicit flag like -replace-pins
2. Filter identity

Right now it uses:

Handle: 1
Priority: 1
Name: tensorq-router-tc

That is enough for a controlled environment. In a larger host, make those configurable.

3. Qdisc lifecycle

This code ensures clsact exists, but does not remove it on shutdown. That is the correct default. clsact is infrastructure, not process-owned state.

4. Privileges

This must run privileged enough to:

manage qdiscs/filters
load BPF programs
pin maps in bpffs

So in Kubernetes this belongs in a privileged DaemonSet or similarly constrained component, not in a public-facing service.
