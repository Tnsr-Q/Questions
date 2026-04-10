package objects

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang -cflags "-O2 -g -Wall -Werror -I../../../bpf/include" RouterTC ../../../bpf/tc/router_tc.c -- -target bpf
