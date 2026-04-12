package main

import (
	"fmt"
	"os"
)

func main() {
	// TODO: Implement mutation-firewalld daemon
	// This daemon coordinates firewall rule mutations with the Python Overseer
	// and the eBPF router.
	fmt.Fprintln(os.Stderr, "mutation-firewalld: not yet implemented")
	os.Exit(1)
}
