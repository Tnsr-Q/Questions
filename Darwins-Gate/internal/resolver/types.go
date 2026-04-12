package resolver

// Resolution represents the result of an IP-to-MAC address resolution.
type Resolution struct {
	NextHopMAC string // The MAC address of the next hop
	EgressIf   string // The egress interface name
	Resolved   bool   // Whether the resolution was successful
}
