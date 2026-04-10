package resolver

type Resolution struct {
	NextHopMAC string
	EgressIf   string
	Resolved   bool
}