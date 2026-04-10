package resolver

import "context"

type Noop struct{}

func NewNoop() *Noop {
	return &Noop{}
}

func (n *Noop) Resolve(ctx context.Context, targetIP string, macHint string) (Resolution, error) {
	if macHint != "" {
		return Resolution{
			NextHopMAC: macHint,
			Resolved:   true,
		}, nil
	}
	return Resolution{
		Resolved: false,
	}, nil
}