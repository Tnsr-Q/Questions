package cache

import (
	"context"

	"github.com/tnsr-q/Questions/internal/resolver"
)

type Noop struct{}

func NewNoop() *Noop {
	return &Noop{}
}

func (n *Noop) Resolve(ctx context.Context, targetIP string, macHint string) (resolver.Resolution, error) {
	if macHint != "" {
		return resolver.Resolution{
			NextHopMAC: macHint,
			Resolved:   true,
		}, nil
	}
	return resolver.Resolution{
		Resolved: false,
	}, nil
}