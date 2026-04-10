package swarmbridge

import (
	"context"
	"fmt"
	"time"

	"github.com/tnsr-q/Questions/internal/obs/metrics"
)

// LoopConfig configures the background polling loops.
type LoopConfig struct {
	TelemetryInterval time.Duration
	ProposalInterval  time.Duration
	NodeID            string
	VIPs              []string // VIPs to poll for proposals
}

// DefaultLoopConfig returns sensible defaults.
func DefaultLoopConfig() LoopConfig {
	return LoopConfig{
		TelemetryInterval: TelemetryInterval,
		ProposalInterval:  ProposalInterval,
		NodeID:            "gatewayd-0",
		VIPs:              []string{},
	}
}

// TelemetryFn produces a telemetry batch. Called periodically by the loop.
type TelemetryFn func(ctx context.Context) (TelemetryBatch, error)

// ProposalHandler is called when a proposal is fetched. Return true to
// forward to the mutation firewall, false to skip (unhealthy quorum, etc).
type ProposalHandler func(ctx context.Context, proposal *AlphaRouteProposal) (forwarded bool, err error)

// StartLoops runs background goroutines for:
//   - Leader refresh (every TelemetryInterval / 2)
//   - Telemetry push (every TelemetryInterval)
//   - Proposal polling (every ProposalInterval)
//
// Returns a cancel function and error channel.
// The error channel is non-blocking — if it fills up, errors are dropped.
func (c *Client) StartLoops(ctx context.Context, cfg LoopConfig, telemetryFn TelemetryFn, proposalHandler ProposalHandler) (cancel context.CancelFunc, errs <-chan error) {
	ctx, cancel = context.WithCancel(ctx)
	errCh := make(chan error, 64)

	// ── Leader refresh loop ──────────────────────────────────────────
	go func() {
		ticker := time.NewTicker(cfg.TelemetryInterval / 2)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			leader, err := c.GetLeader(ctx)
			if err != nil {
				select {
				case errCh <- fmt.Errorf("leader refresh: %w", err):
				default:
				}
				metrics.SwarmMetrics.IncLeaderFetchErrors()
				continue
			}
			metrics.SwarmMetrics.IncLeaderFetches()

			if leader.QuorumStatus == "HEALTHY" {
				metrics.SwarmMetrics.QuorumHealthy.Store(1)
			} else {
				metrics.SwarmMetrics.QuorumHealthy.Store(0)
			}
			metrics.SwarmMetrics.CurrentEpoch.Store(leader.Epoch)
		}
	}()

	// ── Telemetry push loop ──────────────────────────────────────────
	go func() {
		ticker := time.NewTicker(cfg.TelemetryInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			if telemetryFn == nil {
				continue
			}

			batch, err := telemetryFn(ctx)
			if err != nil {
				select {
				case errCh <- fmt.Errorf("telemetry batch: %w", err):
				default:
				}
				metrics.SwarmMetrics.IncTelemetryErrors()
				continue
			}

			if len(batch.Samples) == 0 {
				continue // nothing to push
			}

			if _, err := c.PostTelemetry(ctx, batch); err != nil {
				select {
				case errCh <- fmt.Errorf("post telemetry: %w", err):
				default:
				}
				metrics.SwarmMetrics.IncTelemetryErrors()
				continue
			}
			metrics.SwarmMetrics.IncTelemetryBatches()
		}
	}()

	// ── Proposal poll loop ───────────────────────────────────────────
	go func() {
		if proposalHandler == nil || len(cfg.VIPs) == 0 {
			return
		}

		ticker := time.NewTicker(cfg.ProposalInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			// Skip if quorum is unhealthy or leader lease is invalid
			if !c.IsQuorumHealthy() || !c.IsLeaderLeaseValid() {
				metrics.SwarmMetrics.IncProposalsSkipped()
				continue
			}

			for _, vip := range cfg.VIPs {
				resp, err := c.GetAlphaProposal(ctx, vip)
				if err != nil {
					select {
					case errCh <- fmt.Errorf("get proposal vip=%s: %w", vip, err):
					default:
					}
					continue
				}
				metrics.SwarmMetrics.IncProposalsFetched()

				if resp.Proposal == nil {
					continue // no proposal available
				}

				forwarded, err := proposalHandler(ctx, resp.Proposal)
				if err != nil {
					select {
					case errCh <- fmt.Errorf("proposal handler vip=%s: %w", vip, err):
					default:
					}
					continue
				}
				if forwarded {
					metrics.SwarmMetrics.IncProposalsForwarded()
				} else {
					metrics.SwarmMetrics.IncProposalsSkipped()
				}
			}
		}
	}()

	return cancel, errCh
}
