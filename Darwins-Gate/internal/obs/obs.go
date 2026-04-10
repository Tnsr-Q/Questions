// Package obs provides observability primitives for the Darwinian Gateway.
//
// Subpackages:
//
//	tracing  — trace/span ID generation
//	metrics  — Prometheus-compatible metric collectors and HTTP handler
//
// Usage:
//
//	import "github.com/tnsr-q/QFT-Engine/internal/obs"
//	import "github.com/tnsr-q/QFT-Engine/internal/obs/tracing"
//	import "github.com/tnsr-q/QFT-Engine/internal/obs/metrics"
//
//	// Generate unique IDs
//	mutationID := tracing.NewTraceID()
//
//	// Increment metrics
//	metrics.GatewayMetrics.IncSessionsStarted()
//
//	// Expose metrics via HTTP
//	mux.Handle("/metrics", metrics.DefaultCollector())
package obs

import (
	"github.com/tnsr-q/QFT-Engine/internal/obs/metrics"
	"github.com/tnsr-q/QFT-Engine/internal/obs/tracing"
)

// Re-export for convenience at the top level.
var (
	NewID     = tracing.NewTraceID
	NewSpanID = tracing.NewSpanID
)

// Global metric instances — safe for concurrent use from all goroutines.
var (
	GatewayMetrics = metrics.GatewayMetrics
	EBPFMapMetrics = metrics.EBPFMapMetrics
	SwarmMetrics   = metrics.SwarmMetrics
)

// DefaultCollector returns a Collector wired to the global metric singletons.
func DefaultCollector() *metrics.Collector {
	return metrics.DefaultCollector()
}
