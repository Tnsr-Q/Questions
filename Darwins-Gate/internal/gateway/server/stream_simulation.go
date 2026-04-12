package server

import (
	"context"
	"fmt"

	"connectrpc.com/connect"
	pb "github.com/tnsr-q/Questions/gen/go/darwinianv1"
	"github.com/tnsr-q/Questions/internal/gateway/stream"
	"github.com/tnsr-q/Questions/internal/obs/metrics"
	"github.com/tnsr-q/Questions/internal/obs/tracing"
)

type StreamServer struct {
	streamSvc *stream.Service
	metrics   *metrics.StreamMetrics
}

func NewStreamServer(streamSvc *stream.Service, m *metrics.StreamMetrics) *StreamServer {
	if streamSvc == nil {
		streamSvc = stream.NewService(nil)
	}
	if m == nil {
		m = &metrics.StreamMetrics{}
	}
	return &StreamServer{
		streamSvc: streamSvc,
		metrics:   m,
	}
}

func (s *StreamServer) StreamSimulation(
	ctx context.Context,
	req *connect.Request[pb.SimRequest],
	streamer *connect.ServerStream[pb.CodeChunk],
) error {
	_ = tracing.NewTraceID() // keep for future correlation/log injection
	s.metrics.IncSessionsStarted()

	built, err := s.streamSvc.Build(ctx, stream.BuildInput{
		ModelID:         req.Msg.ModelId,
		HyperParameters: req.Msg.HyperParameters,
		ChunkSizeHint:   req.Msg.ChunkSizeHint,
	})
	if err != nil {
		s.metrics.IncSessionsFailed()
		return connect.NewError(connect.CodeInvalidArgument, fmt.Errorf("build stream: %w", err))
	}

	for _, ch := range built.Chunks {
		if err := streamer.Send(ch); err != nil {
			s.metrics.IncSessionsFailed()
			return connect.NewError(connect.CodeUnknown, fmt.Errorf("send chunk seq=%d: %w", ch.SequenceId, err))
		}
	}

	s.metrics.AddChunksEmitted(len(built.Chunks))
	return nil
}