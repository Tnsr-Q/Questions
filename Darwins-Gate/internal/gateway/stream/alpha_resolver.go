package stream

import (
	"context"
	swarmbridge "github.com/tnsr-q/Questions/internal/swarmbridge/client"
	pb "github.com/tnsr-q/Questions/gen/go/darwinianv1"
)

type AlphaResolver struct{ bridge *swarmbridge.Client }

func NewAlphaResolver(b *swarmbridge.Client) *AlphaResolver { return &AlphaResolver{bridge: b} }

func (r *AlphaResolver) GetAlpha(ctx context.Context, modelID string) (*pb.CodeChunk, error) {
	alphaURI, err := r.bridge.GetConsensus(ctx) // your client.go
	if err != nil { return nil, err }
	code, err := r.bridge.CallSolvePhysics(alphaURI, modelID) // extend client.go if needed
	if err != nil { return nil, err }
	return &pb.CodeChunk{Content: code, SequenceId: 1, Type: pb.ChunkType_CHUNK_TYPE_EXECUTION}, nil
}
