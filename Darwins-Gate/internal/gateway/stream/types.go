package stream

import (
	"fmt"

	pb "github.com/tnsr-q/Questions/gen/go/darwinianv1"
)

type ModelTemplate struct {
	ModelID     string
	Preamble    []string
	Definitions []string
	Execution   []string
}

type BuildInput struct {
	ModelID         string
	HyperParameters map[string]string
	ChunkSizeHint   int32
}

type BuiltStream struct {
	Chunks []*pb.CodeChunk
}

func chunk(content string, t pb.ChunkType, seq uint64) *pb.CodeChunk {
	return &pb.CodeChunk{
		Content:    content,
		Type:       t,
		SequenceId: seq,
	}
}

func normalizeChunkSizeHint(v int32) int32 {
	if v <= 0 {
		return 4096
	}
	if v < 256 {
		return 256
	}
	if v > 64*1024 {
		return 64 * 1024
	}
	return v
}

func validateBuiltStream(bs *BuiltStream) error {
	if bs == nil {
		return fmt.Errorf("nil built stream")
	}
	if len(bs.Chunks) == 0 {
		return fmt.Errorf("no chunks generated")
	}
	var sawExec bool
	var last uint64
	for i, ch := range bs.Chunks {
		if ch == nil {
			return fmt.Errorf("nil chunk at index %d", i)
		}
		if i > 0 && ch.SequenceId <= last {
			return fmt.Errorf("non-monotonic sequence id at index %d", i)
		}
		last = ch.SequenceId
		if ch.Type == pb.ChunkType_CHUNK_TYPE_EXECUTION {
			sawExec = true
		}
	}
	if !sawExec {
		return fmt.Errorf("missing execution chunk")
	}
	return nil
}
