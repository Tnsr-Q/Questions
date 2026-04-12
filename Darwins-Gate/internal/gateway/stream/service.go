package stream

import (
	"context"
	"fmt"

	pb "github.com/tnsr-q/Questions/gen/go/darwinianv1"
)

type Service struct {
	templates map[string]ModelTemplate
}

func NewService(templates map[string]ModelTemplate) *Service {
	if templates == nil {
		templates = DefaultTemplates()
	}
	return &Service{templates: templates}
}

func (s *Service) Build(ctx context.Context, in BuildInput) (*BuiltStream, error) {
	tpl, ok := s.templates[in.ModelID]
	if !ok {
		return nil, fmt.Errorf("unknown model %q", in.ModelID)
	}
	_ = normalizeChunkSizeHint(in.ChunkSizeHint)

	seq := uint64(1)
	chunks := make([]*pb.CodeChunk, 0, 3)

	preamble := joinLines(tpl.Preamble) + "\n" + injectParams(in.HyperParameters)
	chunks = append(chunks, chunk(preamble, pb.ChunkType_CHUNK_TYPE_PREAMBLE, seq))
	seq++

	defs := joinLines(tpl.Definitions)
	chunks = append(chunks, chunk(defs, pb.ChunkType_CHUNK_TYPE_DEFINITION, seq))
	seq++

	exec := joinLines(tpl.Execution)
	chunks = append(chunks, chunk(exec, pb.ChunkType_CHUNK_TYPE_EXECUTION, seq))

	out := &BuiltStream{Chunks: chunks}
	if err := validateBuiltStream(out); err != nil {
		return nil, err
	}
	return out, nil
}