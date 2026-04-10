package tracing

import "crypto/rand"

var hexDigits = []byte("0123456789abcdef")

// NewTraceID returns a 32-hex-character string (16 random bytes).
// Suitable for mutation_id, proposal_id, batch_id, observability_id.
func NewTraceID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	out := make([]byte, 32)
	for i := 0; i < 16; i++ {
		out[i*2] = hexDigits[b[i]>>4]
		out[i*2+1] = hexDigits[b[i]&0x0f]
	}
	return string(out)
}

// NewSpanID returns a 16-hex-character string (8 random bytes).
// Suitable for correlating a single RPC or kernel operation.
func NewSpanID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	out := make([]byte, 16)
	for i := 0; i < 8; i++ {
		out[i*2] = hexDigits[b[i]>>4]
		out[i*2+1] = hexDigits[b[i]&0x0f]
	}
	return string(out)
}
