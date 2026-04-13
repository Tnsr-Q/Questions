package protocdoc

import (
	"fmt"
	"io"
	"os"
	"strings"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/pluginpb"
)

// Run reads a CodeGeneratorRequest from stdin, generates documentation, and
// writes a CodeGeneratorResponse to stdout. This is the main entry point for
// the protoc plugin.
func Run(stdin io.Reader, stdout io.Writer) error {
	input, err := io.ReadAll(stdin)
	if err != nil {
		return fmt.Errorf("reading input: %w", err)
	}

	var req pluginpb.CodeGeneratorRequest
	if err := proto.Unmarshal(input, &req); err != nil {
		return fmt.Errorf("unmarshalling request: %w", err)
	}

	format, outFile := parseParams(req.GetParameter())

	// Build a set of files to generate docs for.
	genFiles := make(map[string]bool, len(req.GetFileToGenerate()))
	for _, f := range req.GetFileToGenerate() {
		genFiles[f] = true
	}

	var docs []ProtoFile
	for _, fd := range req.GetProtoFile() {
		if !genFiles[fd.GetName()] {
			continue
		}
		docs = append(docs, ParseFile(fd))
	}

	var buf strings.Builder
	if err := Render(&buf, docs, format); err != nil {
		return fmt.Errorf("rendering: %w", err)
	}

	content := buf.String()
	var resp pluginpb.CodeGeneratorResponse
	resp.File = append(resp.File, &pluginpb.CodeGeneratorResponse_File{
		Name:    &outFile,
		Content: &content,
	})

	out, err := proto.Marshal(&resp)
	if err != nil {
		return fmt.Errorf("marshalling response: %w", err)
	}

	if _, err := stdout.Write(out); err != nil {
		return fmt.Errorf("writing output: %w", err)
	}
	return nil
}

// RunFromOS is a convenience wrapper that uses os.Stdin and os.Stdout.
func RunFromOS() {
	if err := Run(os.Stdin, os.Stdout); err != nil {
		errMsg := err.Error()
		var resp pluginpb.CodeGeneratorResponse
		resp.Error = &errMsg
		out, _ := proto.Marshal(&resp)
		os.Stdout.Write(out) //nolint:errcheck
		os.Exit(1)
	}
}

// parseParams extracts format and output filename from the protoc parameter string.
// Expected format: "format,filename" e.g. "markdown,docs.md" or just "markdown".
func parseParams(param string) (Format, string) {
	format := FormatMarkdown
	outFile := "docs.md"

	if param == "" {
		return format, outFile
	}

	parts := strings.SplitN(param, ",", 2)
	if len(parts) >= 1 {
		if f, err := ParseFormat(parts[0]); err == nil {
			format = f
		}
	}
	if len(parts) >= 2 && parts[1] != "" {
		outFile = parts[1]
	} else {
		// Set default filename based on format.
		switch format {
		case FormatHTML:
			outFile = "docs.html"
		case FormatJSON:
			outFile = "docs.json"
		default:
			outFile = "docs.md"
		}
	}

	return format, outFile
}
