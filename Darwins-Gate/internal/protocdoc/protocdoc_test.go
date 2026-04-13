package protocdoc

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/pluginpb"
)

// testFileDescriptor returns a FileDescriptorProto that resembles the
// darwinian_gateway.proto in the repository.
func testFileDescriptor() *descriptorpb.FileDescriptorProto {
	syntax := "proto3"
	name := "tensorq/darwinian/v1/darwinian_gateway.proto"
	pkg := "tensorq.darwinian.v1"
	goPkg := "github.com/tnsr-q/QFT-engine/gen/go/darwinianv1"

	// Enum: ChunkType
	chunkTypeEnum := &descriptorpb.EnumDescriptorProto{
		Name: proto.String("ChunkType"),
		Value: []*descriptorpb.EnumValueDescriptorProto{
			{Name: proto.String("CHUNK_TYPE_UNSPECIFIED"), Number: proto.Int32(0)},
			{Name: proto.String("CHUNK_TYPE_PREAMBLE"), Number: proto.Int32(1)},
			{Name: proto.String("CHUNK_TYPE_DEFINITION"), Number: proto.Int32(2)},
			{Name: proto.String("CHUNK_TYPE_EXECUTION"), Number: proto.Int32(3)},
		},
	}

	// Enum: RouteStatus
	routeStatusEnum := &descriptorpb.EnumDescriptorProto{
		Name: proto.String("RouteStatus"),
		Value: []*descriptorpb.EnumValueDescriptorProto{
			{Name: proto.String("ROUTE_STATUS_UNSPECIFIED"), Number: proto.Int32(0)},
			{Name: proto.String("ROUTE_STATUS_APPLIED"), Number: proto.Int32(1)},
			{Name: proto.String("ROUTE_STATUS_STAGED"), Number: proto.Int32(2)},
			{Name: proto.String("ROUTE_STATUS_REJECTED"), Number: proto.Int32(3)},
		},
	}

	// Message: SimRequest
	simRequest := &descriptorpb.DescriptorProto{
		Name: proto.String("SimRequest"),
		Field: []*descriptorpb.FieldDescriptorProto{
			{
				Name:   proto.String("model_id"),
				Number: proto.Int32(1),
				Type:   descriptorpb.FieldDescriptorProto_TYPE_STRING.Enum(),
				Label:  descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(),
			},
			{
				Name:   proto.String("chunk_size_hint"),
				Number: proto.Int32(3),
				Type:   descriptorpb.FieldDescriptorProto_TYPE_INT32.Enum(),
				Label:  descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(),
			},
		},
	}

	// Message: CodeChunk
	codeChunk := &descriptorpb.DescriptorProto{
		Name: proto.String("CodeChunk"),
		Field: []*descriptorpb.FieldDescriptorProto{
			{
				Name:   proto.String("content"),
				Number: proto.Int32(1),
				Type:   descriptorpb.FieldDescriptorProto_TYPE_STRING.Enum(),
				Label:  descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(),
			},
			{
				Name:     proto.String("type"),
				Number:   proto.Int32(2),
				Type:     descriptorpb.FieldDescriptorProto_TYPE_ENUM.Enum(),
				TypeName: proto.String(".tensorq.darwinian.v1.ChunkType"),
				Label:    descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(),
			},
			{
				Name:   proto.String("sequence_id"),
				Number: proto.Int32(3),
				Type:   descriptorpb.FieldDescriptorProto_TYPE_UINT64.Enum(),
				Label:  descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(),
			},
		},
	}

	// Message: RouteMutation
	routeMutation := &descriptorpb.DescriptorProto{
		Name: proto.String("RouteMutation"),
		Field: []*descriptorpb.FieldDescriptorProto{
			{
				Name:   proto.String("mutation_id"),
				Number: proto.Int32(1),
				Type:   descriptorpb.FieldDescriptorProto_TYPE_STRING.Enum(),
				Label:  descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(),
			},
			{
				Name:   proto.String("virtual_ip"),
				Number: proto.Int32(2),
				Type:   descriptorpb.FieldDescriptorProto_TYPE_STRING.Enum(),
				Label:  descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(),
			},
		},
	}

	// Message: RouteAck
	routeAck := &descriptorpb.DescriptorProto{
		Name: proto.String("RouteAck"),
		Field: []*descriptorpb.FieldDescriptorProto{
			{
				Name:     proto.String("status"),
				Number:   proto.Int32(1),
				Type:     descriptorpb.FieldDescriptorProto_TYPE_ENUM.Enum(),
				TypeName: proto.String(".tensorq.darwinian.v1.RouteStatus"),
				Label:    descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(),
			},
			{
				Name:   proto.String("status_message"),
				Number: proto.Int32(2),
				Type:   descriptorpb.FieldDescriptorProto_TYPE_STRING.Enum(),
				Label:  descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL.Enum(),
			},
		},
	}

	// Service: CortexGateway
	cortexGateway := &descriptorpb.ServiceDescriptorProto{
		Name: proto.String("CortexGateway"),
		Method: []*descriptorpb.MethodDescriptorProto{
			{
				Name:            proto.String("StreamSimulation"),
				InputType:       proto.String(".tensorq.darwinian.v1.SimRequest"),
				OutputType:      proto.String(".tensorq.darwinian.v1.CodeChunk"),
				ServerStreaming: proto.Bool(true),
				ClientStreaming: proto.Bool(false),
			},
			{
				Name:            proto.String("UpdateAlphaRoute"),
				InputType:       proto.String(".tensorq.darwinian.v1.RouteMutation"),
				OutputType:      proto.String(".tensorq.darwinian.v1.RouteAck"),
				ServerStreaming: proto.Bool(false),
				ClientStreaming: proto.Bool(false),
			},
		},
	}

	return &descriptorpb.FileDescriptorProto{
		Name:    &name,
		Package: &pkg,
		Syntax:  &syntax,
		Options: &descriptorpb.FileOptions{
			GoPackage: &goPkg,
		},
		Service:     []*descriptorpb.ServiceDescriptorProto{cortexGateway},
		MessageType: []*descriptorpb.DescriptorProto{simRequest, codeChunk, routeMutation, routeAck},
		EnumType:    []*descriptorpb.EnumDescriptorProto{chunkTypeEnum, routeStatusEnum},
	}
}

func TestParseFile(t *testing.T) {
	fd := testFileDescriptor()
	pf := ParseFile(fd)

	if pf.Name != "tensorq/darwinian/v1/darwinian_gateway.proto" {
		t.Errorf("Name = %q, want darwinian_gateway.proto", pf.Name)
	}
	if pf.Package != "tensorq.darwinian.v1" {
		t.Errorf("Package = %q, want tensorq.darwinian.v1", pf.Package)
	}
	if pf.Syntax != "proto3" {
		t.Errorf("Syntax = %q, want proto3", pf.Syntax)
	}
	if len(pf.Services) != 1 {
		t.Fatalf("Services count = %d, want 1", len(pf.Services))
	}

	svc := pf.Services[0]
	if svc.Name != "CortexGateway" {
		t.Errorf("Service.Name = %q, want CortexGateway", svc.Name)
	}
	if len(svc.Methods) != 2 {
		t.Fatalf("Methods count = %d, want 2", len(svc.Methods))
	}

	// StreamSimulation
	m := svc.Methods[0]
	if m.Name != "StreamSimulation" {
		t.Errorf("Method[0].Name = %q, want StreamSimulation", m.Name)
	}
	if m.RequestType != "SimRequest" {
		t.Errorf("Method[0].RequestType = %q, want SimRequest", m.RequestType)
	}
	if m.ResponseType != "CodeChunk" {
		t.Errorf("Method[0].ResponseType = %q, want CodeChunk", m.ResponseType)
	}
	if !m.ServerStreaming {
		t.Error("StreamSimulation should be server streaming")
	}
	if m.ClientStreaming {
		t.Error("StreamSimulation should not be client streaming")
	}

	// UpdateAlphaRoute
	m2 := svc.Methods[1]
	if m2.Name != "UpdateAlphaRoute" {
		t.Errorf("Method[1].Name = %q, want UpdateAlphaRoute", m2.Name)
	}
	if m2.ServerStreaming || m2.ClientStreaming {
		t.Error("UpdateAlphaRoute should be unary")
	}

	if len(pf.Messages) != 4 {
		t.Fatalf("Messages count = %d, want 4", len(pf.Messages))
	}
	if pf.Messages[0].Name != "SimRequest" {
		t.Errorf("Messages[0].Name = %q, want SimRequest", pf.Messages[0].Name)
	}
	if len(pf.Messages[0].Fields) != 2 {
		t.Fatalf("SimRequest fields count = %d, want 2", len(pf.Messages[0].Fields))
	}

	if len(pf.Enums) != 2 {
		t.Fatalf("Enums count = %d, want 2", len(pf.Enums))
	}
	if pf.Enums[0].Name != "ChunkType" {
		t.Errorf("Enums[0].Name = %q, want ChunkType", pf.Enums[0].Name)
	}
	if len(pf.Enums[0].Values) != 4 {
		t.Errorf("ChunkType values count = %d, want 4", len(pf.Enums[0].Values))
	}

	// Options
	if len(pf.Options) != 1 {
		t.Fatalf("Options count = %d, want 1", len(pf.Options))
	}
	if pf.Options[0].Name != "go_package" {
		t.Errorf("Options[0].Name = %q, want go_package", pf.Options[0].Name)
	}
}

func TestParseFormat(t *testing.T) {
	tests := []struct {
		input string
		want  Format
		err   bool
	}{
		{"html", FormatHTML, false},
		{"HTML", FormatHTML, false},
		{"markdown", FormatMarkdown, false},
		{"md", FormatMarkdown, false},
		{"json", FormatJSON, false},
		{"JSON", FormatJSON, false},
		{"xml", "", true},
	}
	for _, tt := range tests {
		got, err := ParseFormat(tt.input)
		if tt.err && err == nil {
			t.Errorf("ParseFormat(%q) expected error", tt.input)
		}
		if !tt.err && err != nil {
			t.Errorf("ParseFormat(%q) unexpected error: %v", tt.input, err)
		}
		if got != tt.want {
			t.Errorf("ParseFormat(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestRenderMarkdown(t *testing.T) {
	fd := testFileDescriptor()
	pf := ParseFile(fd)

	var buf bytes.Buffer
	if err := Render(&buf, []ProtoFile{pf}, FormatMarkdown); err != nil {
		t.Fatal(err)
	}

	md := buf.String()

	// Check key content appears in Markdown output.
	checks := []string{
		"# Protocol Documentation",
		"## tensorq/darwinian/v1/darwinian_gateway.proto",
		"### Service: CortexGateway",
		"StreamSimulation",
		"UpdateAlphaRoute",
		"SimRequest",
		"CodeChunk",
		"RouteMutation",
		"RouteAck",
		"ChunkType",
		"RouteStatus",
		"server streaming",
		"unary",
	}
	for _, c := range checks {
		if !strings.Contains(md, c) {
			t.Errorf("Markdown output missing %q", c)
		}
	}
}

func TestRenderHTML(t *testing.T) {
	fd := testFileDescriptor()
	pf := ParseFile(fd)

	var buf bytes.Buffer
	if err := Render(&buf, []ProtoFile{pf}, FormatHTML); err != nil {
		t.Fatal(err)
	}

	html := buf.String()

	checks := []string{
		"<!DOCTYPE html>",
		"Protocol Documentation",
		"CortexGateway",
		"StreamSimulation",
		"server streaming",
		"SimRequest",
		"ChunkType",
	}
	for _, c := range checks {
		if !strings.Contains(html, c) {
			t.Errorf("HTML output missing %q", c)
		}
	}
}

func TestRenderJSON(t *testing.T) {
	fd := testFileDescriptor()
	pf := ParseFile(fd)

	var buf bytes.Buffer
	if err := Render(&buf, []ProtoFile{pf}, FormatJSON); err != nil {
		t.Fatal(err)
	}

	var result []ProtoFile
	if err := json.Unmarshal(buf.Bytes(), &result); err != nil {
		t.Fatalf("JSON output is not valid: %v", err)
	}

	if len(result) != 1 {
		t.Fatalf("JSON has %d files, want 1", len(result))
	}
	if result[0].Name != "tensorq/darwinian/v1/darwinian_gateway.proto" {
		t.Errorf("JSON file name = %q", result[0].Name)
	}
	if len(result[0].Services) != 1 {
		t.Errorf("JSON services count = %d, want 1", len(result[0].Services))
	}
	if len(result[0].Messages) != 4 {
		t.Errorf("JSON messages count = %d, want 4", len(result[0].Messages))
	}
	if len(result[0].Enums) != 2 {
		t.Errorf("JSON enums count = %d, want 2", len(result[0].Enums))
	}
}

func TestRunPlugin(t *testing.T) {
	fd := testFileDescriptor()

	req := &pluginpb.CodeGeneratorRequest{
		FileToGenerate: []string{fd.GetName()},
		ProtoFile:      []*descriptorpb.FileDescriptorProto{fd},
		Parameter:      proto.String("markdown,api.md"),
	}

	reqBytes, err := proto.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}

	var stdout bytes.Buffer
	if err := Run(bytes.NewReader(reqBytes), &stdout); err != nil {
		t.Fatal(err)
	}

	var resp pluginpb.CodeGeneratorResponse
	if err := proto.Unmarshal(stdout.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}

	if resp.Error != nil {
		t.Fatalf("plugin returned error: %s", *resp.Error)
	}

	if len(resp.File) != 1 {
		t.Fatalf("plugin returned %d files, want 1", len(resp.File))
	}

	f := resp.File[0]
	if f.GetName() != "api.md" {
		t.Errorf("output filename = %q, want api.md", f.GetName())
	}
	if !strings.Contains(f.GetContent(), "CortexGateway") {
		t.Error("output missing CortexGateway")
	}
	if !strings.Contains(f.GetContent(), "StreamSimulation") {
		t.Error("output missing StreamSimulation")
	}
}

func TestRunPlugin_HTML(t *testing.T) {
	fd := testFileDescriptor()

	req := &pluginpb.CodeGeneratorRequest{
		FileToGenerate: []string{fd.GetName()},
		ProtoFile:      []*descriptorpb.FileDescriptorProto{fd},
		Parameter:      proto.String("html,api.html"),
	}

	reqBytes, err := proto.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}

	var stdout bytes.Buffer
	if err := Run(bytes.NewReader(reqBytes), &stdout); err != nil {
		t.Fatal(err)
	}

	var resp pluginpb.CodeGeneratorResponse
	if err := proto.Unmarshal(stdout.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}

	if resp.Error != nil {
		t.Fatalf("plugin returned error: %s", *resp.Error)
	}
	if len(resp.File) != 1 {
		t.Fatalf("plugin returned %d files, want 1", len(resp.File))
	}
	if f := resp.File[0]; f.GetName() != "api.html" {
		t.Errorf("output filename = %q, want api.html", f.GetName())
	}
	if !strings.Contains(resp.File[0].GetContent(), "<!DOCTYPE html>") {
		t.Error("HTML output missing DOCTYPE")
	}
}

func TestRunPlugin_JSON(t *testing.T) {
	fd := testFileDescriptor()

	req := &pluginpb.CodeGeneratorRequest{
		FileToGenerate: []string{fd.GetName()},
		ProtoFile:      []*descriptorpb.FileDescriptorProto{fd},
		Parameter:      proto.String("json"),
	}

	reqBytes, err := proto.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}

	var stdout bytes.Buffer
	if err := Run(bytes.NewReader(reqBytes), &stdout); err != nil {
		t.Fatal(err)
	}

	var resp pluginpb.CodeGeneratorResponse
	if err := proto.Unmarshal(stdout.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}

	if resp.Error != nil {
		t.Fatalf("plugin returned error: %s", *resp.Error)
	}
	if len(resp.File) != 1 {
		t.Fatalf("plugin returned %d files, want 1", len(resp.File))
	}
	if f := resp.File[0]; f.GetName() != "docs.json" {
		t.Errorf("output filename = %q, want docs.json", f.GetName())
	}
}

func TestRunPlugin_DefaultParams(t *testing.T) {
	fd := testFileDescriptor()

	req := &pluginpb.CodeGeneratorRequest{
		FileToGenerate: []string{fd.GetName()},
		ProtoFile:      []*descriptorpb.FileDescriptorProto{fd},
	}

	reqBytes, err := proto.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}

	var stdout bytes.Buffer
	if err := Run(bytes.NewReader(reqBytes), &stdout); err != nil {
		t.Fatal(err)
	}

	var resp pluginpb.CodeGeneratorResponse
	if err := proto.Unmarshal(stdout.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}

	if resp.Error != nil {
		t.Fatalf("plugin returned error: %s", *resp.Error)
	}
	if len(resp.File) != 1 {
		t.Fatalf("plugin returned %d files, want 1", len(resp.File))
	}
	// Default: markdown, docs.md
	if f := resp.File[0]; f.GetName() != "docs.md" {
		t.Errorf("default output filename = %q, want docs.md", f.GetName())
	}
}

func TestScalarTypeName(t *testing.T) {
	tests := []struct {
		typ  descriptorpb.FieldDescriptorProto_Type
		want string
	}{
		{descriptorpb.FieldDescriptorProto_TYPE_DOUBLE, "double"},
		{descriptorpb.FieldDescriptorProto_TYPE_FLOAT, "float"},
		{descriptorpb.FieldDescriptorProto_TYPE_INT64, "int64"},
		{descriptorpb.FieldDescriptorProto_TYPE_UINT64, "uint64"},
		{descriptorpb.FieldDescriptorProto_TYPE_INT32, "int32"},
		{descriptorpb.FieldDescriptorProto_TYPE_BOOL, "bool"},
		{descriptorpb.FieldDescriptorProto_TYPE_STRING, "string"},
		{descriptorpb.FieldDescriptorProto_TYPE_BYTES, "bytes"},
		{descriptorpb.FieldDescriptorProto_TYPE_UINT32, "uint32"},
		{descriptorpb.FieldDescriptorProto_TYPE_FIXED32, "fixed32"},
		{descriptorpb.FieldDescriptorProto_TYPE_FIXED64, "fixed64"},
		{descriptorpb.FieldDescriptorProto_TYPE_SFIXED32, "sfixed32"},
		{descriptorpb.FieldDescriptorProto_TYPE_SFIXED64, "sfixed64"},
		{descriptorpb.FieldDescriptorProto_TYPE_SINT32, "sint32"},
		{descriptorpb.FieldDescriptorProto_TYPE_SINT64, "sint64"},
	}
	for _, tt := range tests {
		got := scalarTypeName(tt.typ)
		if got != tt.want {
			t.Errorf("scalarTypeName(%v) = %q, want %q", tt.typ, got, tt.want)
		}
	}
}

func TestFieldTypes(t *testing.T) {
	// Test that enum type references are resolved correctly.
	fd := testFileDescriptor()
	pf := ParseFile(fd)

	// CodeChunk.type should be "ChunkType"
	codeChunk := pf.Messages[1]
	if codeChunk.Name != "CodeChunk" {
		t.Fatalf("expected CodeChunk, got %s", codeChunk.Name)
	}
	typeField := codeChunk.Fields[1]
	if typeField.Name != "type" {
		t.Fatalf("expected type field, got %s", typeField.Name)
	}
	if typeField.Type != "ChunkType" {
		t.Errorf("type field Type = %q, want ChunkType", typeField.Type)
	}
}

func TestParseParams(t *testing.T) {
	tests := []struct {
		input      string
		wantFormat Format
		wantFile   string
	}{
		{"", FormatMarkdown, "docs.md"},
		{"markdown,api.md", FormatMarkdown, "api.md"},
		{"html,docs.html", FormatHTML, "docs.html"},
		{"json", FormatJSON, "docs.json"},
		{"html", FormatHTML, "docs.html"},
		{"md,README.md", FormatMarkdown, "README.md"},
	}
	for _, tt := range tests {
		f, out := parseParams(tt.input)
		if f != tt.wantFormat {
			t.Errorf("parseParams(%q) format = %q, want %q", tt.input, f, tt.wantFormat)
		}
		if out != tt.wantFile {
			t.Errorf("parseParams(%q) file = %q, want %q", tt.input, out, tt.wantFile)
		}
	}
}
