// Package protocdoc generates HTML, Markdown, and JSON documentation from
// Protocol Buffer file descriptors. It is designed to be used as a protoc
// plugin (protoc-gen-doc).
package protocdoc

// ProtoFile holds the documentation model for a single .proto file.
type ProtoFile struct {
	Name        string     `json:"name"`
	Package     string     `json:"package"`
	Description string     `json:"description,omitempty"`
	Services    []Service  `json:"services,omitempty"`
	Messages    []Message  `json:"messages,omitempty"`
	Enums       []Enum     `json:"enums,omitempty"`
	Options     []Option   `json:"options,omitempty"`
	Syntax      string     `json:"syntax,omitempty"`
	HasEnums    bool       `json:"has_enums,omitempty"`
	HasMessages bool       `json:"has_messages,omitempty"`
}

// Service describes a proto service and its RPCs.
type Service struct {
	Name        string   `json:"name"`
	FullName    string   `json:"full_name"`
	Description string   `json:"description,omitempty"`
	Methods     []Method `json:"methods,omitempty"`
}

// Method describes a single RPC method.
type Method struct {
	Name            string `json:"name"`
	Description     string `json:"description,omitempty"`
	RequestType     string `json:"request_type"`
	RequestFullType string `json:"request_full_type"`
	ResponseType    string `json:"response_type"`
	ResponseFullType string `json:"response_full_type"`
	ClientStreaming  bool   `json:"client_streaming,omitempty"`
	ServerStreaming  bool   `json:"server_streaming,omitempty"`
}

// Message describes a proto message type.
type Message struct {
	Name        string   `json:"name"`
	FullName    string   `json:"full_name"`
	Description string   `json:"description,omitempty"`
	Fields      []Field  `json:"fields,omitempty"`
	HasFields   bool     `json:"has_fields,omitempty"`
}

// Field describes a single field in a message.
type Field struct {
	Name         string `json:"name"`
	Description  string `json:"description,omitempty"`
	Type         string `json:"type"`
	Label        string `json:"label,omitempty"`
	Number       int32  `json:"number"`
	DefaultValue string `json:"default_value,omitempty"`
	IsMap        bool   `json:"is_map,omitempty"`
	IsRepeated   bool   `json:"is_repeated,omitempty"`
	IsOneof      bool   `json:"is_oneof,omitempty"`
	OneofName    string `json:"oneof_name,omitempty"`
}

// Enum describes a proto enum type.
type Enum struct {
	Name        string      `json:"name"`
	FullName    string      `json:"full_name"`
	Description string      `json:"description,omitempty"`
	Values      []EnumValue `json:"values,omitempty"`
}

// EnumValue describes a single value in an enum.
type EnumValue struct {
	Name        string `json:"name"`
	Number      int32  `json:"number"`
	Description string `json:"description,omitempty"`
}

// Option holds a top-level file option such as go_package.
type Option struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}
