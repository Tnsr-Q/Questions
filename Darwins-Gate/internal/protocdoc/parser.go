package protocdoc

import (
	"strconv"
	"strings"

	"google.golang.org/protobuf/types/descriptorpb"
)

// ParseFile converts a protobuf FileDescriptorProto into a documentation model.
func ParseFile(fd *descriptorpb.FileDescriptorProto) ProtoFile {
	pf := ProtoFile{
		Name:    fd.GetName(),
		Package: fd.GetPackage(),
		Syntax:  fd.GetSyntax(),
	}

	if fd.GetOptions() != nil {
		if goPkg := fd.GetOptions().GetGoPackage(); goPkg != "" {
			pf.Options = append(pf.Options, Option{Name: "go_package", Value: goPkg})
		}
		if javaPkg := fd.GetOptions().GetJavaPackage(); javaPkg != "" {
			pf.Options = append(pf.Options, Option{Name: "java_package", Value: javaPkg})
		}
	}

	locs := buildLocationMap(fd.GetSourceCodeInfo())

	for i, sd := range fd.GetService() {
		svc := parseService(sd, fd.GetPackage(), locs, []int32{6, int32(i)})
		pf.Services = append(pf.Services, svc)
	}

	for i, md := range fd.GetMessageType() {
		msg := parseMessage(md, fd.GetPackage(), locs, []int32{4, int32(i)})
		pf.Messages = append(pf.Messages, msg)
	}

	for i, ed := range fd.GetEnumType() {
		enum := parseEnum(ed, fd.GetPackage(), locs, []int32{5, int32(i)})
		pf.Enums = append(pf.Enums, enum)
	}

	pf.HasEnums = len(pf.Enums) > 0
	pf.HasMessages = len(pf.Messages) > 0

	if desc := fileDescription(fd, locs); desc != "" {
		pf.Description = desc
	}

	return pf
}

func parseService(sd *descriptorpb.ServiceDescriptorProto, pkg string, locs locationMap, path []int32) Service {
	svc := Service{
		Name:        sd.GetName(),
		FullName:    fullName(pkg, sd.GetName()),
		Description: locs.leadingComment(path),
	}
	for i, md := range sd.GetMethod() {
		mPath := append(append([]int32{}, path...), 2, int32(i))
		svc.Methods = append(svc.Methods, parseMethod(md, pkg, locs, mPath))
	}
	return svc
}

func parseMethod(md *descriptorpb.MethodDescriptorProto, pkg string, locs locationMap, path []int32) Method {
	return Method{
		Name:             md.GetName(),
		Description:      locs.leadingComment(path),
		RequestType:      baseName(md.GetInputType()),
		RequestFullType:  strings.TrimPrefix(md.GetInputType(), "."),
		ResponseType:     baseName(md.GetOutputType()),
		ResponseFullType: strings.TrimPrefix(md.GetOutputType(), "."),
		ClientStreaming:  md.GetClientStreaming(),
		ServerStreaming:  md.GetServerStreaming(),
	}
}

func parseMessage(md *descriptorpb.DescriptorProto, pkg string, locs locationMap, path []int32) Message {
	msg := Message{
		Name:        md.GetName(),
		FullName:    fullName(pkg, md.GetName()),
		Description: locs.leadingComment(path),
	}

	// Build a set of nested map-entry message names so we can detect map fields.
	mapEntries := buildMapEntrySet(md)

	oneofNames := make(map[int32]string)
	for _, od := range md.GetOneofDecl() {
		oneofNames[int32(len(oneofNames))] = od.GetName()
	}

	for i, fd := range md.GetField() {
		fPath := append(append([]int32{}, path...), 2, int32(i))
		f := parseField(fd, locs, fPath, oneofNames, mapEntries)
		msg.Fields = append(msg.Fields, f)
	}

	msg.HasFields = len(msg.Fields) > 0
	return msg
}

// buildMapEntrySet returns a map of fully-qualified type names that are
// compiler-generated map entry messages. The key and value type names are
// stored as the map value in "key_type, value_type" format.
func buildMapEntrySet(md *descriptorpb.DescriptorProto) map[string]mapEntryInfo {
	entries := make(map[string]mapEntryInfo)
	for _, nested := range md.GetNestedType() {
		if nested.GetOptions().GetMapEntry() {
			var keyType, valueType string
			for _, f := range nested.GetField() {
				switch f.GetName() {
				case "key":
					keyType = fieldTypeName(f)
				case "value":
					valueType = fieldTypeName(f)
				}
			}
			entries[nested.GetName()] = mapEntryInfo{keyType: keyType, valueType: valueType}
		}
	}
	return entries
}

type mapEntryInfo struct {
	keyType   string
	valueType string
}

func parseField(fd *descriptorpb.FieldDescriptorProto, locs locationMap, path []int32, oneofNames map[int32]string, mapEntries map[string]mapEntryInfo) Field {
	f := Field{
		Name:        fd.GetName(),
		Description: locs.leadingComment(path),
		Type:        fieldTypeName(fd),
		Number:      fd.GetNumber(),
		DefaultValue: fd.GetDefaultValue(),
	}

	if fd.GetLabel() == descriptorpb.FieldDescriptorProto_LABEL_REPEATED {
		f.Label = "repeated"
		f.IsRepeated = true
	} else if fd.GetLabel() == descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL {
		f.Label = "optional"
	}

	// Detect map fields using the compiler-generated map entry messages.
	if fd.GetType() == descriptorpb.FieldDescriptorProto_TYPE_MESSAGE && fd.GetLabel() == descriptorpb.FieldDescriptorProto_LABEL_REPEATED {
		typeName := baseName(fd.GetTypeName())
		if entry, ok := mapEntries[typeName]; ok {
			f.IsMap = true
			f.IsRepeated = false
			f.Label = ""
			f.Type = "map<" + entry.keyType + ", " + entry.valueType + ">"
		}
	}

	if fd.OneofIndex != nil {
		f.IsOneof = true
		if name, ok := oneofNames[fd.GetOneofIndex()]; ok {
			f.OneofName = name
		}
	}

	return f
}

func parseEnum(ed *descriptorpb.EnumDescriptorProto, pkg string, locs locationMap, path []int32) Enum {
	enum := Enum{
		Name:        ed.GetName(),
		FullName:    fullName(pkg, ed.GetName()),
		Description: locs.leadingComment(path),
	}
	for i, vd := range ed.GetValue() {
		vPath := append(append([]int32{}, path...), 2, int32(i))
		enum.Values = append(enum.Values, EnumValue{
			Name:        vd.GetName(),
			Number:      vd.GetNumber(),
			Description: locs.leadingComment(vPath),
		})
	}
	return enum
}

// locationMap indexes source code locations by their path for quick lookup.
type locationMap map[string]*descriptorpb.SourceCodeInfo_Location

func buildLocationMap(sci *descriptorpb.SourceCodeInfo) locationMap {
	m := make(locationMap)
	if sci == nil {
		return m
	}
	for _, loc := range sci.GetLocation() {
		key := pathKey(loc.GetPath())
		m[key] = loc
	}
	return m
}

func (lm locationMap) leadingComment(path []int32) string {
	loc, ok := lm[pathKey(path)]
	if !ok {
		return ""
	}
	comment := loc.GetLeadingComments()
	if comment == "" {
		// Fall back to leading detached comments.
		for _, c := range loc.GetLeadingDetachedComments() {
			if c != "" {
				comment = c
				break
			}
		}
	}
	return cleanComment(comment)
}

func pathKey(path []int32) string {
	parts := make([]string, len(path))
	for i, p := range path {
		parts[i] = strconv.FormatInt(int64(p), 10)
	}
	return strings.Join(parts, ".")
}

func fieldTypeName(fd *descriptorpb.FieldDescriptorProto) string {
	if fd.GetTypeName() != "" {
		return baseName(fd.GetTypeName())
	}
	return scalarTypeName(fd.GetType())
}

func scalarTypeName(t descriptorpb.FieldDescriptorProto_Type) string {
	switch t {
	case descriptorpb.FieldDescriptorProto_TYPE_DOUBLE:
		return "double"
	case descriptorpb.FieldDescriptorProto_TYPE_FLOAT:
		return "float"
	case descriptorpb.FieldDescriptorProto_TYPE_INT64:
		return "int64"
	case descriptorpb.FieldDescriptorProto_TYPE_UINT64:
		return "uint64"
	case descriptorpb.FieldDescriptorProto_TYPE_INT32:
		return "int32"
	case descriptorpb.FieldDescriptorProto_TYPE_FIXED64:
		return "fixed64"
	case descriptorpb.FieldDescriptorProto_TYPE_FIXED32:
		return "fixed32"
	case descriptorpb.FieldDescriptorProto_TYPE_BOOL:
		return "bool"
	case descriptorpb.FieldDescriptorProto_TYPE_STRING:
		return "string"
	case descriptorpb.FieldDescriptorProto_TYPE_BYTES:
		return "bytes"
	case descriptorpb.FieldDescriptorProto_TYPE_UINT32:
		return "uint32"
	case descriptorpb.FieldDescriptorProto_TYPE_SFIXED32:
		return "sfixed32"
	case descriptorpb.FieldDescriptorProto_TYPE_SFIXED64:
		return "sfixed64"
	case descriptorpb.FieldDescriptorProto_TYPE_SINT32:
		return "sint32"
	case descriptorpb.FieldDescriptorProto_TYPE_SINT64:
		return "sint64"
	default:
		return "unknown"
	}
}

func baseName(fqn string) string {
	if idx := strings.LastIndex(fqn, "."); idx >= 0 {
		return fqn[idx+1:]
	}
	return fqn
}

func fullName(pkg, name string) string {
	if pkg == "" {
		return name
	}
	return pkg + "." + name
}

func cleanComment(s string) string {
	s = strings.TrimSpace(s)
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = strings.TrimSpace(l)
	}
	return strings.Join(lines, "\n")
}

func fileDescription(fd *descriptorpb.FileDescriptorProto, locs locationMap) string {
	// The file-level comment is at path [12] (syntax) or the overall file path [].
	if c := locs.leadingComment([]int32{12}); c != "" {
		return c
	}
	if c := locs.leadingComment(nil); c != "" {
		return c
	}
	return ""
}
