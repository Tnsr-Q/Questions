package protocdoc

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"text/template"
)

// Format enumerates supported output formats.
type Format string

const (
	FormatHTML     Format = "html"
	FormatMarkdown Format = "markdown"
	FormatJSON     Format = "json"
)

// Render writes documentation for the given files in the specified format.
func Render(w io.Writer, files []ProtoFile, format Format) error {
	switch format {
	case FormatHTML:
		return renderHTML(w, files)
	case FormatMarkdown:
		return renderMarkdown(w, files)
	case FormatJSON:
		return renderJSON(w, files)
	default:
		return fmt.Errorf("unsupported format: %s", format)
	}
}

// ParseFormat converts a string to a Format, returning an error for unknown formats.
func ParseFormat(s string) (Format, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "html":
		return FormatHTML, nil
	case "markdown", "md":
		return FormatMarkdown, nil
	case "json":
		return FormatJSON, nil
	default:
		return "", fmt.Errorf("unknown format %q: supported formats are html, markdown, json", s)
	}
}

// renderJSON writes the documentation as pretty-printed JSON.
func renderJSON(w io.Writer, files []ProtoFile) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(files)
}

// renderHTML writes the documentation as an HTML page.
func renderHTML(w io.Writer, files []ProtoFile) error {
	t, err := template.New("doc").Parse(htmlTemplate)
	if err != nil {
		return fmt.Errorf("parsing html template: %w", err)
	}
	if err := t.Execute(w, files); err != nil {
		return fmt.Errorf("executing html template: %w", err)
	}
	return nil
}

// renderMarkdown writes the documentation as Markdown.
func renderMarkdown(w io.Writer, files []ProtoFile) error {
	t, err := template.New("doc").Funcs(mdFuncs).Parse(markdownTemplate)
	if err != nil {
		return fmt.Errorf("parsing markdown template: %w", err)
	}
	if err := t.Execute(w, files); err != nil {
		return fmt.Errorf("executing markdown template: %w", err)
	}
	return nil
}

var mdFuncs = template.FuncMap{
	"streamLabel": func(client, server bool) string {
		if client && server {
			return "bidirectional streaming"
		}
		if client {
			return "client streaming"
		}
		if server {
			return "server streaming"
		}
		return "unary"
	},
}

const markdownTemplate = `# Protocol Documentation
{{range .}}
## {{.Name}}
{{- if .Description}}

{{.Description}}
{{- end}}

| | |
|---|---|
| **Package** | {{.Package}} |
| **Syntax** | {{.Syntax}} |
{{- range .Options}}
| **{{.Name}}** | ` + "`" + `{{.Value}}` + "`" + ` |
{{- end}}
{{range .Services}}
### Service: {{.Name}}
{{- if .Description}}

{{.Description}}
{{- end}}

| Method | Request | Response | Type |
|--------|---------|----------|------|
{{- range .Methods}}
| {{.Name}} | {{.RequestType}} | {{.ResponseType}} | {{streamLabel .ClientStreaming .ServerStreaming}} |
{{- end}}
{{range .Methods}}
#### {{.Name}}
{{- if .Description}}

> {{.Description}}
{{- end}}

- **Request**: [{{.RequestType}}](#{{.RequestFullType}})
- **Response**: [{{.ResponseType}}](#{{.ResponseFullType}})
{{- if .ClientStreaming}}
- Client streaming: yes
{{- end}}
{{- if .ServerStreaming}}
- Server streaming: yes
{{- end}}
{{end}}
{{- end}}
{{- if .HasMessages}}
### Messages
{{range .Messages}}
#### {{.Name}}

{{- if .Description}}

{{.Description}}
{{- end}}

| Field | Type | Label | Number | Description |
|-------|------|-------|--------|-------------|
{{- range .Fields}}
| {{.Name}} | {{.Type}} | {{.Label}} | {{.Number}} | {{.Description}} |
{{- end}}
{{end}}
{{- end}}
{{- if .HasEnums}}
### Enums
{{range .Enums}}
#### {{.Name}}

{{- if .Description}}

{{.Description}}
{{- end}}

| Name | Number | Description |
|------|--------|-------------|
{{- range .Values}}
| {{.Name}} | {{.Number}} | {{.Description}} |
{{- end}}
{{end}}
{{- end}}
{{- end}}
`

const htmlTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Protocol Documentation</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 960px; margin: 0 auto; padding: 2rem; color: #333; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 0.5rem; }
  h2 { color: #0366d6; margin-top: 2rem; }
  h3 { color: #24292e; }
  h4 { color: #586069; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #dfe2e5; padding: 0.5rem 0.75rem; text-align: left; }
  th { background-color: #f6f8fa; font-weight: 600; }
  tr:nth-child(even) { background-color: #f9f9f9; }
  .description { color: #586069; margin: 0.5rem 0; }
  .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px; font-size: 0.85em; font-weight: 500; }
  .badge-stream { background: #e3f2fd; color: #1565c0; }
  .badge-unary { background: #e8f5e9; color: #2e7d32; }
  code { background: #f6f8fa; padding: 0.15rem 0.3rem; border-radius: 3px; font-size: 0.9em; }
  .meta { color: #586069; font-size: 0.9em; }
  .file-section { margin-bottom: 3rem; }
</style>
</head>
<body>
<h1>Protocol Documentation</h1>
{{range .}}
<div class="file-section">
<h2>{{.Name}}</h2>
{{if .Description}}<p class="description">{{.Description}}</p>{{end}}
<p class="meta"><strong>Package:</strong> {{.Package}} &middot; <strong>Syntax:</strong> {{.Syntax}}</p>
{{range .Options}}<p class="meta"><strong>{{.Name}}:</strong> <code>{{.Value}}</code></p>{{end}}

{{range .Services}}
<h3>Service: {{.Name}}</h3>
{{if .Description}}<p class="description">{{.Description}}</p>{{end}}
<table>
<thead><tr><th>Method</th><th>Request</th><th>Response</th><th>Type</th></tr></thead>
<tbody>
{{range .Methods}}
<tr>
<td><strong>{{.Name}}</strong></td>
<td><code>{{.RequestType}}</code></td>
<td><code>{{.ResponseType}}</code></td>
<td>{{if and .ClientStreaming .ServerStreaming}}<span class="badge badge-stream">bidirectional streaming</span>{{else if .ClientStreaming}}<span class="badge badge-stream">client streaming</span>{{else if .ServerStreaming}}<span class="badge badge-stream">server streaming</span>{{else}}<span class="badge badge-unary">unary</span>{{end}}</td>
</tr>
{{end}}
</tbody>
</table>

{{range .Methods}}
<h4>{{.Name}}</h4>
{{if .Description}}<p class="description">{{.Description}}</p>{{end}}
<ul>
<li><strong>Request:</strong> <code>{{.RequestFullType}}</code></li>
<li><strong>Response:</strong> <code>{{.ResponseFullType}}</code></li>
{{if .ClientStreaming}}<li>Client streaming</li>{{end}}
{{if .ServerStreaming}}<li>Server streaming</li>{{end}}
</ul>
{{end}}
{{end}}

{{if .HasMessages}}
<h3>Messages</h3>
{{range .Messages}}
<h4 id="{{.FullName}}">{{.Name}}</h4>
{{if .Description}}<p class="description">{{.Description}}</p>{{end}}
{{if .HasFields}}
<table>
<thead><tr><th>Field</th><th>Type</th><th>Label</th><th>Number</th><th>Description</th></tr></thead>
<tbody>
{{range .Fields}}
<tr>
<td>{{.Name}}</td>
<td><code>{{.Type}}</code></td>
<td>{{.Label}}</td>
<td>{{.Number}}</td>
<td>{{.Description}}</td>
</tr>
{{end}}
</tbody>
</table>
{{end}}
{{end}}
{{end}}

{{if .HasEnums}}
<h3>Enums</h3>
{{range .Enums}}
<h4 id="{{.FullName}}">{{.Name}}</h4>
{{if .Description}}<p class="description">{{.Description}}</p>{{end}}
<table>
<thead><tr><th>Name</th><th>Number</th><th>Description</th></tr></thead>
<tbody>
{{range .Values}}
<tr>
<td>{{.Name}}</td>
<td>{{.Number}}</td>
<td>{{.Description}}</td>
</tr>
{{end}}
</tbody>
</table>
{{end}}
{{end}}

</div>
{{end}}
</body>
</html>
`
