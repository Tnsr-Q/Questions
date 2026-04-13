// protoc-gen-doc is a protoc plugin that generates documentation from
// Protocol Buffer definitions. It supports HTML, Markdown, and JSON output.
//
// Usage with protoc:
//
//	protoc --doc_out=. --doc_opt=markdown,docs.md your_service.proto
//	protoc --doc_out=. --doc_opt=html,docs.html your_service.proto
//	protoc --doc_out=. --doc_opt=json,docs.json your_service.proto
//
// The doc_opt parameter format is: format,filename
// Supported formats: html, markdown (or md), json
package main

import (
	"github.com/tnsr-q/Questions/internal/protocdoc"
)

func main() {
	protocdoc.RunFromOS()
}
