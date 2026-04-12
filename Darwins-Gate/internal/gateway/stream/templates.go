package stream

import (
	"fmt"
	"sort"
	"strings"
)

func DefaultTemplates() map[string]ModelTemplate {
	return map[string]ModelTemplate{
		"black_hole_merger_v4": {
			ModelID: "black_hole_merger_v4",
			Preamble: []string{
				"import math",
				"import json",
				"params = {}",
			},
			Definitions: []string{
				"def run_simulation(params):",
				"    gravity = float(params.get('gravity', '1.0'))",
				"    viscosity = float(params.get('viscosity', '0.01'))",
				"    return {'gravity': gravity, 'viscosity': viscosity, 'signal': gravity * (1.0 - viscosity)}",
			},
			Execution: []string{
				"result = run_simulation(params)",
				"print(json.dumps(result))",
			},
		},
	}
}

func injectParams(params map[string]string) string {
	if len(params) == 0 {
		return "params = {}"
	}
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var b strings.Builder
	b.WriteString("params = {\n")
	for _, k := range keys {
		fmt.Fprintf(&b, "    %q: %q,\n", k, params[k])
	}
	b.WriteString("}\n")
	return b.String()
}

func joinLines(lines []string) string {
	if len(lines) == 0 {
		return ""
	}
	return strings.Join(lines, "\n") + "\n"
}