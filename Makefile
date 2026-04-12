.PHONY: help work-init work-sync work-use tidy vendor build clean build-darwin-gate build-jaxd test-jaxd

help:
	@echo "Questions Monorepo - Build Targets"
	@echo "=================================="
	@echo "Workspace management:"
	@echo "  make work-init         - Initialize Go workspace (go work init)"
	@echo "  make work-sync         - Sync workspace dependencies (go work sync)"
	@echo "  make work-use          - Add all local modules to workspace"
	@echo ""
	@echo "Build targets:"
	@echo "  make tidy              - Run go mod tidy across all modules"
	@echo "  make vendor            - Vendor Darwins-Gate dependencies"
	@echo "  make build             - Build all components"
	@echo "  make build-darwin-gate - Build Darwins-Gate binaries"
	@echo "  make build-jaxd        - Install JAX'D Python dependencies"
	@echo "  make test-jaxd         - Run JAX'D tests"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean             - Clean build artifacts"

# Initialize Go workspace (only needed once)
work-init:
	@echo "Initializing Go workspace..."
	go work init ./Darwins-Gate
	@echo "Workspace initialized. See go.work"

# Sync workspace dependencies down to individual go.mod files
work-sync:
	@echo "Syncing workspace dependencies..."
	go work sync
	@echo "Workspace sync complete"

# Add all local Go modules to the workspace (recursive)
work-use:
	@echo "Adding all local Go modules to workspace..."
	go work use -r .
	@echo "Workspace updated"

# Tidy Go modules across workspace
tidy:
	@echo "Running 'go mod tidy' in Darwins-Gate..."
	cd Darwins-Gate && go mod tidy
	@echo "Syncing workspace..."
	go work sync

# Vendor Darwins-Gate dependencies (disables workspace mode for the vendor operation)
vendor:
	@echo "Vendoring Darwins-Gate dependencies..."
	cd Darwins-Gate && GOWORK=off go mod vendor
	@echo "Vendor directory created at Darwins-Gate/vendor"

# Build all components
build: build-darwin-gate
	@echo "Build complete"

# Build Darwins-Gate binaries (uses workspace automatically)
build-darwin-gate:
	@echo "Building Darwins-Gate components..."
	@mkdir -p Darwins-Gate/bin
	@cd Darwins-Gate && \
	for cmd in cmd/*/; do \
		if [ -f "$$cmd/main.go" ]; then \
			name=$$(basename $$cmd); \
			echo "Building $$name..."; \
			go build -o ./bin/$$name ./$$cmd || echo "  warning: $$name failed to build"; \
		fi; \
	done
	@echo "Darwins-Gate build complete"

# Install JAX'D Python dependencies
build-jaxd:
	@echo "Installing JAX'D Python dependencies..."
	@if [ -f requirements.txt ]; then \
		pip install -r requirements.txt; \
	else \
		echo "  warning: requirements.txt not found"; \
	fi

# Run JAX'D tests
test-jaxd:
	@echo "Running JAX'D tests..."
	@cd "JAX'D" && python -m pytest tests/

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf Darwins-Gate/bin
	rm -rf Darwins-Gate/vendor
	rm -rf vendor
	@echo "Clean complete"
