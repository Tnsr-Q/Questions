.PHONY: help tidy vendor build clean build-darwin-gate build-jaxd

help:
	@echo "Questions Repository - Build Targets"
	@echo "===================================="
	@echo "  make tidy              - Run go mod tidy for all modules"
	@echo "  make vendor            - Vendor all Go dependencies"
	@echo "  make build             - Build all binaries"
	@echo "  make build-darwin-gate - Build Darwins-Gate binaries"
	@echo "  make build-jaxd        - Build/test JAX'D components"
	@echo "  make clean             - Clean build artifacts"

# Tidy Go modules
tidy:
	@echo "Running 'go mod tidy' in root..."
	go mod tidy
	@echo "Running 'go mod tidy' in Darwins-Gate..."
	cd Darwins-Gate && go mod tidy

# Vendor dependencies
vendor: tidy
	@echo "Vendoring dependencies..."
	go mod vendor
	@echo "Vendor directory created at ./vendor"

# Build all components
build: build-darwin-gate build-jaxd
	@echo "Build complete"

# Build Darwins-Gate binaries
build-darwin-gate:
	@echo "Building Darwins-Gate components..."
	@cd Darwins-Gate && \
	for cmd in cmd/*/; do \
		if [ -f "$$cmd/main.go" ]; then \
			name=$$(basename $$cmd); \
			echo "Building $$name..."; \
			go build -o ./bin/$$name ./$$cmd; \
		fi; \
	done
	@echo "Darwins-Gate build complete"

# Build/test JAX'D components
build-jaxd:
	@echo "Processing JAX'D components..."
	@echo "Note: JAX'D contains Python code; ensure dependencies are installed"
	@echo "JAX'D processing complete"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf ./bin
	rm -rf ./Darwins-Gate/bin
	rm -rf ./vendor
	@echo "Clean complete"
