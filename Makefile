# Makefile for Instryx Programming Language

.PHONY: all build test clean install fmt lint doc bench examples

# Default target
all: build test

# Build the compiler
build:
	cargo build --release

# Build in debug mode
debug:
	cargo build

# Run all tests
test:
	cargo test

# Run specific test
test-integration:
	cargo test --test integration_tests

# Clean build artifacts
clean:
	cargo clean

# Install the compiler
install:
	cargo install --path .

# Format code
fmt:
	cargo fmt

# Check formatting
fmt-check:
	cargo fmt -- --check

# Run linter
lint:
	cargo clippy -- -D warnings

# Generate documentation
doc:
	cargo doc --no-deps --open

# Run benchmarks
bench:
	cargo bench

# Build examples
examples: build
	@echo "Building examples..."
	./target/release/instryx build examples/hello_world.ix -o hello_world
	./target/release/instryx build examples/variables.ix -o variables
	./target/release/instryx build examples/functions.ix -o functions

# Development targets
dev-setup:
	rustup component add rustfmt clippy
	cargo install cargo-watch cargo-tarpaulin

# Watch for changes and rebuild
watch:
	cargo watch -x build

# Watch for changes and run tests
watch-test:
	cargo watch -x test

# Code coverage
coverage:
	cargo tarpaulin --out Html

# Release build with optimizations
release: test lint
	cargo build --release
	strip target/release/instryx

# Create distribution package
dist: release
	mkdir -p dist/instryx-$(shell cargo pkgid | cut -d'#' -f2)
	cp target/release/instryx dist/instryx-$(shell cargo pkgid | cut -d'#' -f2)/
	cp README.md LICENSE CONTRIBUTING.md dist/instryx-$(shell cargo pkgid | cut -d'#' -f2)/
	tar -czf dist/instryx-$(shell cargo pkgid | cut -d'#' -f2).tar.gz -C dist instryx-$(shell cargo pkgid | cut -d'#' -f2)

# Check everything before commit
pre-commit: fmt lint test doc
	@echo "All checks passed!"

# Quick development cycle
dev: debug test

# Performance profiling
profile: release
	perf record -g ./target/release/instryx build examples/complex.ix
	perf report

# Memory check (requires valgrind)
memcheck: debug
	valgrind --tool=memcheck --leak-check=full ./target/debug/instryx build examples/hello_world.ix

# Help target
help:
	@echo "Available targets:"
	@echo "  build       - Build the compiler in release mode"
	@echo "  debug       - Build the compiler in debug mode"
	@echo "  test        - Run all tests"
	@echo "  clean       - Clean build artifacts"
	@echo "  install     - Install the compiler globally"
	@echo "  fmt         - Format code"
	@echo "  lint        - Run linter"
	@echo "  doc         - Generate documentation"
	@echo "  bench       - Run benchmarks"
	@echo "  examples    - Build example programs"
	@echo "  dev-setup   - Setup development environment"
	@echo "  watch       - Watch for changes and rebuild"
	@echo "  coverage    - Generate code coverage report"
	@echo "  release     - Create optimized release build"
	@echo "  dist        - Create distribution package"
	@echo "  pre-commit  - Run all checks before committing"
	@echo "  help        - Show this help message"