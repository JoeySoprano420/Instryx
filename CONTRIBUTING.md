# Contributing to Instryx

Thank you for your interest in contributing to the Instryx programming language! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs or request features
- Provide clear, detailed descriptions with examples
- Include version information and system details

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes following our coding standards
4. Add tests for your changes
5. Ensure all tests pass
6. Commit with clear, descriptive messages
7. Push to your fork and create a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/Instryx.git
cd Instryx

# Build the project
make build

# Run tests
make test

# Run the compiler
./instryx --help
```

## Coding Standards

- Follow the existing code style
- Write clear, self-documenting code
- Add comments for complex logic
- Include tests for new features
- Update documentation as needed

## Areas for Contribution

- **Core Language**: Lexer, parser, type checker, code generation
- **Standard Library**: Built-in functions, data structures, utilities
- **Tooling**: Package manager, build system, IDE support
- **Documentation**: Tutorials, examples, API documentation
- **Testing**: Unit tests, integration tests, benchmarks

## Backwards Compatibility

When making changes:
- Ensure existing code continues to work
- Mark deprecated features clearly
- Provide migration guides for breaking changes
- Consider the impact on the ecosystem

## Code Review Process

All contributions go through code review to ensure:
- Code quality and consistency
- Backwards compatibility
- Performance considerations
- Security implications
- Documentation completeness

## Getting Help

- Join our Discord community (coming soon)
- Open a discussion on GitHub
- Check the documentation and examples

## Recognition

Contributors are recognized in our CONTRIBUTORS.md file and release notes.

Thank you for helping make Instryx better!