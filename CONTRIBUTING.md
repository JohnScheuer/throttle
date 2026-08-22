# Contributing to Throttle

Thank you for considering contributing to Throttle! This document provides basic guidance for contributors.

## Running Tests

The test suite is offline-only and blocks non-loopback DNS/socket use to ensure tests don't make external network calls.

```bash
# Install in development mode
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

# Run the full test suite
PYTHONPATH=src python -m unittest discover -s tests -v
```

All tests should pass before submitting a pull request. The suite covers:
- All CLI modes (plan, smoke, benchmark, diagnose, experimental-tuning, golden, compare, proxy)
- URL/proxy safety validation
- Response validation and streaming termination
- Cost separation and billing models
- Confidence interval and boundary logic
- Manifest tampering detection
- Saved run comparisons
- GuideLLM subprocess boundary
- Six-run golden protocol validation
- Experimental tuning collector/analyzer/safety chain

## Code Style

Throttle prioritizes:
- **Correctness over cleverness**: Clear, auditable code
- **Explicit over implicit**: No hidden behavior or silent fallbacks
- **Safety by default**: Fail closed on invalid inputs
- **Evidence-based claims**: All performance numbers must be validated and traceable

There is no automated linter configuration. Follow the existing code style in the file you're modifying.

## Pull Request Process

1. **Fork the repository** and create a feature branch from `main`
2. **Make your changes** with clear, focused commits
3. **Run the test suite** and ensure all tests pass
4. **Update documentation** if you're adding features or changing behavior
5. **Submit a pull request** with a clear description of:
   - What problem you're solving
   - How your changes address it
   - Any new test coverage you've added

## What to Contribute

Throttle welcomes contributions in several areas:

- **Bug fixes**: Incorrect behavior, edge cases, or safety issues
- **Test coverage**: Additional test cases for existing functionality
- **Documentation**: Clarifications, examples, or corrections
- **Performance improvements**: With validated benchmark evidence
- **New backends**: Support for additional inference engines
- **Protocol improvements**: Enhancements to the golden protocol or validation

## What NOT to Contribute (Without Discussion)

Please open an issue first before working on:

- Major architectural changes
- New CLI modes or subcommands
- Changes to the golden protocol validation logic
- Automatic configuration or "magic" behavior
- Breaking changes to saved report schema

## Reporting Issues

When reporting bugs, please include:

- Throttle version (`throttle --version`)
- Python version
- Operating system
- Minimal reproduction steps
- Expected vs actual behavior
- Sanitized command output (remove credentials/URLs)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
