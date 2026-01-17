# Contributing to ml_compiler

Thank you for your interest in contributing to ml_compiler! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.9 or higher
- Git

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/ndpage/ml_compiler.git
cd ml_compiler
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in editable mode with development dependencies:
```bash
pip install -e .[onnx,dev]
```

## Development Workflow

### Running Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Run a specific test file:
```bash
python -m pytest tests/test_graph.py -v
```

### Code Quality

Run the linter:
```bash
python -m ruff check .
```

Auto-fix linter issues:
```bash
python -m ruff check . --fix
```

Run the type checker:
```bash
python -m mypy src/ml_compiler --ignore-missing-imports
```

### Running the Compiler

Compile an ONNX model to C code:
```bash
python -m ml_compiler.cli compile --model path/to/model.onnx --out build_dir --passes quant_stub
```

Example with the included test model:
```bash
python -m ml_compiler.cli compile --model models/example_cnn.onnx --out build_dir --passes quant_stub
```

### Building Generated C Code

After generating C code, you can compile it with GCC:
```bash
cd build_dir
gcc -o inference main.c model.c operators.c -lm
./inference
```

## Contribution Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for all function signatures
- Keep line length to 100 characters (enforced by ruff)
- Add docstrings to public functions and classes

### Making Changes

1. Create a new branch for your feature or fix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes, ensuring:
   - All tests pass
   - Linter passes (`ruff check .`)
   - Type checker passes (`mypy src/ml_compiler --ignore-missing-imports`)
   - New features include tests
   - Code follows existing patterns and conventions

3. Commit your changes with clear, descriptive commit messages:
```bash
git add .
git commit -m "Add feature: description of your changes"
```

4. Push your branch and create a pull request:
```bash
git push origin feature/your-feature-name
```

### Testing Guidelines
- Add unit tests for new functionality
- Ensure tests are isolated and don't depend on external resources
- Use descriptive test names that explain what is being tested
- Keep tests focused on a single behavior

### Adding New Operators

When adding support for a new operator:
1. Update `src/ml_compiler/frontend/onnx_loader.py` to parse the operator
2. Add kernel template in `src/ml_compiler/codegen/esp32_c.py`
3. Add a test case in `tests/test_loader_example_model.py` or create a new test file
4. Update documentation if the operator has special considerations

### Adding New Passes

When adding a new compiler pass:
1. Create a new file in `src/ml_compiler/passes/`
2. Register the pass in `src/ml_compiler/cli.py` PASS_REGISTRY
3. Add unit tests for the pass
4. Document the pass behavior in `docs/architecture.md`

## Project Structure

```
ml_compiler/
├── src/ml_compiler/     # Main source code
│   ├── ir/              # Intermediate representation
│   ├── frontend/        # ONNX model loader
│   ├── passes/          # Optimization passes
│   ├── codegen/         # Code generation backends
│   └── cli.py           # Command-line interface
├── tests/               # Test suite
├── tools/               # Utility scripts
├── models/              # Example models
├── docs/                # Documentation
└── examples/            # Integration examples

```

## Getting Help

- Check the [README.md](README.md) for project overview
- Review [docs/architecture.md](docs/architecture.md) for design details
- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas

## License

By contributing to ml_compiler, you agree that your contributions will be licensed under the MIT License.
