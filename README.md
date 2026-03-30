# Ive Programming Language

MLIR-based programming language, optimized for DSP and ML workloads.

## Overview

Ive is a domain-specific language built on top of MLIR (Multi-Level Intermediate Representation) that provides high-level abstracts for tensor operations, automatic differentiation, and optimization for digital signal processing and machine learning applications.

## Features

- **Tensor Operations**: Built-in support for multi-dimensional arrays and operations
- **Type Inference**: Automatic shape and type deduction
- **MLIR Backend**: Leverages MLIR's optimization infrastructure
- **Multiple Compilation Targets**: AST, MLIR, LLVM IR, and JIT execution
- **Struct Types**: User-defined composite data types
- **Generic Functions**: Functions that operate on unknown-shaped arguments

## Building

### Prerequisites

- CMake 3.20+
- LLVM/MLIR (compatible version)
- C++17 compatible compiler
- Ninja (optional)

### Build Instructions

**Using Make:**
```bash
mkdir build
cd build
cmake ..
make
```

**Using Ninja (recommended):**
```bash
cmake -B build -G Ninja
cmake --build build
```

## Usage

### Basic Command

```bash
./build/ive [options] <input-file>
```

### Command Line Options

#### Input Type (`-x`)
- `ive` (default): Load input file as ive source code
- `mlir`: Load input file as MLIR IR

#### Output Format (`-emit`)
- `ast`: Print the Abstract Syntax Tree
- `mlir`: Print MLIR representation  
- `mlir-affine`: Print MLIR after affine lowering
- `mlir-llvm`: Print MLIR after LLVM lowering
- `llvm`: Print LLVM IR
- `jit`: JIT compile and execute (calls `main()` function)

#### Optimization
- `--opt`: Enable optimization passes

### Examples

**Execute a program:**
```bash
./build/ive -emit=jit examples/multiply_transpose.ive
```

**View AST:**
```bash
./build/ive -emit=ast examples/multiply_transpose.ive
```

**View MLIR:**
```bash
./build/ive -emit=mlir examples/multiply_transpose.ive
```

**View LLVM IR with optimizations:**
```bash
./build/ive -emit=llvm --opt examples/multiply_transpose.ive
```

**Load and process MLIR file:**
```bash
./build/ive -x mlir -emit=mlir-llvm examples/multiply_transpose.mlir
```

## Language Syntax

### Variable Declaration

```ive
var a = [[1, 2, 3], [4, 5, 6]];         # Inferred shape: <2x3>
var b<2, 3> = [1, 2, 3, 4, 5, 6];      # Explicit shape annotation
```

### Function Definition

```ive
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

### Struct Types

```ive
struct Struct {
  var a;
  var b;
}

# User defined generic function may operate on struct types as well.
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # We pass these arguments to functions like we do with variables.
  var c = multiply_transpose(value);
  print(c);
}
```

### Built-in Operations

- Arithmetic: `+`, `-`, `*`, `/`
- Tensor operations: `transpose()`, `reshape()`
- Shape inference: Automatic deduction of result shapes

## Testing

### Running Tests with LLVM LIT

```bash
llvm-lit -v tests/
```

### Generating Test Checks

Automatically generate FileCheck assertions using the provided script:

**Setup:**
```bash
export LLVM_SRC=~/workspace/llvm-project
```

**Generate checks:**
```bash
bash gen_checks.sh tests/example.ive
```

The script will:
1. Compile the test file and capture MLIR output
2. Auto-generate CHECK assertions based on the output
3. Insert the assertions directly into your test file

## Project Structure

```
ive/
├── build/              # Build directory
├── examples/           # Example programs
│   ├── multiply_transpose.ive
│   └── struct.ive
├── include/ive/        # Header files
│   ├── AST.hpp
│   ├── Lexer.hpp
│   ├── Parser.hpp
│   └── Dialect.hpp
├── parser/             # Lexer and Parser implementation
├── mlir/               # MLIR dialect implementation
├── tests/              # LIT tests
│   ├── example.ive
│   └── lit.site.cfg.py
├── utils/              # Utility scripts
│   └── gen_checks.sh
└── main.cpp            # Compiler driver
```

## Development

### Adding New Tests

1. Create a `.ive` file in `tests/`
2. Add RUN directive: `# RUN: %ive %s -emit=mlir 2>&1 | FileCheck %s`
3. Generate checks: `./gen_checks.sh tests/yourtest.ive`
4. Run tests: `llvm-lit -v tests/yourtest.ive`

### Debugging

Print AST for debugging parsing issues:
```bash
./build/ive -emit=ast examples/yourfile.ive
```

Print MLIR for type/shape issues:
```bash
./build/ive -emit=mlir examples/yourfile.ive
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Tests pass: `llvm-lit -v tests/`
- New features include test coverage

## License

Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE.txt for details.

## Acknowledgments

Built on top of [MLIR](https://mlir.llvm.org/), part of the LLVM project.
