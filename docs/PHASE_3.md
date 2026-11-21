# **Phase 3 — Matrix Engine**

Now we're moving into one of the most important components of your C machine learning library: **the matrix engine**.

Matrices are the backbone of machine learning. Neural network layers, datasets, transformations, linear systems, batch operations, and gradient calculations all depend heavily on matrix operations.

You'll build this from scratch in C, using your own memory, error-handling, and vector modules.

**Difficulty:** Intermediate → Advanced

**Estimated time:** 2–3 weeks

**Prerequisites:** Phase 0 infrastructure, Phase 2 vector engine

## **Objective**

Build a robust, dynamically allocated matrix library supporting:

- Matrix creation and destruction
- Element access and mutation
- Matrix copying and resizing
- Basic arithmetic
- Matrix multiplication
- Transpose
- Row and column operations
- Matrix-vector multiplication
- Diagonal operations
- Matrix statistics
- Numerical validation
- In-place operations
- Memory-safe handling of dimensions

By the end of this phase, you'll have the foundation for implementing linear regression and neural networks.

# **1. Project structure**

Add the matrix module to your CML library.

```
cml/
├── include/
│   └── cml/
│       ├── memory.h
│       ├── error.h
│       ├── math.h
│       ├── vector.h
│       └── matrix.h
├── src/
│   ├── memory.c
│   ├── error.c
│   ├── math.c
│   ├── vector.c
│   └── matrix.c
├── tests/
│   ├── test_memory.c
│   ├── test_error.c
│   ├── test_vector.c
│   └── test_matrix.c
├── examples/
│   ├── vector_example.c
│   └── matrix_example.c
├── Makefile
└── README.md
```

You'll create:

- `include/cml/matrix.h`
- `src/matrix.c`
- `tests/test_matrix.c`
- `examples/matrix_example.c`

# **2. Matrix data structure**

A matrix represents a two-dimensional collection of values.

For example:

$A= \begin{bmatrix} 1&2&3\\\ 4&5&6 \end{bmatrix}$

It has:

- 2 rows
- 3 columns
- 6 elements

## **Your task**

Design a `CMLMatrix` struct.

It should contain at least:

- A pointer to numerical data
- Number of rows
- Number of columns

Consider adding:

- Capacity
- Stride
- Ownership flag
- Layout information

### **Important design decision: memory layout**

For this assignment, use **row-major contiguous memory**.

That means the matrix:

$\begin{bmatrix} 1&2&3\\\ 4&5&6 \end{bmatrix}$

is stored internally as:

```
[1, 2, 3, 4, 5, 6]
```

The element at row `r`, column `c` can be located using:

$\text{index}=r\times\text{cols}+c$

For example:

| Element | Row | Column | Linear index |
| ------- | --- | ------ | ------------ |
| 1       | 0   | 0      | 0            |
| 2       | 0   | 1      | 1            |
| 3       | 0   | 2      | 2            |
| 4       | 1   | 0      | 3            |
| 5       | 1   | 1      | 4            |
| 6       | 1   | 2      | 5            |

**You must implement the indexing logic yourself.**

# **3. Matrix lifecycle**

Implement:

| Operation     | Description                          |
| ------------- | ------------------------------------ |
| Create        | Allocate a matrix                    |
| Create filled | Allocate and initialize all elements |
| Clone         | Deep-copy a matrix                   |
| Destroy       | Release its resources                |
| Resize        | Change dimensions                    |
| Clear         | Reset its contents                   |
| Identity      | Create an identity matrix            |
| Zeros         | Create a zero-filled matrix          |
| Ones          | Create a matrix filled with ones     |

### **Requirements**

- Validate dimensions.
- Check multiplication overflow when computing total elements.
- Check multiplication overflow when computing allocation size.
- Avoid memory leaks.
- Support zero-sized matrices according to your documented policy.
- Correctly preserve data when resizing, where applicable.
- Newly added elements should have documented initialization behavior.
- Clones must be independent.

### **Test cases**

Create matrices with:

- `0 × 0`
- `1 × 1`
- `2 × 2`
- `2 × 3`
- `3 × 2`
- `100 × 100`

Test:

- Clone independence
- Destroying a matrix
- Resize from `2 × 3` to `3 × 3`
- Resize from `3 × 3` to `1 × 2`
- Resize to zero dimensions
- Overflow dimensions
- Allocation failures

# **4. Element access**

Implement these operations:

- Get element
- Set element
- Fill matrix
- Swap elements
- Get raw data pointer (optional)
- Get row count
- Get column count

Example matrix:

$A= \begin{bmatrix} 10&20&30\\\ 40&50&60 \end{bmatrix}$

Expected:

| Access    | Value |
| --------- | ----- |
| `A[0][0]` | 10    |
| `A[0][2]` | 30    |
| `A[1][0]` | 40    |
| `A[1][2]` | 60    |

Your API should preferably use function calls rather than exposing unsafe indexing to users.

### **Requirements**

- Check row bounds.
- Check column bounds.
- Validate matrix pointers.
- Don't allow invalid accesses.
- Return meaningful errors.

### **Test cases**

- Get the first element.
- Get the last element.
- Set an element.
- Access a row beyond the matrix.
- Access a column beyond the matrix.
- Fill the entire matrix.
- Swap two elements.
- Swap an element with itself.

# **5. Matrix arithmetic**

Implement the fundamental operations.

## **A. Matrix addition**

$C=A+B$

Example:

$\begin{bmatrix}1&2\\3&4\end{bmatrix}+\begin{bmatrix}5&6\\7&8\end{bmatrix}=\begin{bmatrix}6&8\\10&12\end{bmatrix} $

Requirements:

- Dimensions must match.
- Handle invalid dimensions.
- Consider output/input aliasing.
- Support in-place addition.

## **B. Matrix subtraction**

$C=A-B$

Example:

$\begin{bmatrix} 10&20\\\ 30&40 \end{bmatrix} - \begin{bmatrix} 1&2\\\ 3&4 \end{bmatrix} = \begin{bmatrix} 9&18\\\ 27&36 \end{bmatrix}$

## **C. Element-wise multiplication**

$C\_{ij}=A\_{ij}B\_{ij}$

This is also known as the Hadamard product.

## **D. Element-wise division**

$C\_{ij}=\frac{A\_{ij}}{B\_{ij}}$

Define your division-by-zero behavior.

## **E. Scalar multiplication**

$C=\alpha A$

Example:

$3 \begin{bmatrix} 1&2\\\ 3&4 \end{bmatrix} = \begin{bmatrix} 3&6\\\ 9&12 \end{bmatrix}$

## **F. Scalar addition**

$C=A+\alpha$

Every matrix element receives the scalar.

## **G. Negation**

$C=-A$

# **6. Matrix multiplication — the core challenge**

This is the most important operation in this phase.

Unlike element-wise multiplication, matrix multiplication involves summing products.

For:

$A\in\mathbb{R}^{m\times n}$

and:

$B\in\mathbb{R}^{n\times p}$

the result is:

$C=AB\in\mathbb{R}^{m\times p}$

The dimensions must satisfy:

$\boxed{\text{A.columns}=\text{B.rows}}$

The resulting dimensions are:

$\boxed{\text{C.rows}=\text{A.rows}}$

$\boxed{\text{C.columns}=\text{B.columns}}$

## **Example**

$\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\begin{bmatrix}7&8\\9&10\\11&12\end{bmatrix}=\begin{bmatrix}58&64\\139&154\end{bmatrix}$

### **How to calculate the first element**

$C\_{00}=(1\times7)+(2\times9)+(3\times11)$

$C\_{00}=7+18+33=58$

### **How to calculate the second element**

$C\_{01}=(1\times8)+(2\times10)+(3\times12)$

$C\_{01}=8+20+36=64$

Continue the same process for every element.

## **Your assignment**

Implement matrix multiplication.

Requirements:

- Validate dimensions.
- Correctly handle non-square matrices.
- Support a `2 × 3` multiplied by a `3 × 2`.
- Support a `3 × 1` multiplied by a `1 × 3`.
- Support a `1 × 1` multiplication.
- Handle empty dimensions according to your policy.
- Protect against allocation failures.
- Avoid unnecessary memory allocations inside the inner loop.

### **Performance challenge**

Your first implementation should prioritize correctness.

Later, investigate how loop ordering affects cache performance.

For row-major storage, consider why accessing elements sequentially may be more efficient than jumping around memory.

# **7. Transpose**

Transpose swaps the rows and columns of a matrix.

$A^T\_{ij}=A\_{ji}$

Example:

$\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}^{T}=\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}$

Implement:

- New matrix transpose.
- In-place transpose (bonus, especially for square matrices).

### **Test cases**

- Transpose `2 × 3`.
- Transpose `3 × 2`.
- Transpose `1 × 1`.
- Transpose square matrix.
- Transpose zero-sized matrix.
- Transpose twice and verify original values.

Mathematical property:

$(A^T)^T=A$

This is an excellent property-based test.

# **8. Row and column operations**

Implement operations that manipulate complete rows or columns.

### **Required**

- Extract a row into a vector.
- Extract a column into a vector.
- Set an entire row.
- Set an entire column.
- Swap rows.
- Swap columns.

Example:

$A= \begin{bmatrix} 1&2&3\\\ 4&5&6\\\ 7&8&9 \end{bmatrix}$

Extract row 1:

$[4,5,6]$

Extract column 2:

$[3,6,9]$

Swap row 0 and row 2:

$\begin{bmatrix} 7&8&9\\\ 4&5&6\\\ 1&2&3 \end{bmatrix}$

### **Test cases**

- Extract every row.
- Extract every column.
- Set a row.
- Set a column.
- Swap rows.
- Swap columns.
- Invalid row or column.
- Mismatched vector dimensions.

# **9. Matrix-vector multiplication**

This operation bridges your vector and matrix engines.

Given:

$A\in\mathbb{R}^{m\times n}$

and:

$\mathbf{x}\in\mathbb{R}^{n}$

the result is:

$\mathbf{y}=A\mathbf{x}$

where:

$\mathbf{y}\in\mathbb{R}^{m}$

Example:

$\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\begin{bmatrix}2\\3\\4\end{bmatrix}=\begin{bmatrix}20\\47\end{bmatrix}$

Calculation:

$y_0=1(2)+2(3)+3(4)=20$

$y_1=4(2)+5(3)+6(4)=47$

Implement:

- Matrix × vector
- Vector × matrix (bonus)

This will later be extremely useful for machine learning inference.

# **10. Diagonal operations**

Implement:

- Extract diagonal.
- Set diagonal.
- Create diagonal matrix from a vector.
- Sum diagonal elements (trace).

For example:

$A= \begin{bmatrix} 1&2&3\\\ 4&5&6\\\ 7&8&9 \end{bmatrix}$

Diagonal:

$[1,5,9]$

Trace:

$\operatorname{tr}(A)=1+5+9=15$

For non-square matrices, the diagonal length should be:

$\min(\text{rows},\text{columns})$

# **11. Matrix statistics**

Implement:

| Operation      | Description                   |
| -------------- | ----------------------------- |
| Sum            | Sum of all elements           |
| Mean           | Average of all elements       |
| Min            | Smallest element              |
| Max            | Largest element               |
| Frobenius norm | Square root of sum of squares |
| Row sums       | Sum of each row               |
| Column sums    | Sum of each column            |
| Row means      | Average of each row           |
| Column means   | Average of each column        |

The Frobenius norm is:

$\\|A\\|\_F=\sqrt{\sum\_{i=0}^{m-1}\sum\_{j=0}^{n-1}A\_{ij}^2}$

For example:

$A= \begin{bmatrix} 1&2\\\ 3&4 \end{bmatrix}$

$\\|A\\|\_F=\sqrt{30}$

# **12. Matrix properties**

Implement functions that determine whether a matrix satisfies certain properties.

### **Required**

- Is square?
- Is symmetric?
- Is diagonal?
- Is identity?
- Is all-zero?
- Is all-one?

### **Bonus**

- Is upper triangular?
- Is lower triangular?
- Is orthogonal?
- Is positive definite? (More advanced; consider limitations)

## **Symmetric matrix**

$A=A^T$

Example:

$\begin{bmatrix} 1&2&3\\\ 2&4&5\\\ 3&5&6 \end{bmatrix}$

This matrix is symmetric.

Use a floating-point tolerance for comparisons.

# **13. Advanced challenges**

These are optional, but they're excellent preparation for later phases.

## **A. Broadcasting**

Allow operations such as:

- Adding a vector to every row.
- Adding a vector to every column.
- Multiplying each row by a scalar.
- Multiplying each column by a scalar.

This is common in NumPy, PyTorch, and machine learning frameworks.

## **B. Submatrix extraction**

Extract a region of a matrix.

Example:

Extract rows 1–2 and columns 1–2 from:

$\begin{bmatrix} 1&2&3&4\\\ 5&6&7&8\\\ 9&10&11&12\\\ 13&14&15&16 \end{bmatrix}$

Result:

$\begin{bmatrix} 6&7\\\ 10&11 \end{bmatrix}$

## **C. Concatenation**

Implement:

- Horizontal concatenation
- Vertical concatenation

## **D. Matrix comparison**

Implement:

- Exact equality
- Approximate equality with tolerance
- Maximum absolute difference

## **E. Matrix formatting**

Create a clean printing function for debugging.

Example:

```
Matrix (3 x 3):
┌             ┐
│  1.000  2.000  3.000 │
│  4.000  5.000  6.000 │
│  7.000  8.000  9.000 │
└             ┘
```

# **14. API design challenge**

Design your public interface in `include/cml/matrix.h`.

Suggested API categories:

**Lifecycle**

- Create
- Create filled
- Zeros
- Ones
- Identity
- Clone
- Resize
- Clear
- Destroy

**Access**

- Get
- Set
- Fill
- Swap
- Row count
- Column count
- Raw data access

**Arithmetic**

- Add
- Subtract
- Hadamard multiply
- Element-wise divide
- Scalar multiply
- Scalar add
- Negate

**Linear algebra**

- Matrix multiply
- Matrix-vector multiply
- Transpose
- Row extraction
- Column extraction
- Row/column swapping
- Diagonal extraction
- Trace

**Statistics**

- Sum
- Mean
- Min
- Max
- Frobenius norm
- Row/column statistics

**Properties**

- Square
- Symmetric
- Diagonal
- Identity
- Zero
- Equality

### **Your design decisions**

Think carefully about:

1. Should `CMLMatrix` be publicly exposed or opaque?
2. Should your matrix data be contiguous?
3. Should matrix multiplication return a new matrix or modify an existing one?
4. Should you support aliasing?
5. How should dimension mismatches be reported?
6. How should you handle zero rows or columns?
7. Should matrix views be supported?
8. Should rows and columns be represented by vectors?
9. What should resize do to existing data?
10. How should you prevent dimension multiplication overflow?

Don't copy the API of another library blindly. Design CML's API intentionally.

# **15. Comprehensive testing**

Create `tests/test_matrix.c`.

Organize the tests into independent groups:

```
test_matrix_lifecycle()
test_matrix_access()
test_matrix_arithmetic()
test_matrix_multiplication()
test_matrix_transpose()
test_matrix_rows_columns()
test_matrix_vector_multiplication()
test_matrix_diagonal()
test_matrix_statistics()
test_matrix_properties()
test_matrix_edge_cases()
```

## **Minimum test coverage**

- Matrix creation
- Zero matrix
- One matrix
- Identity matrix
- Clone
- Destroy
- Resize
- Fill
- Get/set
- Bounds checking
- Addition
- Subtraction
- Element-wise multiplication
- Element-wise division
- Scalar multiplication
- Scalar addition
- Negation
- Matrix multiplication
- Non-square multiplication
- Transpose
- Double transpose
- Extract row
- Extract column
- Set row
- Set column
- Swap rows
- Swap columns
- Matrix-vector multiplication
- Diagonal extraction
- Trace
- Sum
- Mean
- Min/max
- Frobenius norm
- Row sums
- Column sums
- Symmetric detection
- Identity detection
- Zero matrix detection
- Dimension mismatch
- Overflow dimensions
- Empty matrix
- Memory allocation failure
- Aliasing
- NaN and infinity behavior

# **16. Numerical accuracy**

For floating-point operations, use tolerance-based comparisons.

Instead of:

```
actual == expected
```

Use a test that checks:

$|actual-expected|\leq\epsilon$

For example:

$\epsilon=10^{-9}$

Consider whether the tolerance should scale with the magnitude of the values.

For example, a comparison between `0.000000001` and `0.000000002` is different from a comparison between `1000000000.0` and `1000000000.1`.

Investigate:

- Absolute tolerance
- Relative tolerance
- Floating-point precision
- Accumulation errors
- Numerical stability
- `DBL_EPSILON`

This will matter significantly once we implement gradient descent.

# **17. Performance benchmarks**

Once your matrix implementation is correct, benchmark it.

Test matrices of the following sizes:

| Dimensions    | Elements  |
| ------------- | --------- |
| 10 × 10       | 100       |
| 100 × 100     | 10,000    |
| 500 × 500     | 250,000   |
| 1,000 × 1,000 | 1,000,000 |
| 2,000 × 2,000 | 4,000,000 |

Benchmark:

- Matrix creation
- Matrix addition
- Matrix transpose
- Matrix multiplication
- Matrix-vector multiplication
- Clone
- Resize

### **Performance investigation**

Explore:

- Cache locality
- Row-major storage
- Loop ordering
- Memory allocations
- Compiler optimization
- `-O2` and `-O3`
- SIMD
- Blocking / tiling

For matrix multiplication, investigate why this:

```
i → j → k
```

may perform differently from:

```
i → k → j
```

Don't optimize prematurely. First produce a correct baseline and measure it.

# **18. Build integration**

Update your Makefile.

You should eventually support:

```
make
make test
make test-matrix
make example
make benchmark
make clean
```

Example compilation:

```
cc -Wall -Wextra -Wpedantic -std=c11 \
    -I./include \
    src/memory.c \
    src/error.c \
    src/vector.c \
    src/matrix.c \
    tests/test_matrix.c \
    -lm \
    -o test_matrix
```

Run:

```
./test_matrix
```

With sanitizers:

```
cc -Wall -Wextra -Wpedantic -std=c11 \
    -fsanitize=address,undefined \
    -g -I./include \
    src/memory.c \
    src/error.c \
    src/vector.c \
    src/matrix.c \
    tests/test_matrix.c \
    -lm \
    -o test_matrix
```

# **19. Evaluation rubric**

| Category                              | Points  |
| ------------------------------------- | ------- |
| Matrix architecture and lifecycle     | 15      |
| Memory safety and overflow protection | 15      |
| Element access and manipulation       | 10      |
| Arithmetic                            | 10      |
| Matrix multiplication                 | 15      |
| Transpose and linear algebra          | 10      |
| Statistics and properties             | 10      |
| Testing and edge cases                | 10      |
| Documentation and code quality        | 5       |
| **Total**                             | **100** |

# **20. Milestones**

Don't attempt to implement the entire matrix library at once. Build it incrementally.

1. **Milestone 1 — Matrix structure and API design**
2. **Milestone 2 — Lifecycle and memory safety**
3. **Milestone 3 — Element access**
4. **Milestone 4 — Basic arithmetic**
5. **Milestone 5 — Matrix multiplication**
6. **Milestone 6 — Transpose**
7. **Milestone 7 — Row and column operations**
8. **Milestone 8 — Matrix-vector operations**
9. **Milestone 9 — Statistics and properties**
10. **Milestone 10 — Tests, benchmarks, optimization**

## **Your first task: Milestone 1**

Before writing `matrix.c`, design your structure and API.

Send me:

1. Your proposed `CMLMatrix` struct.
2. Your `matrix.h` header.
3. Your decision on row-major memory layout.
4. Your memory ownership rules.
5. How you'll handle zero-sized matrices.
6. How you'll handle dimension overflow.

I'll review your design, challenge any weaknesses, and help you refine it **without giving away the implementation**.

Once your matrix engine is complete, we'll move to **Phase 4 — Statistics & Data Processing**, where you'll begin implementing data normalization, datasets, feature scaling, data splitting, and preprocessing tools for actual machine learning. 🚀
