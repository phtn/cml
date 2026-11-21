# cml — C Machine Learning Library

A from-scratch machine learning library written in C11.

```
CML
├── Memory       → allocates memory safely (Phase 0)
├── Error        → standardizes failures & logging (Phase 1)
├── Math         → numerical foundation & tolerances (Phase 2)
├── Vector       → vector engine & linear algebra (Phase 2)
├── Matrix       → 2D numerical data structures (Phase 3)
└── ...
```

---

## Build & Test

```sh
# Build default target
make

# Run all test suites (memory, error, vector)
make test

# Run specific test suites
make test-memory
make test-error
make test-vector

# Run the Phase 2 Vector Engine example
make example

# Run with AddressSanitizer & UndefinedBehaviorSanitizer
make sanitize
```

---

## Phase 2 · Vector Engine

Declared in [`include/cml/vector.h`](include/cml/vector.h) and [`include/cml/math.h`](include/cml/math.h), implemented in [`src/vector.c`](src/vector.c) and [`src/math.c`](src/math.c).

### `CMLVector` Structure

```c
typedef struct CMLVector {
    double *data;     /* contiguous dynamically allocated array */
    size_t size;      /* number of active elements */
    size_t capacity;  /* allocated buffer capacity */
} CMLVector;
```

### Key Capabilities

1. **Lifecycle & Memory Safety:**
   - `cml_vector_create(size)` — allocates contiguous memory zero-initialized via `cml_calloc`.
   - `cml_vector_create_filled(size, value)` — creates vector filled with `value`.
   - `cml_vector_from_array(arr, size)` — creates vector from raw C array.
   - `cml_vector_clone(src)` — deep copy with independent storage.
   - `cml_vector_resize(vec, new_size)` — safe reallocation preserving data; zeroes new elements.
   - `cml_vector_clear(vec)` — resets length to 0 while keeping capacity for reuse.
   - `cml_vector_destroy(vec)` — safely releases buffer and vector header.

2. **Element Access & Mutation:**
   - `cml_vector_get`, `cml_vector_set` — bounds-checked access returning `CML_ERROR_INDEX_OUT_OF_BOUNDS`.
   - `cml_vector_fill` — fills all elements.
   - `cml_vector_swap` — swaps two elements with bounds checking.

3. **Arithmetic Operations (Safe Aliasing Supported):**
   - Addition (`cml_vector_add`), Subtraction (`cml_vector_sub`), Element-wise multiplication (`cml_vector_mul`).
   - Element-wise division (`cml_vector_div`) with division-by-zero detection returning `CML_ERROR_NUMERICAL`.
   - Scalar operations: `cml_vector_scale`, `cml_vector_add_scalar`, `cml_vector_negate`.
   - In-place variations: `cml_vector_add_inplace`, `cml_vector_sub_inplace`, etc.
   - Allocating helpers: `cml_vector_add_new`, `cml_vector_sub_new`, etc.

4. **Advanced Operations & Norms:**
   - Dot product (`cml_vector_dot`): $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i$.
   - L1 norm (`cml_vector_norm_l1`): $\sum |v_i|$.
   - L2 Euclidean magnitude (`cml_vector_norm_l2`, `cml_vector_norm`): $\sqrt{\sum v_i^2}$.
   - Infinity norm (`cml_vector_norm_inf`): $\max |v_i|$.
   - Normalization (`cml_vector_normalize`): $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|_2$ (returns `CML_ERROR_NUMERICAL` on zero-vector).

5. **Descriptive Statistics:**
   - `cml_vector_sum`, `cml_vector_mean`, `cml_vector_min`, `cml_vector_max`.
   - `cml_vector_variance_pop` & `cml_vector_variance_sample` (divisors $n$ vs $n-1$).
   - `cml_vector_stddev_pop` & `cml_vector_stddev_sample`.

---

## Phase 1 · Error & Logging Module

Declared in [`include/cml/error.h`](include/cml/error.h), implemented in [`src/error.c`](src/error.c).
- Standardized `CMLStatus` error codes.
- `cml_status_string()` for human-readable error descriptions.
- `cml_log()` printf-style variadic logger with severity levels (`DEBUG`, `INFO`, `WARN`, `ERROR`, `FATAL`).
- Runtime log level filtering (`cml_log_set_level()`).
- File logging & mirroring (`cml_log_set_file()`, `cml_log_close_file()`).

---

## Phase 0 · Memory Module

Declared in [`include/cml/memory.h`](include/cml/memory.h), implemented in [`src/memory.c`](src/memory.c).
- `cml_malloc`, `cml_calloc`, `cml_realloc`, `cml_free`, `cml_mem_copy`.
- Safe failure handling and multiplication overflow protection.
