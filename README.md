# cml — C Machine Learning Library

A from-scratch machine learning library written in C11.

## Phase 0 · Assignment 01 — Memory Module

### Build

```sh
make
```

Compile with sanitizers during development:

```sh
make sanitize
```

### Test

```sh
make test
```

---

## Memory API Reference

All public functions are declared in [`include/cml/memory.h`](include/cml/memory.h)
and implemented in [`src/memory.c`](src/memory.c).

### Design conventions

| Convention | Detail |
|------------|--------|
| Sizes | All byte counts use `size_t` |
| Failure | Functions return `NULL`; they never abort or print messages |
| Zero sizes | Zero-sized requests consistently return `NULL` |
| Ownership | The caller owns every pointer and must free it with `cml_free` |

---

### `cml_malloc`

Allocate uninitialised memory.

```c
void *cml_malloc(size_t size);
```

| Parameter | Description |
|-----------|-------------|
| `size`    | Number of bytes to allocate (must be > 0) |

**Returns:** pointer to allocated memory, or `NULL` on failure or if `size == 0`.

**Edge cases:**
- `size == 0` → returns `NULL` (consistent, no undefined behaviour).

**Example:**

```c
int *buf = cml_malloc(10 * sizeof(int));
if (!buf) { /* handle failure */ }
/* use buf ... */
cml_free(buf);
```

---

### `cml_calloc`

Allocate a zero-initialised array. Detects multiplication overflow.

```c
void *cml_calloc(size_t count, size_t element_size);
```

| Parameter      | Description |
|----------------|-------------|
| `count`        | Number of elements (must be > 0) |
| `element_size` | Size of each element in bytes (must be > 0) |

**Returns:** pointer to zero-initialised memory, or `NULL` on failure.

**Failure causes:** `count == 0`, `element_size == 0`, `count * element_size` overflows `SIZE_MAX`, or the system allocation fails.

**Example:**

```c
double *weights = cml_calloc(128, sizeof(double));
if (!weights) { /* handle failure */ }
/* weights[i] == 0.0 guaranteed */
cml_free(weights);
```

---

### `cml_realloc`

Resize an existing allocation. Safe on failure — the original block is preserved.

```c
void *cml_realloc(void *ptr, size_t new_size);
```

| Parameter  | Description |
|------------|-------------|
| `ptr`      | Existing allocation, or `NULL` (acts like `cml_malloc`) |
| `new_size` | Desired new size in bytes |

**Returns:** pointer to resized memory on success. `NULL` if `new_size == 0` (frees `ptr`) or if allocation fails.

**Critical pattern — never overwrite the original pointer directly:**

```c
/* WRONG — leaks ptr if realloc fails */
ptr = cml_realloc(ptr, new_size);

/* CORRECT */
void *tmp = cml_realloc(ptr, new_size);
if (!tmp) {
    /* ptr is still valid; handle the error */
    cml_free(ptr);
    return NULL;
}
ptr = tmp;
```

**Edge cases:**
- `new_size == 0` → frees `ptr` and returns `NULL`.
- `ptr == NULL` → equivalent to `cml_malloc(new_size)`.

---

### `cml_free`

Release allocated memory.

```c
void cml_free(void *ptr);
```

| Parameter | Description |
|-----------|-------------|
| `ptr`     | Pointer to free, or `NULL` (safe no-op) |

**Edge cases:**
- `ptr == NULL` → safe, no effect.
- **Double-free is undefined behaviour.** Do not call `cml_free` twice on the same pointer.

**Ownership rule:** after `cml_free(ptr)`, `ptr` is invalid and must not be read or written.

---

### `cml_mem_copy`

Copy bytes between non-overlapping memory regions.

```c
void *cml_mem_copy(void *dst, const void *src, size_t n);
```

| Parameter | Description |
|-----------|-------------|
| `dst`     | Destination buffer (at least `n` writable bytes) |
| `src`     | Source buffer (at least `n` readable bytes) |
| `n`       | Number of bytes to copy |

**Returns:** `dst` on success, `NULL` if `dst == NULL`, `src == NULL`, or `n == 0`.

**Overlap:** regions must NOT overlap. This function uses `memcpy`, which is undefined for overlapping regions. For overlapping copies use `memmove` directly.

**Example:**

```c
float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
float b[4];
cml_mem_copy(b, a, 4 * sizeof(float));
/* b now contains {1.0f, 2.0f, 3.0f, 4.0f} */
```

---

## Compiler flags

| Flag | Purpose |
|------|---------|
| `-Wall -Wextra -Wpedantic` | Standard warnings |
| `-Wconversion` | Implicit conversion warnings |
| `-Wshadow` | Variable shadowing |
| `-fsanitize=address,undefined` | Runtime memory/UB detection (dev only) |
