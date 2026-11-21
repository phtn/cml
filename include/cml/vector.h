#ifndef CML_VECTOR_H
#define CML_VECTOR_H

/*
 * cml/vector.h — CML Vector Engine
 *
 * Core vector abstraction for the C Machine Learning Library.
 * Supports:
 *   - Dynamic lifecycle (create, initialize, resize, clear, clone, destroy).
 *   - Element access, bounds checking, fill, and swap.
 *   - Vector arithmetic (addition, subtraction, element-wise mul/div, scaling,
 * negation).
 *   - Both out-of-place and in-place arithmetic (with safe aliasing support).
 *   - Advanced operations (dot product, L1/L2/infinity norms, unit
 * normalization).
 *   - Descriptive statistics (sum, mean, min, max, sample/population variance
 * and stddev).
 *   - Integration with CMLStatus error codes and CML memory management.
 */

#include "cml/error.h"

#include <stdbool.h>
#include <stddef.h>

/* ===========================================================================
 * CMLVector Structure
 *
 * Contiguous heap-allocated array of double-precision floating-point numbers.
 * Fields:
 *   data     — pointer to contiguous array of doubles (owned by the vector).
 *   size     — number of active elements in the vector.
 *   capacity — allocated capacity in number of doubles.
 * ========================================================================= */

typedef struct CMLVector {
  double *data;
  size_t size;
  size_t capacity;
} CMLVector;

/* ===========================================================================
 * Lifecycle & Memory Management
 * ========================================================================= */

/*
 * Create a vector of given size with all elements initialized to 0.0.
 * A size of 0 creates a valid empty vector with NULL data.
 * Returns NULL on allocation failure or size overflow.
 */
CMLVector *cml_vector_create(size_t size);

/*
 * Create a vector of given size with all elements initialized to `value`.
 * Returns NULL on allocation failure or size overflow.
 */
CMLVector *cml_vector_create_filled(size_t size, double value);

/*
 * Create a vector by copying data from a raw C array.
 * If size == 0, returns a valid empty vector.
 * Returns NULL on allocation failure or if array is NULL when size > 0.
 */
CMLVector *cml_vector_from_array(const double *array, size_t size);

/*
 * Create an independent deep copy of `src`.
 * Returns NULL if src is NULL or if allocation fails.
 */
CMLVector *cml_vector_clone(const CMLVector *src);

/*
 * Resize `vec` to `new_size`.
 * If growing, new elements are initialized to 0.0.
 * If resizing fails, the original data and size remain valid and unchanged.
 */
CMLStatus cml_vector_resize(CMLVector *vec, size_t new_size);

/*
 * Reset vector length to 0 while preserving allocated capacity for future
 * reuse.
 */
CMLStatus cml_vector_clear(CMLVector *vec);

/*
 * Release all memory owned by `vec`.
 * Safe no-op if `vec` is NULL.
 */
void cml_vector_destroy(CMLVector *vec);

/* ===========================================================================
 * Element Access & Mutation
 * ========================================================================= */

/*
 * Retrieve element at `index`.
 * Returns CML_ERROR_INDEX_OUT_OF_BOUNDS if index >= vec->size.
 */
CMLStatus cml_vector_get(const CMLVector *vec, size_t index, double *out);

/*
 * Set element at `index` to `value`.
 * Returns CML_ERROR_INDEX_OUT_OF_BOUNDS if index >= vec->size.
 */
CMLStatus cml_vector_set(CMLVector *vec, size_t index, double value);

/*
 * Fill all elements of `vec` with `value`.
 */
CMLStatus cml_vector_fill(CMLVector *vec, double value);

/*
 * Swap elements at indices `i` and `j`.
 * Swapping an index with itself is a safe no-op.
 */
CMLStatus cml_vector_swap(CMLVector *vec, size_t i, size_t j);

/* ===========================================================================
 * Arithmetic Operations (Supports Safe Aliasing: out == a or out == b)
 * ========================================================================= */

/* out = a + b */
CMLStatus cml_vector_add(const CMLVector *a, const CMLVector *b,
                         CMLVector *out);

/* out = a - b */
CMLStatus cml_vector_sub(const CMLVector *a, const CMLVector *b,
                         CMLVector *out);

/* out = a * b (element-wise / Hadamard product) */
CMLStatus cml_vector_mul(const CMLVector *a, const CMLVector *b,
                         CMLVector *out);

/* out = a / b (element-wise division; returns CML_ERROR_NUMERICAL if any b[i]
 * == 0.0) */
CMLStatus cml_vector_div(const CMLVector *a, const CMLVector *b,
                         CMLVector *out);

/* out = a * scalar */
CMLStatus cml_vector_scale(const CMLVector *a, double scalar, CMLVector *out);

/* out = a + scalar */
CMLStatus cml_vector_add_scalar(const CMLVector *a, double scalar,
                                CMLVector *out);

/* out = -a */
CMLStatus cml_vector_negate(const CMLVector *a, CMLVector *out);

/* ===========================================================================
 * In-Place Arithmetic
 * ========================================================================= */

/* a += b */
CMLStatus cml_vector_add_inplace(CMLVector *a, const CMLVector *b);

/* a -= b */
CMLStatus cml_vector_sub_inplace(CMLVector *a, const CMLVector *b);

/* a *= b (element-wise) */
CMLStatus cml_vector_mul_inplace(CMLVector *a, const CMLVector *b);

/* a /= b (element-wise) */
CMLStatus cml_vector_div_inplace(CMLVector *a, const CMLVector *b);

/* a *= scalar */
CMLStatus cml_vector_scale_inplace(CMLVector *a, double scalar);

/* a += scalar */
CMLStatus cml_vector_add_scalar_inplace(CMLVector *a, double scalar);

/* a = -a */
CMLStatus cml_vector_negate_inplace(CMLVector *a);

/* ===========================================================================
 * Allocating Arithmetic Helpers (convenience functions returning new vectors)
 * ========================================================================= */

CMLVector *cml_vector_add_new(const CMLVector *a, const CMLVector *b);
CMLVector *cml_vector_sub_new(const CMLVector *a, const CMLVector *b);
CMLVector *cml_vector_mul_new(const CMLVector *a, const CMLVector *b);
CMLVector *cml_vector_div_new(const CMLVector *a, const CMLVector *b);
CMLVector *cml_vector_scale_new(const CMLVector *a, double scalar);
CMLVector *cml_vector_add_scalar_new(const CMLVector *a, double scalar);
CMLVector *cml_vector_negate_new(const CMLVector *a);

/* ===========================================================================
 * Advanced Vector Operations
 * ========================================================================= */

/*
 * Dot product: a . b = sum(a[i] * b[i]).
 * Empty vectors evaluate to 0.0.
 */
CMLStatus cml_vector_dot(const CMLVector *a, const CMLVector *b, double *out);

/*
 * L1 norm (Manhattan): sum(|v[i]|).
 */
CMLStatus cml_vector_norm_l1(const CMLVector *vec, double *out);

/*
 * L2 norm (Euclidean magnitude): sqrt(sum(v[i]^2)).
 */
CMLStatus cml_vector_norm_l2(const CMLVector *vec, double *out);

/*
 * General norm alias (defaults to L2 Euclidean norm).
 */
CMLStatus cml_vector_norm(const CMLVector *vec, double *out);

/*
 * Infinity norm (Maximum absolute value): max(|v[i]|).
 */
CMLStatus cml_vector_norm_inf(const CMLVector *vec, double *out);

/*
 * Normalize vector to unit length (out = vec / ||vec||_2).
 * Returns CML_ERROR_NUMERICAL if ||vec||_2 is zero or non-finite.
 * Safely supports aliasing (out == vec).
 */
CMLStatus cml_vector_normalize(const CMLVector *vec, CMLVector *out);

/* In-place normalization */
CMLStatus cml_vector_normalize_inplace(CMLVector *vec);

/* Allocating normalization */
CMLVector *cml_vector_normalize_new(const CMLVector *vec);

/* ===========================================================================
 * Statistics
 * ========================================================================= */

/* Sum of all elements */
CMLStatus cml_vector_sum(const CMLVector *vec, double *out);

/* Arithmetic mean: sum / n (returns CML_ERROR_INVALID_DIMENSION if size == 0)
 */
CMLStatus cml_vector_mean(const CMLVector *vec, double *out);

/* Minimum element (returns CML_ERROR_INVALID_DIMENSION if size == 0) */
CMLStatus cml_vector_min(const CMLVector *vec, double *out);

/* Maximum element (returns CML_ERROR_INVALID_DIMENSION if size == 0) */
CMLStatus cml_vector_max(const CMLVector *vec, double *out);

/* Both minimum and maximum in a single pass */
CMLStatus cml_vector_min_max(const CMLVector *vec, double *out_min,
                             double *out_max);

/*
 * Variance:
 *   is_sample == 0 -> Population variance (divisor n)
 *   is_sample != 0 -> Sample variance (divisor n - 1; requires size >= 2)
 */
CMLStatus cml_vector_variance(const CMLVector *vec, int is_sample, double *out);
CMLStatus cml_vector_variance_pop(const CMLVector *vec, double *out);
CMLStatus cml_vector_variance_sample(const CMLVector *vec, double *out);

/* Standard deviation (sqrt of variance) */
CMLStatus cml_vector_stddev(const CMLVector *vec, int is_sample, double *out);
CMLStatus cml_vector_stddev_pop(const CMLVector *vec, double *out);
CMLStatus cml_vector_stddev_sample(const CMLVector *vec, double *out);

/* ===========================================================================
 * Utilities
 * ========================================================================= */

/*
 * Compare two vectors element-by-element with tolerance `tol`.
 * Returns true if equal within tolerance, false otherwise.
 */
bool cml_vector_is_equal(const CMLVector *a, const CMLVector *b, double tol);

/*
 * Print vector to stdout in format:
 *   [name] = [v0, v1, ..., vN-1]
 */
void cml_vector_print(const CMLVector *vec, const char *name);

#endif /* CML_VECTOR_H */
