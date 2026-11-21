#ifndef CML_MATRIX_H
#define CML_MATRIX_H

/* ==============================================================================
 * cml/matrix.h — CML Matrix Engine
 *
 * Core two-dimensional numerical engine for the C Machine Learning Library.
 * Supports:
 *   - Contiguous row-major heap storage with dynamic lifecycle management.
 *   - Element access, bounds checking, fill, and element/row/col swapping.
 *   - Complete matrix arithmetic (addition, subtraction, Hadamard product,
 *     element-wise division, scaling, scalar addition, negation).
 *   - In-place arithmetic and safe aliasing support.
 *   - Cache-optimized matrix multiplication (matmul).
 *   - Transpose operations (out-of-place and in-place).
 *   - Row and column extraction and vector conversions (via CMLVector).
 *   - Matrix-vector and vector-matrix products.
 *   - Diagonal operations and matrix trace.
 *   - Matrix statistics (sum, mean, min, max, Frobenius norm, row/col stats).
 *   - Structural and numerical matrix property validation (symmetry, identity,
 *     orthogonality, triangularity, etc.).
 *   - Advanced utilities: submatrix slicing, horizontal/vertical concatenation,
 *     row/column broadcasting, tolerance-based equality, and formatted
 * printing.
 * ------------------------------------------------------------------------------
 */

#include "cml/error.h"
#include "cml/vector.h"

#include <stdbool.h>
#include <stddef.h>

/* ==============================================================================
 * CMLMatrix Structure
 *
 * Contiguous heap-allocated row-major 2D array of double-precision numbers.
 * An element at row `r` and column `c` is mapped to linear index:
 *   index = r * cols + c
 *
 * Fields:
 *   data     — pointer to contiguous array of doubles (owned by the matrix).
 *   rows     — number of rows in the matrix.
 *   cols     — number of columns in the matrix.
 *   capacity — total allocated capacity in number of double elements.
 * ------------------------------------------------------------------------------
 */
typedef struct CMLMatrix {
  double *data;
  size_t rows;
  size_t cols;
  size_t capacity;
} CMLMatrix;

/* ==============================================================================
 * Lifecycle & Memory Management
 * ------------------------------------------------------------------------------
 */

/*
 * Create a matrix of dimensions `rows` x `cols` with all elements
 * zero-initialized. Zero-sized matrices (rows == 0 or cols == 0) return a valid
 * matrix with NULL data. Returns NULL on allocation failure or integer overflow
 * in dimensions.
 */
CMLMatrix *cml_matrix_create(size_t rows, size_t cols);

/*
 * Create a matrix of dimensions `rows` x `cols` filled with `value`.
 * Returns NULL on allocation failure or integer overflow in dimensions.
 */
CMLMatrix *cml_matrix_create_filled(size_t rows, size_t cols, double value);

/*
 * Allocate a matrix filled with zeros (convenience alias for
 * cml_matrix_create).
 */
CMLMatrix *cml_matrix_zeros(size_t rows, size_t cols);

/*
 * Allocate a matrix filled with ones (1.0).
 */
CMLMatrix *cml_matrix_ones(size_t rows, size_t cols);

/*
 * Allocate an `n` x `n` identity matrix (diagonal elements = 1.0, off-diagonal
 * = 0.0). If n == 0, returns a valid empty matrix.
 */
CMLMatrix *cml_matrix_identity(size_t n);

/*
 * Create a matrix from a raw contiguous row-major double array.
 * If rows == 0 or cols == 0, returns a valid empty matrix.
 * Returns NULL on allocation failure, overflow, or if array is NULL when
 * rows*cols > 0.
 */
CMLMatrix *cml_matrix_from_array(const double *array, size_t rows, size_t cols);

/*
 * Create an independent deep copy of `src`.
 * Returns NULL if src is NULL or if allocation fails.
 */
CMLMatrix *cml_matrix_clone(const CMLMatrix *src);

/*
 * Resize `mat` to `new_rows` x `new_cols`.
 * Preserves existing 2D grid contents up to min(old_rows, new_rows) and
 * min(old_cols, new_cols). Newly added elements are initialized to 0.0.
 * If resizing fails, original matrix data and dimensions remain intact.
 */
CMLStatus cml_matrix_resize(CMLMatrix *mat, size_t new_rows, size_t new_cols);

/*
 * Reset matrix dimensions to 0 x 0 while retaining allocated memory for future
 * use.
 */
CMLStatus cml_matrix_clear(CMLMatrix *mat);

/*
 * Release all memory owned by `mat`. Safe no-op if `mat` is NULL.
 */
void cml_matrix_destroy(CMLMatrix *mat);

/* ==============================================================================
 * Element Access & Mutation
 * ------------------------------------------------------------------------------
 */

/*
 * Retrieve element at (row, col).
 * Returns CML_ERROR_INDEX_OUT_OF_BOUNDS if row >= mat->rows or col >=
 * mat->cols.
 */
CMLStatus cml_matrix_get(const CMLMatrix *mat, size_t row, size_t col,
                         double *out);

/*
 * Set element at (row, col) to `value`.
 * Returns CML_ERROR_INDEX_OUT_OF_BOUNDS if row >= mat->rows or col >=
 * mat->cols.
 */
CMLStatus cml_matrix_set(CMLMatrix *mat, size_t row, size_t col, double value);

/*
 * Fill all elements of `mat` with `value`.
 */
CMLStatus cml_matrix_fill(CMLMatrix *mat, double value);

/*
 * Swap two elements (r1, c1) and (r2, c2). Swapping an element with itself is a
 * no-op.
 */
CMLStatus cml_matrix_swap_elements(CMLMatrix *mat, size_t r1, size_t c1,
                                   size_t r2, size_t c2);

/*
 * Safe dimension and buffer accessors.
 */
size_t cml_matrix_rows(const CMLMatrix *mat);
size_t cml_matrix_cols(const CMLMatrix *mat);
double *cml_matrix_data(CMLMatrix *mat);
const double *cml_matrix_data_const(const CMLMatrix *mat);

/* ==============================================================================
 * Matrix Arithmetic (Element-wise, supports safe aliasing: out == a or out ==
 * b)
 * ------------------------------------------------------------------------------
 */

/* out = a + b */
CMLStatus cml_matrix_add(const CMLMatrix *a, const CMLMatrix *b,
                         CMLMatrix *out);

/* out = a - b */
CMLStatus cml_matrix_sub(const CMLMatrix *a, const CMLMatrix *b,
                         CMLMatrix *out);

/* out = a * b (Hadamard / element-wise multiplication) */
CMLStatus cml_matrix_mul(const CMLMatrix *a, const CMLMatrix *b,
                         CMLMatrix *out);

/* out = a / b (Element-wise division; returns CML_ERROR_NUMERICAL if any b
 * element is 0.0 or NaN) */
CMLStatus cml_matrix_div(const CMLMatrix *a, const CMLMatrix *b,
                         CMLMatrix *out);

/* out = a * scalar */
CMLStatus cml_matrix_scale(const CMLMatrix *a, double scalar, CMLMatrix *out);

/* out = a + scalar */
CMLStatus cml_matrix_add_scalar(const CMLMatrix *a, double scalar,
                                CMLMatrix *out);

/* out = -a */
CMLStatus cml_matrix_negate(const CMLMatrix *a, CMLMatrix *out);

/* ==============================================================================
 * In-Place Matrix Arithmetic
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_matrix_add_inplace(CMLMatrix *a, const CMLMatrix *b);
CMLStatus cml_matrix_sub_inplace(CMLMatrix *a, const CMLMatrix *b);
CMLStatus cml_matrix_mul_inplace(CMLMatrix *a, const CMLMatrix *b);
CMLStatus cml_matrix_div_inplace(CMLMatrix *a, const CMLMatrix *b);
CMLStatus cml_matrix_scale_inplace(CMLMatrix *a, double scalar);
CMLStatus cml_matrix_add_scalar_inplace(CMLMatrix *a, double scalar);
CMLStatus cml_matrix_negate_inplace(CMLMatrix *a);

/* ==============================================================================
 * Allocating Arithmetic Helpers
 * ------------------------------------------------------------------------------
 */

CMLMatrix *cml_matrix_add_new(const CMLMatrix *a, const CMLMatrix *b);
CMLMatrix *cml_matrix_sub_new(const CMLMatrix *a, const CMLMatrix *b);
CMLMatrix *cml_matrix_mul_new(const CMLMatrix *a, const CMLMatrix *b);
CMLMatrix *cml_matrix_div_new(const CMLMatrix *a, const CMLMatrix *b);
CMLMatrix *cml_matrix_scale_new(const CMLMatrix *a, double scalar);
CMLMatrix *cml_matrix_add_scalar_new(const CMLMatrix *a, double scalar);
CMLMatrix *cml_matrix_negate_new(const CMLMatrix *a);

/* ==============================================================================
 * Linear Algebra & Matrix Multiplication
 * ------------------------------------------------------------------------------
 */

/*
 * Matrix multiplication: out = a * b
 * a: (m x n), b: (n x p), out: (m x p)
 * Cache-optimized loop structure (i -> k -> j). Fully alias-safe (out == a or
 * out == b).
 */
CMLStatus cml_matrix_matmul(const CMLMatrix *a, const CMLMatrix *b,
                            CMLMatrix *out);

/* Allocating matrix multiplication */
CMLMatrix *cml_matrix_matmul_new(const CMLMatrix *a, const CMLMatrix *b);

/*
 * Matrix transpose: out = src^T
 * src: (m x n), out: (n x m). Fully alias-safe (out == src).
 */
CMLStatus cml_matrix_transpose(const CMLMatrix *src, CMLMatrix *out);

/* Allocating matrix transpose */
CMLMatrix *cml_matrix_transpose_new(const CMLMatrix *src);

/*
 * In-place matrix transpose. For square matrices, swaps elements in place.
 * For rectangular matrices, performs transposed reordering and swaps
 * dimensions.
 */
CMLStatus cml_matrix_transpose_inplace(CMLMatrix *mat);

/* ==============================================================================
 * Row and Column Operations
 * ------------------------------------------------------------------------------
 */

/* Extract row `row` into vector `out` (must have size == mat->cols) */
CMLStatus cml_matrix_get_row(const CMLMatrix *mat, size_t row, CMLVector *out);
CMLVector *cml_matrix_get_row_new(const CMLMatrix *mat, size_t row);

/* Extract column `col` into vector `out` (must have size == mat->rows) */
CMLStatus cml_matrix_get_col(const CMLMatrix *mat, size_t col, CMLVector *out);
CMLVector *cml_matrix_get_col_new(const CMLMatrix *mat, size_t col);

/* Overwrite row `row` with contents of `vec` (must have size == mat->cols) */
CMLStatus cml_matrix_set_row(CMLMatrix *mat, size_t row, const CMLVector *vec);

/* Overwrite column `col` with contents of `vec` (must have size == mat->rows)
 */
CMLStatus cml_matrix_set_col(CMLMatrix *mat, size_t col, const CMLVector *vec);

/* Swap entire row `r1` with row `r2` */
CMLStatus cml_matrix_swap_rows(CMLMatrix *mat, size_t r1, size_t r2);

/* Swap entire column `c1` with column `c2` */
CMLStatus cml_matrix_swap_cols(CMLMatrix *mat, size_t c1, size_t c2);

/* ==============================================================================
 * Matrix-Vector & Vector-Matrix Operations
 * ------------------------------------------------------------------------------
 */

/*
 * Matrix-vector product: out = mat * vec
 * mat: (m x n), vec: (n), out: (m). Fully alias-safe (out == vec).
 */
CMLStatus cml_matrix_vec_mul(const CMLMatrix *mat, const CMLVector *vec,
                             CMLVector *out);
CMLVector *cml_matrix_vec_mul_new(const CMLMatrix *mat, const CMLVector *vec);

/*
 * Vector-matrix product: out = vec^T * mat
 * vec: (m), mat: (m x n), out: (n). Fully alias-safe (out == vec).
 */
CMLStatus cml_vector_mat_mul(const CMLVector *vec, const CMLMatrix *mat,
                             CMLVector *out);
CMLVector *cml_vector_mat_mul_new(const CMLVector *vec, const CMLMatrix *mat);

/* ==============================================================================
 * Diagonal Operations
 * ------------------------------------------------------------------------------
 */

/*
 * Extract the main diagonal into vector `out`.
 * Length of diagonal is min(mat->rows, mat->cols). out->size must match.
 */
CMLStatus cml_matrix_get_diag(const CMLMatrix *mat, CMLVector *out);
CMLVector *cml_matrix_get_diag_new(const CMLMatrix *mat);

/*
 * Set the main diagonal elements of `mat` from `vec`.
 * vec->size must equal min(mat->rows, mat->cols).
 */
CMLStatus cml_matrix_set_diag(CMLMatrix *mat, const CMLVector *vec);

/*
 * Create a square diagonal matrix from a vector.
 * Off-diagonal elements are 0.0.
 */
CMLMatrix *cml_matrix_from_diag(const CMLVector *vec);

/*
 * Sum of diagonal elements (trace): tr(mat) = sum(mat[i, i]).
 * For non-square matrices, sums min(rows, cols) diagonal elements. Empty
 * matrices return 0.0.
 */
CMLStatus cml_matrix_trace(const CMLMatrix *mat, double *out);

/* ==============================================================================
 * Matrix Statistics
 * ------------------------------------------------------------------------------
 */

/* Sum of all matrix elements */
CMLStatus cml_matrix_sum(const CMLMatrix *mat, double *out);

/* Arithmetic mean of all matrix elements (returns CML_ERROR_INVALID_DIMENSION
 * if empty) */
CMLStatus cml_matrix_mean(const CMLMatrix *mat, double *out);

/* Minimum element (returns CML_ERROR_INVALID_DIMENSION if empty) */
CMLStatus cml_matrix_min(const CMLMatrix *mat, double *out);

/* Maximum element (returns CML_ERROR_INVALID_DIMENSION if empty) */
CMLStatus cml_matrix_max(const CMLMatrix *mat, double *out);

/* Both minimum and maximum elements in a single pass */
CMLStatus cml_matrix_min_max(const CMLMatrix *mat, double *out_min,
                             double *out_max);

/* Frobenius norm: sqrt(sum(mat[i, j]^2)) */
CMLStatus cml_matrix_norm_frobenius(const CMLMatrix *mat, double *out);

/* Sum of each row (out must have size == mat->rows) */
CMLStatus cml_matrix_row_sums(const CMLMatrix *mat, CMLVector *out);

/* Sum of each column (out must have size == mat->cols) */
CMLStatus cml_matrix_col_sums(const CMLMatrix *mat, CMLVector *out);

/* Average of each row (out must have size == mat->rows) */
CMLStatus cml_matrix_row_means(const CMLMatrix *mat, CMLVector *out);

/* Average of each column (out must have size == mat->cols) */
CMLStatus cml_matrix_col_means(const CMLMatrix *mat, CMLVector *out);

/* ==============================================================================
 * Matrix Properties & Predicates
 * ------------------------------------------------------------------------------
 */

/* True if rows == cols and rows > 0 */
bool cml_matrix_is_square(const CMLMatrix *mat);

/* True if mat == mat^T within tolerance `tol` */
bool cml_matrix_is_symmetric(const CMLMatrix *mat, double tol);

/* True if square and all off-diagonal elements <= tol */
bool cml_matrix_is_diagonal(const CMLMatrix *mat, double tol);

/* True if square, diagonal elements are 1.0 within tol, and off-diagonal
 * elements <= tol */
bool cml_matrix_is_identity(const CMLMatrix *mat, double tol);

/* True if all elements <= tol */
bool cml_matrix_is_zero(const CMLMatrix *mat, double tol);

/* True if all elements are close to 1.0 within tol */
bool cml_matrix_is_one(const CMLMatrix *mat, double tol);

/* True if all elements below the main diagonal are close to 0.0 */
bool cml_matrix_is_upper_triangular(const CMLMatrix *mat, double tol);

/* True if all elements above the main diagonal are close to 0.0 */
bool cml_matrix_is_lower_triangular(const CMLMatrix *mat, double tol);

/* True if square and mat^T * mat == I within tolerance `tol` */
bool cml_matrix_is_orthogonal(const CMLMatrix *mat, double tol);

/* True if matrices have identical dimensions and element-wise close within tol
 */
bool cml_matrix_is_equal(const CMLMatrix *a, const CMLMatrix *b, double tol);

/* Maximum absolute difference between corresponding elements: max(|a_ij -
 * b_ij|) */
CMLStatus cml_matrix_max_abs_diff(const CMLMatrix *a, const CMLMatrix *b,
                                  double *out);

/* ==============================================================================
 * Advanced Utilities: Submatrix, Concatenation, Broadcasting, and Printing
 * ------------------------------------------------------------------------------
 */

/* Extract submatrix of size row_count x col_count starting at (row_start,
 * col_start) */
CMLStatus cml_matrix_submatrix(const CMLMatrix *src, size_t row_start,
                               size_t row_count, size_t col_start,
                               size_t col_count, CMLMatrix *out);
CMLMatrix *cml_matrix_submatrix_new(const CMLMatrix *src, size_t row_start,
                                    size_t row_count, size_t col_start,
                                    size_t col_count);

/* Horizontal concatenation: out = [A | B] (a and b must have identical row
 * count) */
CMLStatus cml_matrix_concat_h(const CMLMatrix *a, const CMLMatrix *b,
                              CMLMatrix *out);
CMLMatrix *cml_matrix_concat_h_new(const CMLMatrix *a, const CMLMatrix *b);

/* Vertical concatenation: out = [A / B] (a and b must have identical column
 * count) */
CMLStatus cml_matrix_concat_v(const CMLMatrix *a, const CMLMatrix *b,
                              CMLMatrix *out);
CMLMatrix *cml_matrix_concat_v_new(const CMLMatrix *a, const CMLMatrix *b);

/* Add row vector `vec` (length mat->cols) to every row of `mat` */
CMLStatus cml_matrix_add_row_vec(const CMLMatrix *mat, const CMLVector *vec,
                                 CMLMatrix *out);

/* Add column vector `vec` (length mat->rows) to every column of `mat` */
CMLStatus cml_matrix_add_col_vec(const CMLMatrix *mat, const CMLVector *vec,
                                 CMLMatrix *out);

/* Pretty-print matrix to stdout */
void cml_matrix_print(const CMLMatrix *mat, const char *name);

#endif /* CML_MATRIX_H */
