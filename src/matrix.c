
/* ==============================================================================
 * src/matrix.c — CML Matrix Engine Implementation
 * ------------------------------------------------------------------------------
 */

#include "cml/matrix.h"
#include "cml/math.h"
#include "cml/memory.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ==============================================================================
 * Lifecycle & Memory Management
 * ------------------------------------------------------------------------------
 */

CMLMatrix *cml_matrix_create(size_t rows, size_t cols) {
  if (rows > 0 && cols > SIZE_MAX / rows) {
    return NULL;
  }
  size_t total = rows * cols;
  if (total > SIZE_MAX / sizeof(double)) {
    return NULL;
  }

  CMLMatrix *mat = (CMLMatrix *)cml_malloc(sizeof(CMLMatrix));
  if (mat == NULL) {
    return NULL;
  }

  if (total == 0) {
    mat->data = NULL;
    mat->rows = 0;
    mat->cols = 0;
    mat->capacity = 0;
    return mat;
  }

  mat->data = (double *)cml_calloc(total, sizeof(double));
  if (mat->data == NULL) {
    cml_free(mat);
    return NULL;
  }

  mat->rows = rows;
  mat->cols = cols;
  mat->capacity = total;
  return mat;
}

CMLMatrix *cml_matrix_create_filled(size_t rows, size_t cols, double value) {
  CMLMatrix *mat = cml_matrix_create(rows, cols);
  if (mat == NULL) {
    return NULL;
  }

  if (value != 0.0 && mat->data != NULL) {
    size_t total = rows * cols;
    for (size_t i = 0; i < total; ++i) {
      mat->data[i] = value;
    }
  }

  return mat;
}

CMLMatrix *cml_matrix_zeros(size_t rows, size_t cols) {
  return cml_matrix_create(rows, cols);
}

CMLMatrix *cml_matrix_ones(size_t rows, size_t cols) {
  return cml_matrix_create_filled(rows, cols, 1.0);
}

CMLMatrix *cml_matrix_identity(size_t n) {
  CMLMatrix *mat = cml_matrix_create(n, n);
  if (mat == NULL) {
    return NULL;
  }

  if (n > 0 && mat->data != NULL) {
    for (size_t i = 0; i < n; ++i) {
      mat->data[i * n + i] = 1.0;
    }
  }

  return mat;
}

CMLMatrix *cml_matrix_from_array(const double *array, size_t rows,
                                 size_t cols) {
  if (rows > 0 && cols > SIZE_MAX / rows) {
    return NULL;
  }
  size_t total = rows * cols;
  if (total > 0 && array == NULL) {
    return NULL;
  }

  CMLMatrix *mat = cml_matrix_create(rows, cols);
  if (mat == NULL) {
    return NULL;
  }

  if (total > 0 && mat->data != NULL) {
    cml_mem_copy(mat->data, array, total * sizeof(double));
  }

  return mat;
}

CMLMatrix *cml_matrix_clone(const CMLMatrix *src) {
  if (src == NULL) {
    return NULL;
  }

  CMLMatrix *clone = cml_matrix_create(src->rows, src->cols);
  if (clone == NULL) {
    return NULL;
  }

  size_t total = src->rows * src->cols;
  if (total > 0 && src->data != NULL && clone->data != NULL) {
    cml_mem_copy(clone->data, src->data, total * sizeof(double));
  }

  return clone;
}

CMLStatus cml_matrix_resize(CMLMatrix *mat, size_t new_rows, size_t new_cols) {
  if (mat == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  if (new_rows == 0 || new_cols == 0) {
    if (mat->data != NULL) {
      cml_free(mat->data);
      mat->data = NULL;
    }
    mat->rows = 0;
    mat->cols = 0;
    mat->capacity = 0;
    return CML_SUCCESS;
  }

  if (new_rows > SIZE_MAX / new_cols) {
    return CML_ERROR_OVERFLOW;
  }
  size_t new_total = new_rows * new_cols;
  if (new_total > SIZE_MAX / sizeof(double)) {
    return CML_ERROR_OVERFLOW;
  }

  double *new_data = (double *)cml_calloc(new_total, sizeof(double));
  if (new_data == NULL) {
    return CML_ERROR_OUT_OF_MEMORY;
  }

  if (mat->data != NULL && mat->rows > 0 && mat->cols > 0) {
    size_t min_r = (mat->rows < new_rows) ? mat->rows : new_rows;
    size_t min_c = (mat->cols < new_cols) ? mat->cols : new_cols;
    for (size_t r = 0; r < min_r; ++r) {
      cml_mem_copy(&new_data[r * new_cols], &mat->data[r * mat->cols],
                   min_c * sizeof(double));
    }
    cml_free(mat->data);
  }

  mat->data = new_data;
  mat->rows = new_rows;
  mat->cols = new_cols;
  mat->capacity = new_total;
  return CML_SUCCESS;
}

CMLStatus cml_matrix_clear(CMLMatrix *mat) {
  if (mat == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  mat->rows = 0;
  mat->cols = 0;
  return CML_SUCCESS;
}

void cml_matrix_destroy(CMLMatrix *mat) {
  if (mat == NULL) {
    return;
  }

  if (mat->data != NULL) {
    cml_free(mat->data);
    mat->data = NULL;
  }
  cml_free(mat);
}

/* ==============================================================================
 * Element Access & Mutation
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_matrix_get(const CMLMatrix *mat, size_t row, size_t col,
                         double *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (row >= mat->rows || col >= mat->cols || mat->data == NULL) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }

  *out = mat->data[row * mat->cols + col];
  return CML_SUCCESS;
}

CMLStatus cml_matrix_set(CMLMatrix *mat, size_t row, size_t col, double value) {
  if (mat == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (row >= mat->rows || col >= mat->cols || mat->data == NULL) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }

  mat->data[row * mat->cols + col] = value;
  return CML_SUCCESS;
}

CMLStatus cml_matrix_fill(CMLMatrix *mat, double value) {
  if (mat == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  size_t total = mat->rows * mat->cols;
  if (total > 0 && mat->data != NULL) {
    for (size_t i = 0; i < total; ++i) {
      mat->data[i] = value;
    }
  }

  return CML_SUCCESS;
}

CMLStatus cml_matrix_swap_elements(CMLMatrix *mat, size_t r1, size_t c1,
                                   size_t r2, size_t c2) {
  if (mat == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (r1 >= mat->rows || c1 >= mat->cols || r2 >= mat->rows ||
      c2 >= mat->cols || mat->data == NULL) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }

  if (r1 == r2 && c1 == c2) {
    return CML_SUCCESS;
  }

  size_t idx1 = r1 * mat->cols + c1;
  size_t idx2 = r2 * mat->cols + c2;
  double tmp = mat->data[idx1];
  mat->data[idx1] = mat->data[idx2];
  mat->data[idx2] = tmp;
  return CML_SUCCESS;
}

size_t cml_matrix_rows(const CMLMatrix *mat) { return mat ? mat->rows : 0; }

size_t cml_matrix_cols(const CMLMatrix *mat) { return mat ? mat->cols : 0; }

double *cml_matrix_data(CMLMatrix *mat) { return mat ? mat->data : NULL; }

const double *cml_matrix_data_const(const CMLMatrix *mat) {
  return mat ? mat->data : NULL;
}

/* ==============================================================================
 * Matrix Arithmetic
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_matrix_add(const CMLMatrix *a, const CMLMatrix *b,
                         CMLMatrix *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->rows != b->rows || a->cols != b->cols || out->rows != a->rows ||
      out->cols != a->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t total = a->rows * a->cols;
  for (size_t i = 0; i < total; ++i) {
    out->data[i] = a->data[i] + b->data[i];
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_sub(const CMLMatrix *a, const CMLMatrix *b,
                         CMLMatrix *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->rows != b->rows || a->cols != b->cols || out->rows != a->rows ||
      out->cols != a->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t total = a->rows * a->cols;
  for (size_t i = 0; i < total; ++i) {
    out->data[i] = a->data[i] - b->data[i];
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_mul(const CMLMatrix *a, const CMLMatrix *b,
                         CMLMatrix *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->rows != b->rows || a->cols != b->cols || out->rows != a->rows ||
      out->cols != a->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t total = a->rows * a->cols;
  for (size_t i = 0; i < total; ++i) {
    out->data[i] = a->data[i] * b->data[i];
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_div(const CMLMatrix *a, const CMLMatrix *b,
                         CMLMatrix *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->rows != b->rows || a->cols != b->cols || out->rows != a->rows ||
      out->cols != a->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t total = a->rows * a->cols;
  for (size_t i = 0; i < total; ++i) {
    if (b->data[i] == 0.0 || isnan(b->data[i])) {
      return CML_ERROR_NUMERICAL;
    }
  }

  for (size_t i = 0; i < total; ++i) {
    out->data[i] = a->data[i] / b->data[i];
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_scale(const CMLMatrix *a, double scalar, CMLMatrix *out) {
  if (a == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->rows != a->rows || out->cols != a->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t total = a->rows * a->cols;
  for (size_t i = 0; i < total; ++i) {
    out->data[i] = a->data[i] * scalar;
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_add_scalar(const CMLMatrix *a, double scalar,
                                CMLMatrix *out) {
  if (a == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->rows != a->rows || out->cols != a->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t total = a->rows * a->cols;
  for (size_t i = 0; i < total; ++i) {
    out->data[i] = a->data[i] + scalar;
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_negate(const CMLMatrix *a, CMLMatrix *out) {
  if (a == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->rows != a->rows || out->cols != a->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t total = a->rows * a->cols;
  for (size_t i = 0; i < total; ++i) {
    out->data[i] = -a->data[i];
  }
  return CML_SUCCESS;
}

/* In-place */
CMLStatus cml_matrix_add_inplace(CMLMatrix *a, const CMLMatrix *b) {
  return cml_matrix_add(a, b, a);
}

CMLStatus cml_matrix_sub_inplace(CMLMatrix *a, const CMLMatrix *b) {
  return cml_matrix_sub(a, b, a);
}

CMLStatus cml_matrix_mul_inplace(CMLMatrix *a, const CMLMatrix *b) {
  return cml_matrix_mul(a, b, a);
}

CMLStatus cml_matrix_div_inplace(CMLMatrix *a, const CMLMatrix *b) {
  return cml_matrix_div(a, b, a);
}

CMLStatus cml_matrix_scale_inplace(CMLMatrix *a, double scalar) {
  return cml_matrix_scale(a, scalar, a);
}

CMLStatus cml_matrix_add_scalar_inplace(CMLMatrix *a, double scalar) {
  return cml_matrix_add_scalar(a, scalar, a);
}

CMLStatus cml_matrix_negate_inplace(CMLMatrix *a) {
  return cml_matrix_negate(a, a);
}

/* Allocating helpers */
CMLMatrix *cml_matrix_add_new(const CMLMatrix *a, const CMLMatrix *b) {
  if (a == NULL || b == NULL || a->rows != b->rows || a->cols != b->cols) {
    return NULL;
  }
  CMLMatrix *out = cml_matrix_create(a->rows, a->cols);
  if (out == NULL) {
    return NULL;
  }
  if (cml_matrix_add(a, b, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }
  return out;
}

CMLMatrix *cml_matrix_sub_new(const CMLMatrix *a, const CMLMatrix *b) {
  if (a == NULL || b == NULL || a->rows != b->rows || a->cols != b->cols) {
    return NULL;
  }
  CMLMatrix *out = cml_matrix_create(a->rows, a->cols);
  if (out == NULL) {
    return NULL;
  }
  if (cml_matrix_sub(a, b, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }
  return out;
}

CMLMatrix *cml_matrix_mul_new(const CMLMatrix *a, const CMLMatrix *b) {
  if (a == NULL || b == NULL || a->rows != b->rows || a->cols != b->cols) {
    return NULL;
  }
  CMLMatrix *out = cml_matrix_create(a->rows, a->cols);
  if (out == NULL) {
    return NULL;
  }
  if (cml_matrix_mul(a, b, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }
  return out;
}

CMLMatrix *cml_matrix_div_new(const CMLMatrix *a, const CMLMatrix *b) {
  if (a == NULL || b == NULL || a->rows != b->rows || a->cols != b->cols) {
    return NULL;
  }
  CMLMatrix *out = cml_matrix_create(a->rows, a->cols);
  if (out == NULL) {
    return NULL;
  }
  if (cml_matrix_div(a, b, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }
  return out;
}

CMLMatrix *cml_matrix_scale_new(const CMLMatrix *a, double scalar) {
  if (a == NULL) {
    return NULL;
  }
  CMLMatrix *out = cml_matrix_create(a->rows, a->cols);
  if (out == NULL) {
    return NULL;
  }
  if (cml_matrix_scale(a, scalar, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }
  return out;
}

CMLMatrix *cml_matrix_add_scalar_new(const CMLMatrix *a, double scalar) {
  if (a == NULL) {
    return NULL;
  }
  CMLMatrix *out = cml_matrix_create(a->rows, a->cols);
  if (out == NULL) {
    return NULL;
  }
  if (cml_matrix_add_scalar(a, scalar, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }
  return out;
}

CMLMatrix *cml_matrix_negate_new(const CMLMatrix *a) {
  if (a == NULL) {
    return NULL;
  }
  CMLMatrix *out = cml_matrix_create(a->rows, a->cols);
  if (out == NULL) {
    return NULL;
  }
  if (cml_matrix_negate(a, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }
  return out;
}

/* ==============================================================================
 * Linear Algebra & Matrix Multiplication
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_matrix_matmul(const CMLMatrix *a, const CMLMatrix *b,
                            CMLMatrix *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->cols != b->rows || out->rows != a->rows || out->cols != b->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t m = a->rows;
  size_t n = a->cols;
  size_t p = b->cols;

  if (m == 0 || p == 0) {
    return CML_SUCCESS;
  }

  if (n == 0) {
    memset(out->data, 0, m * p * sizeof(double));
    return CML_SUCCESS;
  }

  bool aliased = (out->data == a->data || out->data == b->data);
  double *dest = out->data;

  if (aliased) {
    dest = (double *)cml_calloc(m * p, sizeof(double));
    if (dest == NULL) {
      return CML_ERROR_OUT_OF_MEMORY;
    }
  } else {
    memset(dest, 0, m * p * sizeof(double));
  }

  /* Cache-friendly loop order: i -> k -> j */
  for (size_t i = 0; i < m; ++i) {
    size_t a_row_offset = i * n;
    size_t c_row_offset = i * p;
    for (size_t k = 0; k < n; ++k) {
      double aik = a->data[a_row_offset + k];
      if (aik == 0.0) {
        continue;
      }
      size_t b_row_offset = k * p;
      for (size_t j = 0; j < p; ++j) {
        dest[c_row_offset + j] += aik * b->data[b_row_offset + j];
      }
    }
  }

  if (aliased) {
    cml_mem_copy(out->data, dest, m * p * sizeof(double));
    cml_free(dest);
  }

  return CML_SUCCESS;
}

CMLMatrix *cml_matrix_matmul_new(const CMLMatrix *a, const CMLMatrix *b) {
  if (a == NULL || b == NULL || a->cols != b->rows) {
    return NULL;
  }

  CMLMatrix *out = cml_matrix_create(a->rows, b->cols);
  if (out == NULL) {
    return NULL;
  }

  if (cml_matrix_matmul(a, b, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }

  return out;
}

CMLStatus cml_matrix_transpose(const CMLMatrix *src, CMLMatrix *out) {
  if (src == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->rows != src->cols || out->cols != src->rows) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  if (src->rows == 0 || src->cols == 0) {
    return CML_SUCCESS;
  }

  if (out->data == src->data) {
    if (src->rows == src->cols) {
      size_t n = src->rows;
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
          size_t idx1 = i * n + j;
          size_t idx2 = j * n + i;
          double tmp = out->data[idx1];
          out->data[idx1] = out->data[idx2];
          out->data[idx2] = tmp;
        }
      }
      return CML_SUCCESS;
    } else {
      size_t total = src->rows * src->cols;
      double *temp = (double *)cml_malloc(total * sizeof(double));
      if (temp == NULL) {
        return CML_ERROR_OUT_OF_MEMORY;
      }
      for (size_t i = 0; i < src->rows; ++i) {
        for (size_t j = 0; j < src->cols; ++j) {
          temp[j * out->cols + i] = src->data[i * src->cols + j];
        }
      }
      cml_mem_copy(out->data, temp, total * sizeof(double));
      cml_free(temp);
      return CML_SUCCESS;
    }
  }

  for (size_t i = 0; i < src->rows; ++i) {
    for (size_t j = 0; j < src->cols; ++j) {
      out->data[j * out->cols + i] = src->data[i * src->cols + j];
    }
  }

  return CML_SUCCESS;
}

CMLMatrix *cml_matrix_transpose_new(const CMLMatrix *src) {
  if (src == NULL) {
    return NULL;
  }

  CMLMatrix *out = cml_matrix_create(src->cols, src->rows);
  if (out == NULL) {
    return NULL;
  }

  if (cml_matrix_transpose(src, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }

  return out;
}

CMLStatus cml_matrix_transpose_inplace(CMLMatrix *mat) {
  if (mat == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (mat->rows == 0 || mat->cols == 0) {
    return CML_SUCCESS;
  }

  if (mat->rows == mat->cols) {
    size_t n = mat->rows;
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        size_t idx1 = i * n + j;
        size_t idx2 = j * n + i;
        double tmp = mat->data[idx1];
        mat->data[idx1] = mat->data[idx2];
        mat->data[idx2] = tmp;
      }
    }
    return CML_SUCCESS;
  }

  size_t total = mat->rows * mat->cols;
  double *temp = (double *)cml_malloc(total * sizeof(double));
  if (temp == NULL) {
    return CML_ERROR_OUT_OF_MEMORY;
  }

  for (size_t i = 0; i < mat->rows; ++i) {
    for (size_t j = 0; j < mat->cols; ++j) {
      temp[j * mat->rows + i] = mat->data[i * mat->cols + j];
    }
  }

  cml_mem_copy(mat->data, temp, total * sizeof(double));
  cml_free(temp);

  size_t tmp_dim = mat->rows;
  mat->rows = mat->cols;
  mat->cols = tmp_dim;
  return CML_SUCCESS;
}

/* ==============================================================================
 * Row and Column Operations
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_matrix_get_row(const CMLMatrix *mat, size_t row, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (row >= mat->rows) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }
  if (out->size != mat->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  if (mat->cols > 0 && mat->data != NULL) {
    cml_mem_copy(out->data, &mat->data[row * mat->cols],
                 mat->cols * sizeof(double));
  }
  return CML_SUCCESS;
}

CMLVector *cml_matrix_get_row_new(const CMLMatrix *mat, size_t row) {
  if (mat == NULL || row >= mat->rows) {
    return NULL;
  }

  CMLVector *vec = cml_vector_create(mat->cols);
  if (vec == NULL) {
    return NULL;
  }

  if (cml_matrix_get_row(mat, row, vec) != CML_SUCCESS) {
    cml_vector_destroy(vec);
    return NULL;
  }
  return vec;
}

CMLStatus cml_matrix_get_col(const CMLMatrix *mat, size_t col, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (col >= mat->cols) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }
  if (out->size != mat->rows) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t i = 0; i < mat->rows; ++i) {
    out->data[i] = mat->data[i * mat->cols + col];
  }
  return CML_SUCCESS;
}

CMLVector *cml_matrix_get_col_new(const CMLMatrix *mat, size_t col) {
  if (mat == NULL || col >= mat->cols) {
    return NULL;
  }

  CMLVector *vec = cml_vector_create(mat->rows);
  if (vec == NULL) {
    return NULL;
  }

  if (cml_matrix_get_col(mat, col, vec) != CML_SUCCESS) {
    cml_vector_destroy(vec);
    return NULL;
  }
  return vec;
}

CMLStatus cml_matrix_set_row(CMLMatrix *mat, size_t row, const CMLVector *vec) {
  if (mat == NULL || vec == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (row >= mat->rows) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }
  if (vec->size != mat->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  if (mat->cols > 0 && mat->data != NULL) {
    cml_mem_copy(&mat->data[row * mat->cols], vec->data,
                 mat->cols * sizeof(double));
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_set_col(CMLMatrix *mat, size_t col, const CMLVector *vec) {
  if (mat == NULL || vec == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (col >= mat->cols) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }
  if (vec->size != mat->rows) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t i = 0; i < mat->rows; ++i) {
    mat->data[i * mat->cols + col] = vec->data[i];
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_swap_rows(CMLMatrix *mat, size_t r1, size_t r2) {
  if (mat == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (r1 >= mat->rows || r2 >= mat->rows) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }

  if (r1 == r2 || mat->cols == 0) {
    return CML_SUCCESS;
  }

  for (size_t c = 0; c < mat->cols; ++c) {
    size_t idx1 = r1 * mat->cols + c;
    size_t idx2 = r2 * mat->cols + c;
    double tmp = mat->data[idx1];
    mat->data[idx1] = mat->data[idx2];
    mat->data[idx2] = tmp;
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_swap_cols(CMLMatrix *mat, size_t c1, size_t c2) {
  if (mat == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (c1 >= mat->cols || c2 >= mat->cols) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }

  if (c1 == c2 || mat->rows == 0) {
    return CML_SUCCESS;
  }

  for (size_t r = 0; r < mat->rows; ++r) {
    size_t idx1 = r * mat->cols + c1;
    size_t idx2 = r * mat->cols + c2;
    double tmp = mat->data[idx1];
    mat->data[idx1] = mat->data[idx2];
    mat->data[idx2] = tmp;
  }
  return CML_SUCCESS;
}

/* ==============================================================================
 * Matrix-Vector & Vector-Matrix Operations
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_matrix_vec_mul(const CMLMatrix *mat, const CMLVector *vec,
                             CMLVector *out) {
  if (mat == NULL || vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size != mat->cols || out->size != mat->rows) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  if (mat->rows == 0) {
    return CML_SUCCESS;
  }

  bool aliased = (out->data == vec->data);
  double stack_buf[64];
  double *dest = out->data;

  if (aliased) {
    if (mat->rows <= 64) {
      dest = stack_buf;
    } else {
      dest = (double *)cml_malloc(mat->rows * sizeof(double));
      if (dest == NULL) {
        return CML_ERROR_OUT_OF_MEMORY;
      }
    }
  }

  for (size_t i = 0; i < mat->rows; ++i) {
    double sum = 0.0;
    size_t row_offset = i * mat->cols;
    for (size_t j = 0; j < mat->cols; ++j) {
      sum += mat->data[row_offset + j] * vec->data[j];
    }
    dest[i] = sum;
  }

  if (aliased) {
    cml_mem_copy(out->data, dest, mat->rows * sizeof(double));
    if (dest != stack_buf) {
      cml_free(dest);
    }
  }

  return CML_SUCCESS;
}

CMLVector *cml_matrix_vec_mul_new(const CMLMatrix *mat, const CMLVector *vec) {
  if (mat == NULL || vec == NULL || vec->size != mat->cols) {
    return NULL;
  }

  CMLVector *out = cml_vector_create(mat->rows);
  if (out == NULL) {
    return NULL;
  }

  if (cml_matrix_vec_mul(mat, vec, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }

  return out;
}

CMLStatus cml_vector_mat_mul(const CMLVector *vec, const CMLMatrix *mat,
                             CMLVector *out) {
  if (vec == NULL || mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size != mat->rows || out->size != mat->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  if (mat->cols == 0) {
    return CML_SUCCESS;
  }

  bool aliased = (out->data == vec->data);
  double stack_buf[64];
  double *dest = out->data;

  if (aliased) {
    if (mat->cols <= 64) {
      dest = stack_buf;
    } else {
      dest = (double *)cml_malloc(mat->cols * sizeof(double));
      if (dest == NULL) {
        return CML_ERROR_OUT_OF_MEMORY;
      }
    }
  }

  for (size_t j = 0; j < mat->cols; ++j) {
    double sum = 0.0;
    for (size_t i = 0; i < mat->rows; ++i) {
      sum += vec->data[i] * mat->data[i * mat->cols + j];
    }
    dest[j] = sum;
  }

  if (aliased) {
    cml_mem_copy(out->data, dest, mat->cols * sizeof(double));
    if (dest != stack_buf) {
      cml_free(dest);
    }
  }

  return CML_SUCCESS;
}

CMLVector *cml_vector_mat_mul_new(const CMLVector *vec, const CMLMatrix *mat) {
  if (vec == NULL || mat == NULL || vec->size != mat->rows) {
    return NULL;
  }

  CMLVector *out = cml_vector_create(mat->cols);
  if (out == NULL) {
    return NULL;
  }

  if (cml_vector_mat_mul(vec, mat, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }

  return out;
}

/* ==============================================================================
 * Diagonal Operations
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_matrix_get_diag(const CMLMatrix *mat, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t d = (mat->rows < mat->cols) ? mat->rows : mat->cols;
  if (out->size != d) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t i = 0; i < d; ++i) {
    out->data[i] = mat->data[i * mat->cols + i];
  }
  return CML_SUCCESS;
}

CMLVector *cml_matrix_get_diag_new(const CMLMatrix *mat) {
  if (mat == NULL) {
    return NULL;
  }

  size_t d = (mat->rows < mat->cols) ? mat->rows : mat->cols;
  CMLVector *out = cml_vector_create(d);
  if (out == NULL) {
    return NULL;
  }

  if (cml_matrix_get_diag(mat, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }

  return out;
}

CMLStatus cml_matrix_set_diag(CMLMatrix *mat, const CMLVector *vec) {
  if (mat == NULL || vec == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t d = (mat->rows < mat->cols) ? mat->rows : mat->cols;
  if (vec->size != d) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t i = 0; i < d; ++i) {
    mat->data[i * mat->cols + i] = vec->data[i];
  }
  return CML_SUCCESS;
}

CMLMatrix *cml_matrix_from_diag(const CMLVector *vec) {
  if (vec == NULL) {
    return NULL;
  }

  CMLMatrix *mat = cml_matrix_create(vec->size, vec->size);
  if (mat == NULL) {
    return NULL;
  }

  for (size_t i = 0; i < vec->size; ++i) {
    mat->data[i * vec->size + i] = vec->data[i];
  }
  return mat;
}

CMLStatus cml_matrix_trace(const CMLMatrix *mat, double *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  size_t d = (mat->rows < mat->cols) ? mat->rows : mat->cols;
  double sum = 0.0;
  for (size_t i = 0; i < d; ++i) {
    sum += mat->data[i * mat->cols + i];
  }

  *out = sum;
  return CML_SUCCESS;
}

/* ==============================================================================
 * Matrix Statistics
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_matrix_sum(const CMLMatrix *mat, double *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  size_t total = mat->rows * mat->cols;
  double sum = 0.0;
  for (size_t i = 0; i < total; ++i) {
    sum += mat->data[i];
  }

  *out = sum;
  return CML_SUCCESS;
}

CMLStatus cml_matrix_mean(const CMLMatrix *mat, double *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t total = mat->rows * mat->cols;
  if (total == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double sum = 0.0;
  for (size_t i = 0; i < total; ++i) {
    sum += mat->data[i];
  }

  *out = sum / (double)total;
  return CML_SUCCESS;
}

CMLStatus cml_matrix_min(const CMLMatrix *mat, double *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t total = mat->rows * mat->cols;
  if (total == 0 || mat->data == NULL) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double m = mat->data[0];
  for (size_t i = 1; i < total; ++i) {
    if (mat->data[i] < m) {
      m = mat->data[i];
    }
  }

  *out = m;
  return CML_SUCCESS;
}

CMLStatus cml_matrix_max(const CMLMatrix *mat, double *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t total = mat->rows * mat->cols;
  if (total == 0 || mat->data == NULL) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double m = mat->data[0];
  for (size_t i = 1; i < total; ++i) {
    if (mat->data[i] > m) {
      m = mat->data[i];
    }
  }

  *out = m;
  return CML_SUCCESS;
}

CMLStatus cml_matrix_min_max(const CMLMatrix *mat, double *out_min,
                             double *out_max) {
  if (mat == NULL || out_min == NULL || out_max == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t total = mat->rows * mat->cols;
  if (total == 0 || mat->data == NULL) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double min_val = mat->data[0];
  double max_val = mat->data[0];
  for (size_t i = 1; i < total; ++i) {
    double v = mat->data[i];
    if (v < min_val) {
      min_val = v;
    }
    if (v > max_val) {
      max_val = v;
    }
  }

  *out_min = min_val;
  *out_max = max_val;
  return CML_SUCCESS;
}

CMLStatus cml_matrix_norm_frobenius(const CMLMatrix *mat, double *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  size_t total = mat->rows * mat->cols;
  double sum_sq = 0.0;
  for (size_t i = 0; i < total; ++i) {
    sum_sq += mat->data[i] * mat->data[i];
  }

  *out = sqrt(sum_sq);
  return CML_SUCCESS;
}

CMLStatus cml_matrix_row_sums(const CMLMatrix *mat, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != mat->rows) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t i = 0; i < mat->rows; ++i) {
    double s = 0.0;
    size_t row_offset = i * mat->cols;
    for (size_t j = 0; j < mat->cols; ++j) {
      s += mat->data[row_offset + j];
    }
    out->data[i] = s;
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_col_sums(const CMLMatrix *mat, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != mat->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t j = 0; j < mat->cols; ++j) {
    double s = 0.0;
    for (size_t i = 0; i < mat->rows; ++i) {
      s += mat->data[i * mat->cols + j];
    }
    out->data[j] = s;
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_row_means(const CMLMatrix *mat, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != mat->rows || mat->cols == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double denom = (double)mat->cols;
  for (size_t i = 0; i < mat->rows; ++i) {
    double s = 0.0;
    size_t row_offset = i * mat->cols;
    for (size_t j = 0; j < mat->cols; ++j) {
      s += mat->data[row_offset + j];
    }
    out->data[i] = s / denom;
  }
  return CML_SUCCESS;
}

CMLStatus cml_matrix_col_means(const CMLMatrix *mat, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != mat->cols || mat->rows == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double denom = (double)mat->rows;
  for (size_t j = 0; j < mat->cols; ++j) {
    double s = 0.0;
    for (size_t i = 0; i < mat->rows; ++i) {
      s += mat->data[i * mat->cols + j];
    }
    out->data[j] = s / denom;
  }
  return CML_SUCCESS;
}

/* ==============================================================================
 * Matrix Properties & Predicates
 * ------------------------------------------------------------------------------
 */

bool cml_matrix_is_square(const CMLMatrix *mat) {
  return (mat != NULL && mat->rows == mat->cols && mat->rows > 0);
}

bool cml_matrix_is_symmetric(const CMLMatrix *mat, double tol) {
  if (!cml_matrix_is_square(mat)) {
    return false;
  }

  size_t n = mat->rows;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (!cml_is_close(mat->data[i * n + j], mat->data[j * n + i], tol)) {
        return false;
      }
    }
  }
  return true;
}

bool cml_matrix_is_diagonal(const CMLMatrix *mat, double tol) {
  if (!cml_matrix_is_square(mat)) {
    return false;
  }

  size_t n = mat->rows;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i != j && fabs(mat->data[i * n + j]) > tol) {
        return false;
      }
    }
  }
  return true;
}

bool cml_matrix_is_identity(const CMLMatrix *mat, double tol) {
  if (!cml_matrix_is_square(mat)) {
    return false;
  }

  size_t n = mat->rows;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double val = mat->data[i * n + j];
      if (i == j) {
        if (!cml_is_close(val, 1.0, tol)) {
          return false;
        }
      } else {
        if (fabs(val) > tol) {
          return false;
        }
      }
    }
  }
  return true;
}

bool cml_matrix_is_zero(const CMLMatrix *mat, double tol) {
  if (mat == NULL) {
    return false;
  }

  size_t total = mat->rows * mat->cols;
  for (size_t i = 0; i < total; ++i) {
    if (fabs(mat->data[i]) > tol) {
      return false;
    }
  }
  return true;
}

bool cml_matrix_is_one(const CMLMatrix *mat, double tol) {
  if (mat == NULL || mat->rows == 0 || mat->cols == 0) {
    return false;
  }

  size_t total = mat->rows * mat->cols;
  for (size_t i = 0; i < total; ++i) {
    if (!cml_is_close(mat->data[i], 1.0, tol)) {
      return false;
    }
  }
  return true;
}

bool cml_matrix_is_upper_triangular(const CMLMatrix *mat, double tol) {
  if (!cml_matrix_is_square(mat)) {
    return false;
  }

  size_t n = mat->rows;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (fabs(mat->data[i * n + j]) > tol) {
        return false;
      }
    }
  }
  return true;
}

bool cml_matrix_is_lower_triangular(const CMLMatrix *mat, double tol) {
  if (!cml_matrix_is_square(mat)) {
    return false;
  }

  size_t n = mat->rows;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (fabs(mat->data[i * n + j]) > tol) {
        return false;
      }
    }
  }
  return true;
}

bool cml_matrix_is_orthogonal(const CMLMatrix *mat, double tol) {
  if (!cml_matrix_is_square(mat)) {
    return false;
  }

  CMLMatrix *t = cml_matrix_transpose_new(mat);
  if (t == NULL) {
    return false;
  }

  CMLMatrix *prod = cml_matrix_matmul_new(t, mat);
  cml_matrix_destroy(t);
  if (prod == NULL) {
    return false;
  }

  bool res = cml_matrix_is_identity(prod, tol);
  cml_matrix_destroy(prod);
  return res;
}

bool cml_matrix_is_equal(const CMLMatrix *a, const CMLMatrix *b, double tol) {
  if (a == NULL || b == NULL) {
    return false;
  }
  if (a->rows != b->rows || a->cols != b->cols) {
    return false;
  }

  size_t total = a->rows * a->cols;
  for (size_t i = 0; i < total; ++i) {
    if (!cml_is_close(a->data[i], b->data[i], tol)) {
      return false;
    }
  }
  return true;
}

CMLStatus cml_matrix_max_abs_diff(const CMLMatrix *a, const CMLMatrix *b,
                                  double *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->rows != b->rows || a->cols != b->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t total = a->rows * a->cols;
  double max_diff = 0.0;
  for (size_t i = 0; i < total; ++i) {
    double diff = fabs(a->data[i] - b->data[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }

  *out = max_diff;
  return CML_SUCCESS;
}

/* ==============================================================================
 * Advanced Utilities: Submatrix, Concatenation, Broadcasting, and Printing
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_matrix_submatrix(const CMLMatrix *src, size_t row_start,
                               size_t row_count, size_t col_start,
                               size_t col_count, CMLMatrix *out) {
  if (src == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  if (row_start > SIZE_MAX - row_count || col_start > SIZE_MAX - col_count) {
    return CML_ERROR_OVERFLOW;
  }
  if (row_start + row_count > src->rows || col_start + col_count > src->cols) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }
  if (out->rows != row_count || out->cols != col_count) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  if (row_count == 0 || col_count == 0) {
    return CML_SUCCESS;
  }

  for (size_t r = 0; r < row_count; ++r) {
    size_t src_offset = (row_start + r) * src->cols + col_start;
    size_t out_offset = r * col_count;
    cml_mem_copy(&out->data[out_offset], &src->data[src_offset],
                 col_count * sizeof(double));
  }

  return CML_SUCCESS;
}

CMLMatrix *cml_matrix_submatrix_new(const CMLMatrix *src, size_t row_start,
                                    size_t row_count, size_t col_start,
                                    size_t col_count) {
  if (src == NULL) {
    return NULL;
  }

  CMLMatrix *out = cml_matrix_create(row_count, col_count);
  if (out == NULL) {
    return NULL;
  }

  if (cml_matrix_submatrix(src, row_start, row_count, col_start, col_count,
                           out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }

  return out;
}

CMLStatus cml_matrix_concat_h(const CMLMatrix *a, const CMLMatrix *b,
                              CMLMatrix *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->rows != b->rows) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (a->cols > SIZE_MAX - b->cols) {
    return CML_ERROR_OVERFLOW;
  }
  if (out->rows != a->rows || out->cols != a->cols + b->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  if (a->rows == 0 || out->cols == 0) {
    return CML_SUCCESS;
  }

  for (size_t r = 0; r < a->rows; ++r) {
    size_t out_row_offset = r * out->cols;
    if (a->cols > 0) {
      cml_mem_copy(&out->data[out_row_offset], &a->data[r * a->cols],
                   a->cols * sizeof(double));
    }
    if (b->cols > 0) {
      cml_mem_copy(&out->data[out_row_offset + a->cols], &b->data[r * b->cols],
                   b->cols * sizeof(double));
    }
  }

  return CML_SUCCESS;
}

CMLMatrix *cml_matrix_concat_h_new(const CMLMatrix *a, const CMLMatrix *b) {
  if (a == NULL || b == NULL || a->rows != b->rows) {
    return NULL;
  }
  if (a->cols > SIZE_MAX - b->cols) {
    return NULL;
  }

  CMLMatrix *out = cml_matrix_create(a->rows, a->cols + b->cols);
  if (out == NULL) {
    return NULL;
  }

  if (cml_matrix_concat_h(a, b, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }

  return out;
}

CMLStatus cml_matrix_concat_v(const CMLMatrix *a, const CMLMatrix *b,
                              CMLMatrix *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->cols != b->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (a->rows > SIZE_MAX - b->rows) {
    return CML_ERROR_OVERFLOW;
  }
  if (out->cols != a->cols || out->rows != a->rows + b->rows) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  if (out->rows == 0 || a->cols == 0) {
    return CML_SUCCESS;
  }

  size_t a_total = a->rows * a->cols;
  size_t b_total = b->rows * b->cols;
  if (a_total > 0) {
    cml_mem_copy(&out->data[0], a->data, a_total * sizeof(double));
  }
  if (b_total > 0) {
    cml_mem_copy(&out->data[a_total], b->data, b_total * sizeof(double));
  }

  return CML_SUCCESS;
}

CMLMatrix *cml_matrix_concat_v_new(const CMLMatrix *a, const CMLMatrix *b) {
  if (a == NULL || b == NULL || a->cols != b->cols) {
    return NULL;
  }
  if (a->rows > SIZE_MAX - b->rows) {
    return NULL;
  }

  CMLMatrix *out = cml_matrix_create(a->rows + b->rows, a->cols);
  if (out == NULL) {
    return NULL;
  }

  if (cml_matrix_concat_v(a, b, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }

  return out;
}

CMLStatus cml_matrix_add_row_vec(const CMLMatrix *mat, const CMLVector *vec,
                                 CMLMatrix *out) {
  if (mat == NULL || vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size != mat->cols || out->rows != mat->rows ||
      out->cols != mat->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t r = 0; r < mat->rows; ++r) {
    size_t offset = r * mat->cols;
    for (size_t c = 0; c < mat->cols; ++c) {
      out->data[offset + c] = mat->data[offset + c] + vec->data[c];
    }
  }

  return CML_SUCCESS;
}

CMLStatus cml_matrix_add_col_vec(const CMLMatrix *mat, const CMLVector *vec,
                                 CMLMatrix *out) {
  if (mat == NULL || vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size != mat->rows || out->rows != mat->rows ||
      out->cols != mat->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t r = 0; r < mat->rows; ++r) {
    double v = vec->data[r];
    size_t offset = r * mat->cols;
    for (size_t c = 0; c < mat->cols; ++c) {
      out->data[offset + c] = mat->data[offset + c] + v;
    }
  }

  return CML_SUCCESS;
}

void cml_matrix_print(const CMLMatrix *mat, const char *name) {
  const char *label = name ? name : "Matrix";
  if (mat == NULL) {
    printf("%s = (null)\n", label);
    return;
  }

  printf("%s (%zu x %zu):\n", label, mat->rows, mat->cols);
  if (mat->rows == 0 || mat->cols == 0 || mat->data == NULL) {
    printf("  []\n");
    return;
  }

  for (size_t i = 0; i < mat->rows; ++i) {
    printf(i == 0 ? "┌ " : (i == mat->rows - 1 ? "└ " : "│ "));
    for (size_t j = 0; j < mat->cols; ++j) {
      printf("%8.4f ", mat->data[i * mat->cols + j]);
    }
    printf(i == 0 ? " ┐\n" : (i == mat->rows - 1 ? " ┘\n" : " │\n"));
  }
}
