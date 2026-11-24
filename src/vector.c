
/* ==============================================================================
 * src/vector.c — CML Vector Engine Implementation
 * ------------------------------------------------------------------------------
 */

#include "cml/vector.h"
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

CMLVector *cml_vector_create(size_t size) {
  if (size > SIZE_MAX / sizeof(double)) {
    return NULL;
  }

  CMLVector *vec = (CMLVector *)cml_malloc(sizeof(CMLVector));
  if (vec == NULL) {
    return NULL;
  }

  if (size == 0) {
    vec->data = NULL;
    vec->size = 0;
    vec->capacity = 0;
    return vec;
  }

  vec->data = (double *)cml_calloc(size, sizeof(double));
  if (vec->data == NULL) {
    cml_free(vec);
    return NULL;
  }

  vec->size = size;
  vec->capacity = size;
  return vec;
}

CMLVector *cml_vector_create_filled(size_t size, double value) {
  CMLVector *vec = cml_vector_create(size);
  if (vec == NULL) {
    return NULL;
  }

  if (value != 0.0) {
    for (size_t i = 0; i < size; ++i) {
      vec->data[i] = value;
    }
  }

  return vec;
}

CMLVector *cml_vector_from_array(const double *array, size_t size) {
  if (size > 0 && array == NULL) {
    return NULL;
  }

  CMLVector *vec = cml_vector_create(size);
  if (vec == NULL) {
    return NULL;
  }

  if (size > 0) {
    cml_mem_copy(vec->data, array, size * sizeof(double));
  }

  return vec;
}

CMLVector *cml_vector_clone(const CMLVector *src) {
  if (src == NULL) {
    return NULL;
  }

  CMLVector *clone = cml_vector_create(src->size);
  if (clone == NULL) {
    return NULL;
  }

  if (src->size > 0 && src->data != NULL) {
    cml_mem_copy(clone->data, src->data, src->size * sizeof(double));
  }

  return clone;
}

CMLStatus cml_vector_resize(CMLVector *vec, size_t new_size) {
  if (vec == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  if (new_size == vec->size) {
    return CML_SUCCESS;
  }

  if (new_size == 0) {
    if (vec->data != NULL) {
      cml_free(vec->data);
      vec->data = NULL;
    }
    vec->size = 0;
    vec->capacity = 0;
    return CML_SUCCESS;
  }

  if (new_size <= vec->capacity) {
    if (new_size > vec->size) {
      memset(vec->data + vec->size, 0, (new_size - vec->size) * sizeof(double));
    }
    vec->size = new_size;
    return CML_SUCCESS;
  }

  if (new_size > SIZE_MAX / sizeof(double)) {
    return CML_ERROR_OVERFLOW;
  }

  double *new_data =
      (double *)cml_realloc(vec->data, new_size * sizeof(double));
  if (new_data == NULL) {
    return CML_ERROR_OUT_OF_MEMORY;
  }

  memset(new_data + vec->size, 0, (new_size - vec->size) * sizeof(double));
  vec->data = new_data;
  vec->size = new_size;
  vec->capacity = new_size;
  return CML_SUCCESS;
}

CMLStatus cml_vector_clear(CMLVector *vec) {
  if (vec == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  vec->size = 0;
  return CML_SUCCESS;
}

void cml_vector_destroy(CMLVector *vec) {
  if (vec == NULL) {
    return;
  }
  if (vec->data != NULL) {
    cml_free(vec->data);
    vec->data = NULL;
  }
  cml_free(vec);
}

/* ==============================================================================
 * Element Access & Mutation
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_vector_get(const CMLVector *vec, size_t index, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (index >= vec->size) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }
  *out = vec->data[index];
  return CML_SUCCESS;
}

CMLStatus cml_vector_set(CMLVector *vec, size_t index, double value) {
  if (vec == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (index >= vec->size) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }
  vec->data[index] = value;
  return CML_SUCCESS;
}

CMLStatus cml_vector_fill(CMLVector *vec, double value) {
  if (vec == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < vec->size; ++i) {
    vec->data[i] = value;
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_swap(CMLVector *vec, size_t i, size_t j) {
  if (vec == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (i >= vec->size || j >= vec->size) {
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  }
  if (i == j) {
    return CML_SUCCESS;
  }
  double tmp = vec->data[i];
  vec->data[i] = vec->data[j];
  vec->data[j] = tmp;
  return CML_SUCCESS;
}

/* ==============================================================================
 * Arithmetic Operations
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_vector_add(const CMLVector *a, const CMLVector *b,
                         CMLVector *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->size != b->size || out->size != a->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] + b->data[i];
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_sub(const CMLVector *a, const CMLVector *b,
                         CMLVector *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->size != b->size || out->size != a->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] - b->data[i];
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_mul(const CMLVector *a, const CMLVector *b,
                         CMLVector *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->size != b->size || out->size != a->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] * b->data[i];
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_div(const CMLVector *a, const CMLVector *b,
                         CMLVector *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->size != b->size || out->size != a->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  for (size_t i = 0; i < b->size; ++i) {
    if (b->data[i] == 0.0 || isnan(b->data[i])) {
      return CML_ERROR_NUMERICAL;
    }
  }
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] / b->data[i];
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_scale(const CMLVector *a, double scalar, CMLVector *out) {
  if (a == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != a->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] * scalar;
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_add_scalar(const CMLVector *a, double scalar,
                                CMLVector *out) {
  if (a == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != a->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = a->data[i] + scalar;
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_negate(const CMLVector *a, CMLVector *out) {
  if (a == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != a->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  for (size_t i = 0; i < a->size; ++i) {
    out->data[i] = -a->data[i];
  }
  return CML_SUCCESS;
}

/* ==============================================================================
 * In-Place Arithmetic
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_vector_add_inplace(CMLVector *a, const CMLVector *b) {
  return cml_vector_add(a, b, a);
}

CMLStatus cml_vector_sub_inplace(CMLVector *a, const CMLVector *b) {
  return cml_vector_sub(a, b, a);
}

CMLStatus cml_vector_mul_inplace(CMLVector *a, const CMLVector *b) {
  return cml_vector_mul(a, b, a);
}

CMLStatus cml_vector_div_inplace(CMLVector *a, const CMLVector *b) {
  return cml_vector_div(a, b, a);
}

CMLStatus cml_vector_scale_inplace(CMLVector *a, double scalar) {
  return cml_vector_scale(a, scalar, a);
}

CMLStatus cml_vector_add_scalar_inplace(CMLVector *a, double scalar) {
  return cml_vector_add_scalar(a, scalar, a);
}

CMLStatus cml_vector_negate_inplace(CMLVector *a) {
  return cml_vector_negate(a, a);
}

/* ==============================================================================
 * Allocating Arithmetic Helpers
 * ------------------------------------------------------------------------------
 */

CMLVector *cml_vector_add_new(const CMLVector *a, const CMLVector *b) {
  if (a == NULL || b == NULL || a->size != b->size)
    return NULL;
  CMLVector *out = cml_vector_create(a->size);
  if (out == NULL)
    return NULL;
  if (cml_vector_add(a, b, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }
  return out;
}

CMLVector *cml_vector_sub_new(const CMLVector *a, const CMLVector *b) {
  if (a == NULL || b == NULL || a->size != b->size)
    return NULL;
  CMLVector *out = cml_vector_create(a->size);
  if (out == NULL)
    return NULL;
  if (cml_vector_sub(a, b, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }
  return out;
}

CMLVector *cml_vector_mul_new(const CMLVector *a, const CMLVector *b) {
  if (a == NULL || b == NULL || a->size != b->size)
    return NULL;
  CMLVector *out = cml_vector_create(a->size);
  if (out == NULL)
    return NULL;
  if (cml_vector_mul(a, b, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }
  return out;
}

CMLVector *cml_vector_div_new(const CMLVector *a, const CMLVector *b) {
  if (a == NULL || b == NULL || a->size != b->size)
    return NULL;
  CMLVector *out = cml_vector_create(a->size);
  if (out == NULL)
    return NULL;
  if (cml_vector_div(a, b, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }
  return out;
}

CMLVector *cml_vector_scale_new(const CMLVector *a, double scalar) {
  if (a == NULL)
    return NULL;
  CMLVector *out = cml_vector_create(a->size);
  if (out == NULL)
    return NULL;
  if (cml_vector_scale(a, scalar, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }
  return out;
}

CMLVector *cml_vector_add_scalar_new(const CMLVector *a, double scalar) {
  if (a == NULL)
    return NULL;
  CMLVector *out = cml_vector_create(a->size);
  if (out == NULL)
    return NULL;
  if (cml_vector_add_scalar(a, scalar, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }
  return out;
}

CMLVector *cml_vector_negate_new(const CMLVector *a) {
  if (a == NULL)
    return NULL;
  CMLVector *out = cml_vector_create(a->size);
  if (out == NULL)
    return NULL;
  if (cml_vector_negate(a, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }
  return out;
}

/* ==============================================================================
 * Advanced Vector Operations
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_vector_dot(const CMLVector *a, const CMLVector *b, double *out) {
  if (a == NULL || b == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (a->size != b->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (a->size == 0) {
    *out = 0.0;
    return CML_SUCCESS;
  }

  double sum = 0.0;
  for (size_t i = 0; i < a->size; ++i) {
    sum += a->data[i] * b->data[i];
  }

  *out = sum;
  if (isnan(sum) || isinf(sum)) {
    return CML_ERROR_NUMERICAL;
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_norm_l1(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0) {
    *out = 0.0;
    return CML_SUCCESS;
  }

  double sum = 0.0;
  for (size_t i = 0; i < vec->size; ++i) {
    sum += fabs(vec->data[i]);
  }
  *out = sum;
  if (isnan(sum) || isinf(sum)) {
    return CML_ERROR_NUMERICAL;
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_norm_l2(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0) {
    *out = 0.0;
    return CML_SUCCESS;
  }

  double scale = 0.0;
  for (size_t i = 0; i < vec->size; ++i) {
    double magnitude = fabs(vec->data[i]);
    if (magnitude > scale || isnan(magnitude)) {
      scale = magnitude;
    }
  }
  if (scale == 0.0 || !isfinite(scale)) {
    *out = scale;
    return isfinite(scale) ? CML_SUCCESS : CML_ERROR_NUMERICAL;
  }

  /* Normalize the inputs before squaring to preserve the range of the norm. */
  double sum_sq = 0.0;
  for (size_t i = 0; i < vec->size; ++i) {
    double scaled = vec->data[i] / scale;
    sum_sq += scaled * scaled;
  }
  double res = scale * sqrt(sum_sq);
  *out = res;
  if (isnan(res) || isinf(res)) {
    return CML_ERROR_NUMERICAL;
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_norm(const CMLVector *vec, double *out) {
  return cml_vector_norm_l2(vec, out);
}

CMLStatus cml_vector_norm_inf(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0) {
    *out = 0.0;
    return CML_SUCCESS;
  }

  double max_val = 0.0;
  for (size_t i = 0; i < vec->size; ++i) {
    double abs_val = fabs(vec->data[i]);
    if (abs_val > max_val) {
      max_val = abs_val;
    }
  }
  *out = max_val;
  if (isnan(max_val) || isinf(max_val)) {
    return CML_ERROR_NUMERICAL;
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_normalize(const CMLVector *vec, CMLVector *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size != out->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (vec->size == 0) {
    return CML_SUCCESS;
  }

  double magnitude = 0.0;
  CMLStatus st = cml_vector_norm_l2(vec, &magnitude);
  if (st != CML_SUCCESS || magnitude == 0.0 || isnan(magnitude) ||
      isinf(magnitude)) {
    return CML_ERROR_NUMERICAL;
  }

  for (size_t i = 0; i < vec->size; ++i) {
    out->data[i] = vec->data[i] / magnitude;
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_normalize_inplace(CMLVector *vec) {
  return cml_vector_normalize(vec, vec);
}

CMLVector *cml_vector_normalize_new(const CMLVector *vec) {
  if (vec == NULL)
    return NULL;
  CMLVector *out = cml_vector_create(vec->size);
  if (out == NULL)
    return NULL;
  if (cml_vector_normalize(vec, out) != CML_SUCCESS) {
    cml_vector_destroy(out);
    return NULL;
  }
  return out;
}

/* ==============================================================================
 * Statistics
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_vector_sum(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0) {
    *out = 0.0;
    return CML_SUCCESS;
  }

  double sum = 0.0;
  for (size_t i = 0; i < vec->size; ++i) {
    sum += vec->data[i];
  }
  *out = sum;
  return CML_SUCCESS;
}

CMLStatus cml_vector_mean(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double sum = 0.0;
  cml_vector_sum(vec, &sum);
  *out = sum / (double)vec->size;
  return CML_SUCCESS;
}

CMLStatus cml_vector_min_max(const CMLVector *vec, double *out_min,
                             double *out_max) {
  if (vec == NULL || out_min == NULL || out_max == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double min_val = vec->data[0];
  double max_val = vec->data[0];
  for (size_t i = 1; i < vec->size; ++i) {
    if (vec->data[i] < min_val)
      min_val = vec->data[i];
    if (vec->data[i] > max_val)
      max_val = vec->data[i];
  }

  *out_min = min_val;
  *out_max = max_val;
  return CML_SUCCESS;
}

CMLStatus cml_vector_min(const CMLVector *vec, double *out) {
  double max_dummy = 0.0;
  return cml_vector_min_max(vec, out, &max_dummy);
}

CMLStatus cml_vector_max(const CMLVector *vec, double *out) {
  double min_dummy = 0.0;
  return cml_vector_min_max(vec, &min_dummy, out);
}

CMLStatus cml_vector_variance(const CMLVector *vec, int is_sample,
                              double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (is_sample != 0 && vec->size < 2) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double mean = 0.0;
  CMLStatus st = cml_vector_mean(vec, &mean);
  if (st != CML_SUCCESS) {
    return st;
  }

  double sum_sq_diff = 0.0;
  for (size_t i = 0; i < vec->size; ++i) {
    double diff = vec->data[i] - mean;
    sum_sq_diff += diff * diff;
  }

  double divisor =
      (is_sample != 0) ? (double)(vec->size - 1) : (double)vec->size;
  *out = sum_sq_diff / divisor;
  return CML_SUCCESS;
}

CMLStatus cml_vector_variance_pop(const CMLVector *vec, double *out) {
  return cml_vector_variance(vec, 0, out);
}

CMLStatus cml_vector_variance_sample(const CMLVector *vec, double *out) {
  return cml_vector_variance(vec, 1, out);
}

CMLStatus cml_vector_stddev(const CMLVector *vec, int is_sample, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  double var = 0.0;
  CMLStatus st = cml_vector_variance(vec, is_sample, &var);
  if (st != CML_SUCCESS) {
    return st;
  }
  *out = sqrt(var);
  return CML_SUCCESS;
}

CMLStatus cml_vector_stddev_pop(const CMLVector *vec, double *out) {
  return cml_vector_stddev(vec, 0, out);
}

CMLStatus cml_vector_stddev_sample(const CMLVector *vec, double *out) {
  return cml_vector_stddev(vec, 1, out);
}

/* ==============================================================================
 * Utilities
 * ------------------------------------------------------------------------------
 */

bool cml_vector_is_equal(const CMLVector *a, const CMLVector *b, double tol) {
  if (a == NULL || b == NULL) {
    return false;
  }
  if (a->size != b->size) {
    return false;
  }
  for (size_t i = 0; i < a->size; ++i) {
    if (!cml_is_close(a->data[i], b->data[i], tol)) {
      return false;
    }
  }
  return true;
}

void cml_vector_print(const CMLVector *vec, const char *name) {
  if (name != NULL) {
    printf("%s = ", name);
  }
  if (vec == NULL) {
    printf("NULL\n");
    return;
  }
  printf("[");
  for (size_t i = 0; i < vec->size; ++i) {
    printf("%.4f%s", vec->data[i], (i + 1 < vec->size) ? ", " : "");
  }
  printf("]\n");
}
