#ifndef CML_INTERNAL_H
#define CML_INTERNAL_H

#include "cml/matrix.h"

#include <math.h>
#include <stdint.h>

static inline CMLStatus cml_vector_storage(const CMLVector *v, size_t size) {
  if (v == NULL || (size != 0 && v->data == NULL))
    return CML_ERROR_INVALID_ARGUMENT;
  if (v->size != size || v->capacity < size)
    return CML_ERROR_INVALID_DIMENSION;
  return size > SIZE_MAX / sizeof(double) ? CML_ERROR_OVERFLOW : CML_SUCCESS;
}

static inline CMLStatus cml_matrix_storage(const CMLMatrix *m) {
  if (m == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (m->rows == 0 || m->cols == 0)
    return CML_ERROR_INVALID_DIMENSION;
  if (m->rows > SIZE_MAX / sizeof(double) / m->cols)
    return CML_ERROR_OVERFLOW;
  if (m->capacity < m->rows * m->cols)
    return CML_ERROR_INVALID_DIMENSION;
  return m->data == NULL ? CML_ERROR_INVALID_ARGUMENT : CML_SUCCESS;
}

static inline bool cml_values_finite(const double *values, size_t count) {
  for (size_t i = 0; i < count; ++i)
    if (!isfinite(values[i]))
      return false;
  return true;
}

static inline double cml_affine(const double *x, const double *weights,
                                size_t count, double bias) {
  for (size_t i = 0; i < count; ++i)
    bias = fma(x[i], weights[i], bias);
  return bias;
}

static inline double cml_bce_logit(double label, double logit) {
  double signed_error = label == 1.0 ? -logit : logit;
  return fmax(signed_error, 0.0) + log1p(exp(-fabs(signed_error)));
}

/* SplitMix64 accepts every seed, including zero. */
static inline uint64_t cml_random_next(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

static inline size_t cml_random_index(uint64_t *state, size_t bound) {
  uint64_t range = (uint64_t)bound;
  uint64_t minimum = (UINT64_C(0) - range) % range;
  uint64_t value;
  do {
    value = cml_random_next(state);
  } while (value < minimum);
  return (size_t)(value % range);
}

static inline double cml_random_unit(uint64_t *state) {
  return (double)(cml_random_next(state) >> 11) * 0x1p-53;
}

static inline void cml_shuffle_indices(size_t *indices, size_t count,
                                       uint64_t *seed) {
  for (size_t i = count; i > 1; --i) {
    size_t j = cml_random_index(seed, i);
    size_t saved = indices[i - 1];
    indices[i - 1] = indices[j];
    indices[j] = saved;
  }
}

#endif
