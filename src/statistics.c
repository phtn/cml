/* ==============================================================================
 * src/statistics.c — CML Statistical Engine Implementation
 * ------------------------------------------------------------------------------
 */

#include "cml/statistics.h"
#include "cml/math.h"
#include "cml/memory.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Helper double comparator for qsort with NaN safety */
static int compare_doubles(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  if (isnan(da) && isnan(db))
    return 0;
  if (isnan(da))
    return 1;
  if (isnan(db))
    return -1;
  if (da < db)
    return -1;
  if (da > db)
    return 1;
  return 0;
}

/* ==============================================================================
 * Descriptive Statistics on Vectors
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_stat_sum(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0 || vec->data == NULL) {
    *out = 0.0;
    return CML_SUCCESS;
  }

  double s = 0.0;
  for (size_t i = 0; i < vec->size; ++i) {
    s += vec->data[i];
  }
  *out = s;
  return CML_SUCCESS;
}

CMLStatus cml_stat_mean(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0 || vec->data == NULL) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double s = 0.0;
  for (size_t i = 0; i < vec->size; ++i) {
    s += vec->data[i];
  }
  *out = s / (double)vec->size;
  return CML_SUCCESS;
}

CMLStatus cml_stat_min(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0 || vec->data == NULL) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double m = vec->data[0];
  for (size_t i = 1; i < vec->size; ++i) {
    if (vec->data[i] < m) {
      m = vec->data[i];
    }
  }
  *out = m;
  return CML_SUCCESS;
}

CMLStatus cml_stat_max(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0 || vec->data == NULL) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double m = vec->data[0];
  for (size_t i = 1; i < vec->size; ++i) {
    if (vec->data[i] > m) {
      m = vec->data[i];
    }
  }
  *out = m;
  return CML_SUCCESS;
}

CMLStatus cml_stat_min_max(const CMLVector *vec, double *out_min,
                           double *out_max) {
  if (vec == NULL || out_min == NULL || out_max == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (vec->size == 0 || vec->data == NULL) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double min_val = vec->data[0];
  double max_val = vec->data[0];
  for (size_t i = 1; i < vec->size; ++i) {
    double v = vec->data[i];
    if (v < min_val)
      min_val = v;
    if (v > max_val)
      max_val = v;
  }
  *out_min = min_val;
  *out_max = max_val;
  return CML_SUCCESS;
}

CMLStatus cml_stat_range(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  double min_val = 0.0;
  double max_val = 0.0;
  CMLStatus s = cml_stat_min_max(vec, &min_val, &max_val);
  if (s != CML_SUCCESS) {
    return s;
  }
  *out = max_val - min_val;
  return CML_SUCCESS;
}

CMLStatus cml_stat_median(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t n = vec->size;
  if (n == 0 || vec->data == NULL) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double stack_buf[64];
  double *buf =
      (n <= 64) ? stack_buf : (double *)cml_malloc(n * sizeof(double));
  if (buf == NULL) {
    return CML_ERROR_OUT_OF_MEMORY;
  }

  cml_mem_copy(buf, vec->data, n * sizeof(double));
  qsort(buf, n, sizeof(double), compare_doubles);

  if (n % 2 == 1) {
    *out = buf[n / 2];
  } else {
    *out = (buf[(n / 2) - 1] + buf[n / 2]) / 2.0;
  }

  if (buf != stack_buf) {
    cml_free(buf);
  }
  return CML_SUCCESS;
}

CMLStatus cml_stat_variance(const CMLVector *vec, int is_sample, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t n = vec->size;
  if (n == 0 || vec->data == NULL) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (is_sample && n < 2) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  /* Welford's algorithm for numerical stability */
  double mean = 0.0;
  double M2 = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double x = vec->data[i];
    double delta = x - mean;
    mean += delta / (double)(i + 1);
    double delta2 = x - mean;
    M2 += delta * delta2;
  }

  *out = is_sample ? (M2 / (double)(n - 1)) : (M2 / (double)n);
  return CML_SUCCESS;
}

CMLStatus cml_stat_stddev(const CMLVector *vec, int is_sample, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  double var = 0.0;
  CMLStatus s = cml_stat_variance(vec, is_sample, &var);
  if (s != CML_SUCCESS) {
    return s;
  }
  *out = sqrt(var);
  return CML_SUCCESS;
}

CMLStatus cml_stat_percentile(const CMLVector *vec, double p, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (p < 0.0 || p > 100.0) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t n = vec->size;
  if (n == 0 || vec->data == NULL) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double stack_buf[64];
  double *buf =
      (n <= 64) ? stack_buf : (double *)cml_malloc(n * sizeof(double));
  if (buf == NULL) {
    return CML_ERROR_OUT_OF_MEMORY;
  }

  cml_mem_copy(buf, vec->data, n * sizeof(double));
  qsort(buf, n, sizeof(double), compare_doubles);

  if (n == 1) {
    *out = buf[0];
    if (buf != stack_buf)
      cml_free(buf);
    return CML_SUCCESS;
  }

  double index = (double)(n - 1) * (p / 100.0);
  size_t i = (size_t)floor(index);
  double f = index - (double)i;

  if (i >= n - 1) {
    *out = buf[n - 1];
  } else {
    *out = buf[i] + f * (buf[i + 1] - buf[i]);
  }

  if (buf != stack_buf) {
    cml_free(buf);
  }
  return CML_SUCCESS;
}

CMLStatus cml_stat_quantile(const CMLVector *vec, double q, double *out) {
  if (q < 0.0 || q > 1.0) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  return cml_stat_percentile(vec, q * 100.0, out);
}

CMLStatus cml_stat_quartiles(const CMLVector *vec, double *out_q1,
                             double *out_q2, double *out_q3) {
  if (vec == NULL || out_q1 == NULL || out_q2 == NULL || out_q3 == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  CMLStatus s1 = cml_stat_percentile(vec, 25.0, out_q1);
  if (s1 != CML_SUCCESS)
    return s1;
  CMLStatus s2 = cml_stat_percentile(vec, 50.0, out_q2);
  if (s2 != CML_SUCCESS)
    return s2;
  CMLStatus s3 = cml_stat_percentile(vec, 75.0, out_q3);
  if (s3 != CML_SUCCESS)
    return s3;

  return CML_SUCCESS;
}

CMLStatus cml_stat_iqr(const CMLVector *vec, double *out) {
  if (vec == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  double q1 = 0.0, q2 = 0.0, q3 = 0.0;
  CMLStatus s = cml_stat_quartiles(vec, &q1, &q2, &q3);
  if (s != CML_SUCCESS) {
    return s;
  }
  *out = q3 - q1;
  return CML_SUCCESS;
}

/* ==============================================================================
 * Column-Wise Statistics on Matrices
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_stat_matrix_col_means(const CMLMatrix *mat, CMLVector *out) {
  return cml_matrix_col_means(mat, out);
}

CMLStatus cml_stat_matrix_col_medians(const CMLMatrix *mat, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != mat->cols || mat->rows == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  CMLVector *col_vec = cml_vector_create(mat->rows);
  if (col_vec == NULL) {
    return CML_ERROR_OUT_OF_MEMORY;
  }

  for (size_t c = 0; c < mat->cols; ++c) {
    cml_matrix_get_col(mat, c, col_vec);
    double med = 0.0;
    cml_stat_median(col_vec, &med);
    out->data[c] = med;
  }

  cml_vector_destroy(col_vec);
  return CML_SUCCESS;
}

CMLStatus cml_stat_matrix_col_min(const CMLMatrix *mat, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != mat->cols || mat->rows == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t c = 0; c < mat->cols; ++c) {
    double m = mat->data[c];
    for (size_t r = 1; r < mat->rows; ++r) {
      double v = mat->data[r * mat->cols + c];
      if (v < m)
        m = v;
    }
    out->data[c] = m;
  }
  return CML_SUCCESS;
}

CMLStatus cml_stat_matrix_col_max(const CMLMatrix *mat, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != mat->cols || mat->rows == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t c = 0; c < mat->cols; ++c) {
    double m = mat->data[c];
    for (size_t r = 1; r < mat->rows; ++r) {
      double v = mat->data[r * mat->cols + c];
      if (v > m)
        m = v;
    }
    out->data[c] = m;
  }
  return CML_SUCCESS;
}

CMLStatus cml_stat_matrix_col_variance(const CMLMatrix *mat, int is_sample,
                                       CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != mat->cols || mat->rows == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (is_sample && mat->rows < 2) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  CMLVector *col_vec = cml_vector_create(mat->rows);
  if (col_vec == NULL) {
    return CML_ERROR_OUT_OF_MEMORY;
  }

  for (size_t c = 0; c < mat->cols; ++c) {
    cml_matrix_get_col(mat, c, col_vec);
    double var = 0.0;
    cml_stat_variance(col_vec, is_sample, &var);
    out->data[c] = var;
  }

  cml_vector_destroy(col_vec);
  return CML_SUCCESS;
}

CMLStatus cml_stat_matrix_col_stddev(const CMLMatrix *mat, int is_sample,
                                     CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  CMLStatus s = cml_stat_matrix_col_variance(mat, is_sample, out);
  if (s != CML_SUCCESS) {
    return s;
  }
  for (size_t c = 0; c < mat->cols; ++c) {
    out->data[c] = sqrt(out->data[c]);
  }
  return CML_SUCCESS;
}

CMLStatus cml_stat_matrix_col_iqr(const CMLMatrix *mat, CMLVector *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (out->size != mat->cols || mat->rows == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  CMLVector *col_vec = cml_vector_create(mat->rows);
  if (col_vec == NULL) {
    return CML_ERROR_OUT_OF_MEMORY;
  }

  for (size_t c = 0; c < mat->cols; ++c) {
    cml_matrix_get_col(mat, c, col_vec);
    double iqr_val = 0.0;
    cml_stat_iqr(col_vec, &iqr_val);
    out->data[c] = iqr_val;
  }

  cml_vector_destroy(col_vec);
  return CML_SUCCESS;
}

/* ==============================================================================
 * Covariance & Pearson Correlation
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_stat_covariance(const CMLVector *x, const CMLVector *y,
                              int is_sample, double *out) {
  if (x == NULL || y == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (x->size != y->size || x->size == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  size_t n = x->size;
  if (is_sample && n < 2) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double mean_x = 0.0, mean_y = 0.0;
  cml_stat_mean(x, &mean_x);
  cml_stat_mean(y, &mean_y);

  double sum_cov = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum_cov += (x->data[i] - mean_x) * (y->data[i] - mean_y);
  }

  double denom = is_sample ? (double)(n - 1) : (double)n;
  *out = sum_cov / denom;
  return CML_SUCCESS;
}

CMLStatus cml_stat_correlation(const CMLVector *x, const CMLVector *y,
                               double *out) {
  if (x == NULL || y == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (x->size != y->size || x->size == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double cov = 0.0, std_x = 0.0, std_y = 0.0;
  CMLStatus s = cml_stat_covariance(x, y, 1, &cov);
  if (s != CML_SUCCESS) {
    return s;
  }
  cml_stat_stddev(x, 1, &std_x);
  cml_stat_stddev(y, 1, &std_y);

  if (std_x <= 1e-15 || std_y <= 1e-15) {
    *out = 0.0; /* Constant feature has zero correlation */
    return CML_SUCCESS;
  }

  double r = cov / (std_x * std_y);
  if (r > 1.0)
    r = 1.0;
  if (r < -1.0)
    r = -1.0;
  *out = r;
  return CML_SUCCESS;
}

CMLStatus cml_stat_covariance_matrix(const CMLMatrix *mat, int is_sample,
                                     CMLMatrix *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (mat->rows == 0 || out->rows != mat->cols || out->cols != mat->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t d = mat->cols;
  CMLVector *col_i = cml_vector_create(mat->rows);
  CMLVector *col_j = cml_vector_create(mat->rows);
  if (col_i == NULL || col_j == NULL) {
    cml_vector_destroy(col_i);
    cml_vector_destroy(col_j);
    return CML_ERROR_OUT_OF_MEMORY;
  }

  for (size_t i = 0; i < d; ++i) {
    cml_matrix_get_col(mat, i, col_i);
    for (size_t j = i; j < d; ++j) {
      cml_matrix_get_col(mat, j, col_j);
      double cov = 0.0;
      cml_stat_covariance(col_i, col_j, is_sample, &cov);
      out->data[i * d + j] = cov;
      out->data[j * d + i] = cov;
    }
  }

  cml_vector_destroy(col_i);
  cml_vector_destroy(col_j);
  return CML_SUCCESS;
}

CMLMatrix *cml_stat_covariance_matrix_new(const CMLMatrix *mat, int is_sample) {
  if (mat == NULL || mat->rows == 0) {
    return NULL;
  }
  CMLMatrix *out = cml_matrix_create(mat->cols, mat->cols);
  if (out == NULL) {
    return NULL;
  }
  if (cml_stat_covariance_matrix(mat, is_sample, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }
  return out;
}

CMLStatus cml_stat_correlation_matrix(const CMLMatrix *mat, CMLMatrix *out) {
  if (mat == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (mat->rows == 0 || out->rows != mat->cols || out->cols != mat->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  size_t d = mat->cols;
  CMLVector *col_i = cml_vector_create(mat->rows);
  CMLVector *col_j = cml_vector_create(mat->rows);
  if (col_i == NULL || col_j == NULL) {
    cml_vector_destroy(col_i);
    cml_vector_destroy(col_j);
    return CML_ERROR_OUT_OF_MEMORY;
  }

  for (size_t i = 0; i < d; ++i) {
    cml_matrix_get_col(mat, i, col_i);
    out->data[i * d + i] = 1.0;
    for (size_t j = i + 1; j < d; ++j) {
      cml_matrix_get_col(mat, j, col_j);
      double r = 0.0;
      cml_stat_correlation(col_i, col_j, &r);
      out->data[i * d + j] = r;
      out->data[j * d + i] = r;
    }
  }

  cml_vector_destroy(col_i);
  cml_vector_destroy(col_j);
  return CML_SUCCESS;
}

CMLMatrix *cml_stat_correlation_matrix_new(const CMLMatrix *mat) {
  if (mat == NULL || mat->rows == 0) {
    return NULL;
  }
  CMLMatrix *out = cml_matrix_create(mat->cols, mat->cols);
  if (out == NULL) {
    return NULL;
  }
  if (cml_stat_correlation_matrix(mat, out) != CML_SUCCESS) {
    cml_matrix_destroy(out);
    return NULL;
  }
  return out;
}
