/* ==============================================================================
 * src/preprocessing.c — CML Data Preprocessing Engine Implementation
 * ------------------------------------------------------------------------------
 */

#include "cml/preprocessing.h"
#include "cml/math.h"
#include "cml/memory.h"
#include "cml/statistics.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==============================================================================
 * Missing & Invalid Value Detection
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_prep_count_nan(const CMLVector *vec, size_t *out) {
  if (vec == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t cnt = 0;
  for (size_t i = 0; i < vec->size; ++i) {
    if (isnan(vec->data[i]))
      ++cnt;
  }
  *out = cnt;
  return CML_SUCCESS;
}

CMLStatus cml_prep_count_inf(const CMLVector *vec, size_t *out) {
  if (vec == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t cnt = 0;
  for (size_t i = 0; i < vec->size; ++i) {
    if (isinf(vec->data[i]))
      ++cnt;
  }
  *out = cnt;
  return CML_SUCCESS;
}

CMLStatus cml_prep_count_nonfinite(const CMLVector *vec, size_t *out) {
  if (vec == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t cnt = 0;
  for (size_t i = 0; i < vec->size; ++i) {
    if (!isfinite(vec->data[i]))
      ++cnt;
  }
  *out = cnt;
  return CML_SUCCESS;
}

CMLStatus cml_prep_is_finite(const CMLVector *vec, bool *out) {
  if (vec == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < vec->size; ++i) {
    if (!isfinite(vec->data[i])) {
      *out = false;
      return CML_SUCCESS;
    }
  }
  *out = true;
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_count_nan(const CMLMatrix *mat, size_t *out) {
  if (mat == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t total = mat->rows * mat->cols;
  size_t cnt = 0;
  for (size_t i = 0; i < total; ++i) {
    if (isnan(mat->data[i]))
      ++cnt;
  }
  *out = cnt;
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_count_inf(const CMLMatrix *mat, size_t *out) {
  if (mat == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t total = mat->rows * mat->cols;
  size_t cnt = 0;
  for (size_t i = 0; i < total; ++i) {
    if (isinf(mat->data[i]))
      ++cnt;
  }
  *out = cnt;
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_count_nonfinite(const CMLMatrix *mat, size_t *out) {
  if (mat == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t total = mat->rows * mat->cols;
  size_t cnt = 0;
  for (size_t i = 0; i < total; ++i) {
    if (!isfinite(mat->data[i]))
      ++cnt;
  }
  *out = cnt;
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_is_finite(const CMLMatrix *mat, bool *out) {
  if (mat == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t total = mat->rows * mat->cols;
  for (size_t i = 0; i < total; ++i) {
    if (!isfinite(mat->data[i])) {
      *out = false;
      return CML_SUCCESS;
    }
  }
  *out = true;
  return CML_SUCCESS;
}

/* ==============================================================================
 * Missing & Invalid Value Imputation / Handling
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_prep_vector_replace_invalid(CMLVector *vec,
                                          double replacement_val) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < vec->size; ++i) {
    if (!isfinite(vec->data[i])) {
      vec->data[i] = replacement_val;
    }
  }
  return CML_SUCCESS;
}

CMLStatus cml_prep_vector_impute_mean(CMLVector *vec) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  double sum = 0.0;
  size_t valid_cnt = 0;

  for (size_t i = 0; i < vec->size; ++i) {
    if (isfinite(vec->data[i])) {
      sum += vec->data[i];
      ++valid_cnt;
    }
  }

  double fill_val = (valid_cnt > 0) ? (sum / (double)valid_cnt) : 0.0;
  return cml_prep_vector_replace_invalid(vec, fill_val);
}

CMLStatus cml_prep_vector_impute_median(CMLVector *vec) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t valid_cnt = 0;
  for (size_t i = 0; i < vec->size; ++i) {
    if (isfinite(vec->data[i]))
      ++valid_cnt;
  }

  if (valid_cnt == 0) {
    return cml_prep_vector_replace_invalid(vec, 0.0);
  }

  CMLVector *clean = cml_vector_create(valid_cnt);
  if (clean == NULL)
    return CML_ERROR_OUT_OF_MEMORY;

  size_t idx = 0;
  for (size_t i = 0; i < vec->size; ++i) {
    if (isfinite(vec->data[i])) {
      clean->data[idx++] = vec->data[i];
    }
  }

  double med = 0.0;
  cml_stat_median(clean, &med);
  cml_vector_destroy(clean);

  return cml_prep_vector_replace_invalid(vec, med);
}

CMLStatus cml_prep_vector_remove_invalid(const CMLVector *vec,
                                         CMLVector **out_cleaned) {
  if (vec == NULL || out_cleaned == NULL)
    return CML_ERROR_INVALID_ARGUMENT;

  size_t valid_cnt = 0;
  for (size_t i = 0; i < vec->size; ++i) {
    if (isfinite(vec->data[i]))
      ++valid_cnt;
  }

  CMLVector *clean = cml_vector_create(valid_cnt);
  if (clean == NULL)
    return CML_ERROR_OUT_OF_MEMORY;

  size_t idx = 0;
  for (size_t i = 0; i < vec->size; ++i) {
    if (isfinite(vec->data[i])) {
      clean->data[idx++] = vec->data[i];
    }
  }

  *out_cleaned = clean;
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_replace_invalid(CMLMatrix *mat,
                                          double replacement_val) {
  if (mat == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t total = mat->rows * mat->cols;
  for (size_t i = 0; i < total; ++i) {
    if (!isfinite(mat->data[i])) {
      mat->data[i] = replacement_val;
    }
  }
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_impute_mean(CMLMatrix *mat) {
  if (mat == NULL)
    return CML_ERROR_INVALID_ARGUMENT;

  for (size_t c = 0; c < mat->cols; ++c) {
    double sum = 0.0;
    size_t valid_cnt = 0;
    for (size_t r = 0; r < mat->rows; ++r) {
      double val = mat->data[r * mat->cols + c];
      if (isfinite(val)) {
        sum += val;
        ++valid_cnt;
      }
    }
    double mean_val = (valid_cnt > 0) ? (sum / (double)valid_cnt) : 0.0;
    for (size_t r = 0; r < mat->rows; ++r) {
      size_t idx = r * mat->cols + c;
      if (!isfinite(mat->data[idx])) {
        mat->data[idx] = mean_val;
      }
    }
  }

  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_impute_median(CMLMatrix *mat) {
  if (mat == NULL)
    return CML_ERROR_INVALID_ARGUMENT;

  CMLVector *col_clean = cml_vector_create(mat->rows);
  if (col_clean == NULL)
    return CML_ERROR_OUT_OF_MEMORY;

  for (size_t c = 0; c < mat->cols; ++c) {
    size_t valid_cnt = 0;
    for (size_t r = 0; r < mat->rows; ++r) {
      double val = mat->data[r * mat->cols + c];
      if (isfinite(val)) {
        col_clean->data[valid_cnt++] = val;
      }
    }
    col_clean->size = valid_cnt;
    double med = 0.0;
    if (valid_cnt > 0) {
      cml_stat_median(col_clean, &med);
    }
    for (size_t r = 0; r < mat->rows; ++r) {
      size_t idx = r * mat->cols + c;
      if (!isfinite(mat->data[idx])) {
        mat->data[idx] = med;
      }
    }
  }

  cml_vector_destroy(col_clean);
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_remove_rows_with_invalid(const CMLMatrix *mat,
                                                   CMLMatrix **out_cleaned) {
  if (mat == NULL || out_cleaned == NULL)
    return CML_ERROR_INVALID_ARGUMENT;

  size_t valid_rows = 0;
  for (size_t r = 0; r < mat->rows; ++r) {
    bool row_valid = true;
    for (size_t c = 0; c < mat->cols; ++c) {
      if (!isfinite(mat->data[r * mat->cols + c])) {
        row_valid = false;
        break;
      }
    }
    if (row_valid)
      ++valid_rows;
  }

  CMLMatrix *clean = cml_matrix_create(valid_rows, mat->cols);
  if (clean == NULL)
    return CML_ERROR_OUT_OF_MEMORY;

  size_t clean_r = 0;
  for (size_t r = 0; r < mat->rows; ++r) {
    bool row_valid = true;
    for (size_t c = 0; c < mat->cols; ++c) {
      if (!isfinite(mat->data[r * mat->cols + c])) {
        row_valid = false;
        break;
      }
    }
    if (row_valid) {
      cml_mem_copy(&clean->data[clean_r * mat->cols], &mat->data[r * mat->cols],
                   mat->cols * sizeof(double));
      ++clean_r;
    }
  }

  *out_cleaned = clean;
  return CML_SUCCESS;
}

/* ==============================================================================
 * Outlier Detection & Handling
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_prep_outliers_iqr(const CMLVector *vec, double k, bool *out_mask,
                                size_t *out_count, double *out_lower,
                                double *out_upper) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (vec->size == 0)
    return CML_ERROR_INVALID_DIMENSION;

  double q1 = 0.0, q2 = 0.0, q3 = 0.0;
  CMLStatus s = cml_stat_quartiles(vec, &q1, &q2, &q3);
  if (s != CML_SUCCESS)
    return s;

  double iqr = q3 - q1;
  double lower = q1 - k * iqr;
  double upper = q3 + k * iqr;

  size_t cnt = 0;
  for (size_t i = 0; i < vec->size; ++i) {
    double v = vec->data[i];
    bool is_outlier = (v < lower || v > upper);
    if (is_outlier)
      ++cnt;
    if (out_mask != NULL)
      out_mask[i] = is_outlier;
  }

  if (out_count)
    *out_count = cnt;
  if (out_lower)
    *out_lower = lower;
  if (out_upper)
    *out_upper = upper;
  return CML_SUCCESS;
}

CMLStatus cml_prep_outliers_zscore(const CMLVector *vec, double threshold,
                                   bool *out_mask, size_t *out_count) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (vec->size == 0)
    return CML_ERROR_INVALID_DIMENSION;

  double mean = 0.0, stddev = 0.0;
  cml_stat_mean(vec, &mean);
  cml_stat_stddev(vec, 0, &stddev);

  size_t cnt = 0;
  if (stddev <= 1e-15) {
    /* Constant vector -> no outliers */
    if (out_mask != NULL) {
      for (size_t i = 0; i < vec->size; ++i)
        out_mask[i] = false;
    }
    if (out_count)
      *out_count = 0;
    return CML_SUCCESS;
  }

  for (size_t i = 0; i < vec->size; ++i) {
    double z = fabs((vec->data[i] - mean) / stddev);
    bool is_outlier = (z > threshold);
    if (is_outlier)
      ++cnt;
    if (out_mask != NULL)
      out_mask[i] = is_outlier;
  }

  if (out_count)
    *out_count = cnt;
  return CML_SUCCESS;
}

CMLStatus cml_prep_winsorize(CMLVector *vec, double lower_bound,
                             double upper_bound) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < vec->size; ++i) {
    vec->data[i] = cml_clamp(vec->data[i], lower_bound, upper_bound);
  }
  return CML_SUCCESS;
}

CMLStatus cml_prep_outliers_replace_median(CMLVector *vec, const bool *mask) {
  if (vec == NULL || mask == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  double med = 0.0;
  CMLStatus s = cml_stat_median(vec, &med);
  if (s != CML_SUCCESS)
    return s;

  for (size_t i = 0; i < vec->size; ++i) {
    if (mask[i]) {
      vec->data[i] = med;
    }
  }
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_winsorize_cols(CMLMatrix *mat, double k_iqr) {
  if (mat == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (mat->rows == 0 || mat->cols == 0)
    return CML_SUCCESS;

  CMLVector *col_vec = cml_vector_create(mat->rows);
  if (col_vec == NULL)
    return CML_ERROR_OUT_OF_MEMORY;

  for (size_t c = 0; c < mat->cols; ++c) {
    cml_matrix_get_col(mat, c, col_vec);
    double lower = 0.0, upper = 0.0;
    cml_prep_outliers_iqr(col_vec, k_iqr, NULL, NULL, &lower, &upper);
    for (size_t r = 0; r < mat->rows; ++r) {
      size_t idx = r * mat->cols + c;
      mat->data[idx] = cml_clamp(mat->data[idx], lower, upper);
    }
  }

  cml_vector_destroy(col_vec);
  return CML_SUCCESS;
}

/* ==============================================================================
 * Feature Transformations
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_prep_log(CMLVector *vec) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < vec->size; ++i) {
    if (vec->data[i] <= 0.0 || !isfinite(vec->data[i])) {
      return CML_ERROR_NUMERICAL;
    }
  }
  for (size_t i = 0; i < vec->size; ++i) {
    vec->data[i] = log(vec->data[i]);
  }
  return CML_SUCCESS;
}

CMLStatus cml_prep_log1p(CMLVector *vec) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < vec->size; ++i) {
    if (vec->data[i] < -1.0 || !isfinite(vec->data[i])) {
      return CML_ERROR_NUMERICAL;
    }
  }
  for (size_t i = 0; i < vec->size; ++i) {
    vec->data[i] = log1p(vec->data[i]);
  }
  return CML_SUCCESS;
}

CMLStatus cml_prep_power(CMLVector *vec, double power) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < vec->size; ++i) {
    vec->data[i] = pow(vec->data[i], power);
  }
  return CML_SUCCESS;
}

CMLStatus cml_prep_clip(CMLVector *vec, double min_val, double max_val) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < vec->size; ++i) {
    vec->data[i] = cml_clamp(vec->data[i], min_val, max_val);
  }
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_log(CMLMatrix *mat) {
  if (mat == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t total = mat->rows * mat->cols;
  for (size_t i = 0; i < total; ++i) {
    if (mat->data[i] <= 0.0 || !isfinite(mat->data[i])) {
      return CML_ERROR_NUMERICAL;
    }
  }
  for (size_t i = 0; i < total; ++i) {
    mat->data[i] = log(mat->data[i]);
  }
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_log1p(CMLMatrix *mat) {
  if (mat == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t total = mat->rows * mat->cols;
  for (size_t i = 0; i < total; ++i) {
    if (mat->data[i] < -1.0 || !isfinite(mat->data[i])) {
      return CML_ERROR_NUMERICAL;
    }
  }
  for (size_t i = 0; i < total; ++i) {
    mat->data[i] = log1p(mat->data[i]);
  }
  return CML_SUCCESS;
}

CMLStatus cml_prep_matrix_clip(CMLMatrix *mat, double min_val, double max_val) {
  if (mat == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t total = mat->rows * mat->cols;
  for (size_t i = 0; i < total; ++i) {
    mat->data[i] = cml_clamp(mat->data[i], min_val, max_val);
  }
  return CML_SUCCESS;
}

/* ==============================================================================
 * Scalers (Fit & Transform Architecture — Data Leakage Prevention)
 * ------------------------------------------------------------------------------
 */

/* ==============================================================================
 * CMLMinMaxScaler
 * ------------------------------------------------------------------------------
 */

CMLMinMaxScaler *cml_minmax_scaler_create(double feature_min,
                                          double feature_max) {
  if (feature_min >= feature_max)
    return NULL;
  CMLMinMaxScaler *s = (CMLMinMaxScaler *)cml_malloc(sizeof(CMLMinMaxScaler));
  if (s == NULL)
    return NULL;
  s->feature_min = feature_min;
  s->feature_max = feature_max;
  s->num_features = 0;
  s->data_min = NULL;
  s->data_max = NULL;
  s->is_fitted = false;
  return s;
}

CMLStatus cml_minmax_scaler_fit(CMLMinMaxScaler *scaler, const CMLMatrix *X) {
  if (scaler == NULL || X == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (X->rows == 0 || X->cols == 0)
    return CML_ERROR_INVALID_DIMENSION;

  size_t d = X->cols;
  double *min_arr = (double *)cml_malloc(d * sizeof(double));
  double *max_arr = (double *)cml_malloc(d * sizeof(double));
  if (min_arr == NULL || max_arr == NULL) {
    cml_free(min_arr);
    cml_free(max_arr);
    return CML_ERROR_OUT_OF_MEMORY;
  }

  for (size_t c = 0; c < d; ++c) {
    double mn = X->data[c];
    double mx = X->data[c];
    for (size_t r = 1; r < X->rows; ++r) {
      double v = X->data[r * d + c];
      if (v < mn)
        mn = v;
      if (v > mx)
        mx = v;
    }
    min_arr[c] = mn;
    max_arr[c] = mx;
  }

  if (scaler->data_min)
    cml_free(scaler->data_min);
  if (scaler->data_max)
    cml_free(scaler->data_max);

  scaler->data_min = min_arr;
  scaler->data_max = max_arr;
  scaler->num_features = d;
  scaler->is_fitted = true;
  return CML_SUCCESS;
}

CMLStatus cml_minmax_scaler_transform(const CMLMinMaxScaler *scaler,
                                      const CMLMatrix *X, CMLMatrix *out) {
  if (scaler == NULL || X == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (!scaler->is_fitted)
    return CML_ERROR_INVALID_ARGUMENT;
  if (X->cols != scaler->num_features || out->rows != X->rows ||
      out->cols != X->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double f_min = scaler->feature_min;
  double f_max = scaler->feature_max;
  double f_range = f_max - f_min;

  for (size_t r = 0; r < X->rows; ++r) {
    for (size_t c = 0; c < X->cols; ++c) {
      double x_val = X->data[r * X->cols + c];
      double d_min = scaler->data_min[c];
      double d_max = scaler->data_max[c];
      double d_range = d_max - d_min;

      if (d_range <= 1e-15) {
        /* Constant feature policy: set to midpoint */
        out->data[r * X->cols + c] = (f_min + f_max) / 2.0;
      } else {
        out->data[r * X->cols + c] =
            f_min + ((x_val - d_min) / d_range) * f_range;
      }
    }
  }

  return CML_SUCCESS;
}

CMLStatus cml_minmax_scaler_fit_transform(CMLMinMaxScaler *scaler,
                                          const CMLMatrix *X, CMLMatrix *out) {
  CMLStatus s = cml_minmax_scaler_fit(scaler, X);
  if (s != CML_SUCCESS)
    return s;
  return cml_minmax_scaler_transform(scaler, X, out);
}

CMLStatus cml_minmax_scaler_inverse_transform(const CMLMinMaxScaler *scaler,
                                              const CMLMatrix *X,
                                              CMLMatrix *out) {
  if (scaler == NULL || X == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (!scaler->is_fitted)
    return CML_ERROR_INVALID_ARGUMENT;
  if (X->cols != scaler->num_features || out->rows != X->rows ||
      out->cols != X->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  double f_min = scaler->feature_min;
  double f_max = scaler->feature_max;
  double f_range = f_max - f_min;

  for (size_t r = 0; r < X->rows; ++r) {
    for (size_t c = 0; c < X->cols; ++c) {
      double norm_val = X->data[r * X->cols + c];
      double d_min = scaler->data_min[c];
      double d_max = scaler->data_max[c];
      double d_range = d_max - d_min;

      if (d_range <= 1e-15) {
        out->data[r * X->cols + c] = d_min;
      } else {
        out->data[r * X->cols + c] =
            d_min + ((norm_val - f_min) / f_range) * d_range;
      }
    }
  }

  return CML_SUCCESS;
}

void cml_minmax_scaler_destroy(CMLMinMaxScaler *scaler) {
  if (scaler == NULL)
    return;
  if (scaler->data_min)
    cml_free(scaler->data_min);
  if (scaler->data_max)
    cml_free(scaler->data_max);
  cml_free(scaler);
}

/* ==============================================================================
 * CMLStandardScaler
 * ------------------------------------------------------------------------------
 */

CMLStandardScaler *cml_standard_scaler_create(void) {
  CMLStandardScaler *s =
      (CMLStandardScaler *)cml_malloc(sizeof(CMLStandardScaler));
  if (s == NULL)
    return NULL;
  s->num_features = 0;
  s->mean = NULL;
  s->stddev = NULL;
  s->is_fitted = false;
  return s;
}

CMLStatus cml_standard_scaler_fit(CMLStandardScaler *scaler,
                                  const CMLMatrix *X) {
  if (scaler == NULL || X == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (X->rows == 0 || X->cols == 0)
    return CML_ERROR_INVALID_DIMENSION;

  size_t d = X->cols;
  double *mean_arr = (double *)cml_malloc(d * sizeof(double));
  double *std_arr = (double *)cml_malloc(d * sizeof(double));
  if (mean_arr == NULL || std_arr == NULL) {
    cml_free(mean_arr);
    cml_free(std_arr);
    return CML_ERROR_OUT_OF_MEMORY;
  }

  CMLVector *col_vec = cml_vector_create(X->rows);
  if (col_vec == NULL) {
    cml_free(mean_arr);
    cml_free(std_arr);
    return CML_ERROR_OUT_OF_MEMORY;
  }

  for (size_t c = 0; c < d; ++c) {
    cml_matrix_get_col(X, c, col_vec);
    double m = 0.0, sd = 0.0;
    cml_stat_mean(col_vec, &m);
    cml_stat_stddev(col_vec, 0, &sd); /* Population stddev */
    mean_arr[c] = m;
    std_arr[c] = sd;
  }

  cml_vector_destroy(col_vec);

  if (scaler->mean)
    cml_free(scaler->mean);
  if (scaler->stddev)
    cml_free(scaler->stddev);

  scaler->mean = mean_arr;
  scaler->stddev = std_arr;
  scaler->num_features = d;
  scaler->is_fitted = true;
  return CML_SUCCESS;
}

CMLStatus cml_standard_scaler_transform(const CMLStandardScaler *scaler,
                                        const CMLMatrix *X, CMLMatrix *out) {
  if (scaler == NULL || X == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (!scaler->is_fitted)
    return CML_ERROR_INVALID_ARGUMENT;
  if (X->cols != scaler->num_features || out->rows != X->rows ||
      out->cols != X->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t r = 0; r < X->rows; ++r) {
    for (size_t c = 0; c < X->cols; ++c) {
      double x_val = X->data[r * X->cols + c];
      double mu = scaler->mean[c];
      double sig = scaler->stddev[c];
      if (sig <= 1e-15) {
        out->data[r * X->cols + c] =
            0.0; /* Constant feature standardized to 0 */
      } else {
        out->data[r * X->cols + c] = (x_val - mu) / sig;
      }
    }
  }

  return CML_SUCCESS;
}

CMLStatus cml_standard_scaler_fit_transform(CMLStandardScaler *scaler,
                                            const CMLMatrix *X,
                                            CMLMatrix *out) {
  CMLStatus s = cml_standard_scaler_fit(scaler, X);
  if (s != CML_SUCCESS)
    return s;
  return cml_standard_scaler_transform(scaler, X, out);
}

CMLStatus cml_standard_scaler_inverse_transform(const CMLStandardScaler *scaler,
                                                const CMLMatrix *X,
                                                CMLMatrix *out) {
  if (scaler == NULL || X == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (!scaler->is_fitted)
    return CML_ERROR_INVALID_ARGUMENT;
  if (X->cols != scaler->num_features || out->rows != X->rows ||
      out->cols != X->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t r = 0; r < X->rows; ++r) {
    for (size_t c = 0; c < X->cols; ++c) {
      double z_val = X->data[r * X->cols + c];
      out->data[r * X->cols + c] = z_val * scaler->stddev[c] + scaler->mean[c];
    }
  }

  return CML_SUCCESS;
}

void cml_standard_scaler_destroy(CMLStandardScaler *scaler) {
  if (scaler == NULL)
    return;
  if (scaler->mean)
    cml_free(scaler->mean);
  if (scaler->stddev)
    cml_free(scaler->stddev);
  cml_free(scaler);
}

/* ==============================================================================
 * CMLRobustScaler
 * ------------------------------------------------------------------------------
 */

CMLRobustScaler *cml_robust_scaler_create(void) {
  CMLRobustScaler *s = (CMLRobustScaler *)cml_malloc(sizeof(CMLRobustScaler));
  if (s == NULL)
    return NULL;
  s->num_features = 0;
  s->median = NULL;
  s->iqr = NULL;
  s->is_fitted = false;
  return s;
}

CMLStatus cml_robust_scaler_fit(CMLRobustScaler *scaler, const CMLMatrix *X) {
  if (scaler == NULL || X == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (X->rows == 0 || X->cols == 0)
    return CML_ERROR_INVALID_DIMENSION;

  size_t d = X->cols;
  double *med_arr = (double *)cml_malloc(d * sizeof(double));
  double *iqr_arr = (double *)cml_malloc(d * sizeof(double));
  if (med_arr == NULL || iqr_arr == NULL) {
    cml_free(med_arr);
    cml_free(iqr_arr);
    return CML_ERROR_OUT_OF_MEMORY;
  }

  CMLVector *col_vec = cml_vector_create(X->rows);
  if (col_vec == NULL) {
    cml_free(med_arr);
    cml_free(iqr_arr);
    return CML_ERROR_OUT_OF_MEMORY;
  }

  for (size_t c = 0; c < d; ++c) {
    cml_matrix_get_col(X, c, col_vec);
    double med = 0.0, iqr_val = 0.0;
    cml_stat_median(col_vec, &med);
    cml_stat_iqr(col_vec, &iqr_val);
    med_arr[c] = med;
    iqr_arr[c] = iqr_val;
  }

  cml_vector_destroy(col_vec);

  if (scaler->median)
    cml_free(scaler->median);
  if (scaler->iqr)
    cml_free(scaler->iqr);

  scaler->median = med_arr;
  scaler->iqr = iqr_arr;
  scaler->num_features = d;
  scaler->is_fitted = true;
  return CML_SUCCESS;
}

CMLStatus cml_robust_scaler_transform(const CMLRobustScaler *scaler,
                                      const CMLMatrix *X, CMLMatrix *out) {
  if (scaler == NULL || X == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (!scaler->is_fitted)
    return CML_ERROR_INVALID_ARGUMENT;
  if (X->cols != scaler->num_features || out->rows != X->rows ||
      out->cols != X->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t r = 0; r < X->rows; ++r) {
    for (size_t c = 0; c < X->cols; ++c) {
      double x_val = X->data[r * X->cols + c];
      double med = scaler->median[c];
      double iqr_val = scaler->iqr[c];
      if (iqr_val <= 1e-15) {
        out->data[r * X->cols + c] = 0.0;
      } else {
        out->data[r * X->cols + c] = (x_val - med) / iqr_val;
      }
    }
  }

  return CML_SUCCESS;
}

CMLStatus cml_robust_scaler_fit_transform(CMLRobustScaler *scaler,
                                          const CMLMatrix *X, CMLMatrix *out) {
  CMLStatus s = cml_robust_scaler_fit(scaler, X);
  if (s != CML_SUCCESS)
    return s;
  return cml_robust_scaler_transform(scaler, X, out);
}

CMLStatus cml_robust_scaler_inverse_transform(const CMLRobustScaler *scaler,
                                              const CMLMatrix *X,
                                              CMLMatrix *out) {
  if (scaler == NULL || X == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (!scaler->is_fitted)
    return CML_ERROR_INVALID_ARGUMENT;
  if (X->cols != scaler->num_features || out->rows != X->rows ||
      out->cols != X->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }

  for (size_t r = 0; r < X->rows; ++r) {
    for (size_t c = 0; c < X->cols; ++c) {
      double scaled_val = X->data[r * X->cols + c];
      out->data[r * X->cols + c] =
          scaled_val * scaler->iqr[c] + scaler->median[c];
    }
  }

  return CML_SUCCESS;
}

void cml_robust_scaler_destroy(CMLRobustScaler *scaler) {
  if (scaler == NULL)
    return;
  if (scaler->median)
    cml_free(scaler->median);
  if (scaler->iqr)
    cml_free(scaler->iqr);
  cml_free(scaler);
}

/* ==============================================================================
 * Preprocessing Pipeline
 * ------------------------------------------------------------------------------
 */

CMLPipeline *cml_pipeline_create(void) {
  CMLPipeline *p = (CMLPipeline *)cml_malloc(sizeof(CMLPipeline));
  if (p == NULL)
    return NULL;
  p->impute_mean = false;
  p->clip_outliers = false;
  p->clip_lower = 0.0;
  p->clip_upper = 0.0;
  p->scaler_type = CML_SCALER_NONE;
  p->scaler = NULL;
  p->is_fitted = false;
  return p;
}

CMLStatus cml_pipeline_set_imputer(CMLPipeline *pipe, bool enable_mean_impute) {
  if (pipe == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  pipe->impute_mean = enable_mean_impute;
  return CML_SUCCESS;
}

CMLStatus cml_pipeline_set_clipper(CMLPipeline *pipe, bool enable_clip,
                                   double lower, double upper) {
  if (pipe == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  pipe->clip_outliers = enable_clip;
  pipe->clip_lower = lower;
  pipe->clip_upper = upper;
  return CML_SUCCESS;
}

CMLStatus cml_pipeline_set_scaler(CMLPipeline *pipe, CMLScalerType scaler_type,
                                  double minmax_min, double minmax_max) {
  if (pipe == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (pipe->scaler) {
    if (pipe->scaler_type == CML_SCALER_MINMAX) {
      cml_minmax_scaler_destroy((CMLMinMaxScaler *)pipe->scaler);
    } else if (pipe->scaler_type == CML_SCALER_STANDARD) {
      cml_standard_scaler_destroy((CMLStandardScaler *)pipe->scaler);
    } else if (pipe->scaler_type == CML_SCALER_ROBUST) {
      cml_robust_scaler_destroy((CMLRobustScaler *)pipe->scaler);
    }
    pipe->scaler = NULL;
  }

  pipe->scaler_type = scaler_type;
  if (scaler_type == CML_SCALER_MINMAX) {
    pipe->scaler = cml_minmax_scaler_create(minmax_min, minmax_max);
  } else if (scaler_type == CML_SCALER_STANDARD) {
    pipe->scaler = cml_standard_scaler_create();
  } else if (scaler_type == CML_SCALER_ROBUST) {
    pipe->scaler = cml_robust_scaler_create();
  }

  return CML_SUCCESS;
}

CMLStatus cml_pipeline_fit(CMLPipeline *pipe, const CMLMatrix *X) {
  if (pipe == NULL || X == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLMatrix *temp = cml_matrix_clone(X);
  if (temp == NULL)
    return CML_ERROR_OUT_OF_MEMORY;

  if (pipe->impute_mean) {
    cml_prep_matrix_impute_mean(temp);
  }
  if (pipe->clip_outliers) {
    cml_prep_matrix_clip(temp, pipe->clip_lower, pipe->clip_upper);
  }

  CMLStatus s = CML_SUCCESS;
  if (pipe->scaler_type == CML_SCALER_MINMAX) {
    s = cml_minmax_scaler_fit((CMLMinMaxScaler *)pipe->scaler, temp);
  } else if (pipe->scaler_type == CML_SCALER_STANDARD) {
    s = cml_standard_scaler_fit((CMLStandardScaler *)pipe->scaler, temp);
  } else if (pipe->scaler_type == CML_SCALER_ROBUST) {
    s = cml_robust_scaler_fit((CMLRobustScaler *)pipe->scaler, temp);
  }

  cml_matrix_destroy(temp);
  if (s == CML_SUCCESS) {
    pipe->is_fitted = true;
  }
  return s;
}

CMLStatus cml_pipeline_transform(const CMLPipeline *pipe, const CMLMatrix *X,
                                 CMLMatrix *out) {
  if (pipe == NULL || X == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (!pipe->is_fitted && pipe->scaler_type != CML_SCALER_NONE) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  CMLMatrix *temp = cml_matrix_clone(X);
  if (temp == NULL)
    return CML_ERROR_OUT_OF_MEMORY;

  if (pipe->impute_mean) {
    cml_prep_matrix_impute_mean(temp);
  }
  if (pipe->clip_outliers) {
    cml_prep_matrix_clip(temp, pipe->clip_lower, pipe->clip_upper);
  }

  CMLStatus s = CML_SUCCESS;
  if (pipe->scaler_type == CML_SCALER_MINMAX) {
    s = cml_minmax_scaler_transform((const CMLMinMaxScaler *)pipe->scaler, temp,
                                    out);
  } else if (pipe->scaler_type == CML_SCALER_STANDARD) {
    s = cml_standard_scaler_transform((const CMLStandardScaler *)pipe->scaler,
                                      temp, out);
  } else if (pipe->scaler_type == CML_SCALER_ROBUST) {
    s = cml_robust_scaler_transform((const CMLRobustScaler *)pipe->scaler, temp,
                                    out);
  } else {
    cml_mem_copy(out->data, temp->data,
                 temp->rows * temp->cols * sizeof(double));
  }

  cml_matrix_destroy(temp);
  return s;
}

void cml_pipeline_destroy(CMLPipeline *pipe) {
  if (pipe == NULL)
    return;
  if (pipe->scaler) {
    if (pipe->scaler_type == CML_SCALER_MINMAX) {
      cml_minmax_scaler_destroy((CMLMinMaxScaler *)pipe->scaler);
    } else if (pipe->scaler_type == CML_SCALER_STANDARD) {
      cml_standard_scaler_destroy((CMLStandardScaler *)pipe->scaler);
    } else if (pipe->scaler_type == CML_SCALER_ROBUST) {
      cml_robust_scaler_destroy((CMLRobustScaler *)pipe->scaler);
    }
  }
  cml_free(pipe);
}
