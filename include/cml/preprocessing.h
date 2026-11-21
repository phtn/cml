#ifndef CML_PREPROCESSING_H
#define CML_PREPROCESSING_H

/* ==============================================================================
 * cml/preprocessing.h — CML Data Preprocessing Engine
 *
 * Provides:
 *   - Missing data & invalid value detection and imputation strategies.
 *   - Outlier detection (IQR and Z-score methods) and handling (Winsorization,
 * median).
 *   - Feature transformations (log, log1p, power, clipping).
 *   - Feature scalers following the strict Fit & Transform architecture to
 *     prevent data leakage:
 *       - CMLMinMaxScaler (configurable target range [min, max])
 *       - CMLStandardScaler (zero mean, unit variance)
 *       - CMLRobustScaler (median centering, IQR scaling)
 *   - Preprocessing Pipeline (CMLPipeline) for staged transformations.
 * ------------------------------------------------------------------------------
 */

#include "cml/error.h"
#include "cml/matrix.h"
#include "cml/vector.h"

#include <stdbool.h>
#include <stddef.h>

/* ==============================================================================
 * Missing & Invalid Value Detection
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_prep_count_nan(const CMLVector *vec, size_t *out);
CMLStatus cml_prep_count_inf(const CMLVector *vec, size_t *out);
CMLStatus cml_prep_count_nonfinite(const CMLVector *vec, size_t *out);
CMLStatus cml_prep_is_finite(const CMLVector *vec, bool *out);

CMLStatus cml_prep_matrix_count_nan(const CMLMatrix *mat, size_t *out);
CMLStatus cml_prep_matrix_count_inf(const CMLMatrix *mat, size_t *out);
CMLStatus cml_prep_matrix_count_nonfinite(const CMLMatrix *mat, size_t *out);
CMLStatus cml_prep_matrix_is_finite(const CMLMatrix *mat, bool *out);

/* ==============================================================================
 * Missing & Invalid Value Imputation / Handling
 * ------------------------------------------------------------------------------
 */

/* Replace non-finite values in vector with constant replacement_val */
CMLStatus cml_prep_vector_replace_invalid(CMLVector *vec,
                                          double replacement_val);

/* Impute non-finite values with the mean of valid elements */
CMLStatus cml_prep_vector_impute_mean(CMLVector *vec);

/* Impute non-finite values with the median of valid elements */
CMLStatus cml_prep_vector_impute_median(CMLVector *vec);

/* Remove non-finite values, creating a newly allocated cleaned vector */
CMLStatus cml_prep_vector_remove_invalid(const CMLVector *vec,
                                         CMLVector **out_cleaned);

/* Column-wise replacement with constant */
CMLStatus cml_prep_matrix_replace_invalid(CMLMatrix *mat,
                                          double replacement_val);

/* Column-wise mean imputation on matrix columns */
CMLStatus cml_prep_matrix_impute_mean(CMLMatrix *mat);

/* Column-wise median imputation on matrix columns */
CMLStatus cml_prep_matrix_impute_median(CMLMatrix *mat);

/* Remove all rows containing at least one non-finite value */
CMLStatus cml_prep_matrix_remove_rows_with_invalid(const CMLMatrix *mat,
                                                   CMLMatrix **out_cleaned);

/* ==============================================================================
 * Outlier Detection & Handling
 * ------------------------------------------------------------------------------
 */

/*
 * IQR Outlier Detection on vector:
 *   Flags values outside [Q1 - k * IQR, Q3 + k * IQR] (typically k = 1.5).
 * out_mask must have at least vec->size elements (or NULL if not needed).
 */
CMLStatus cml_prep_outliers_iqr(const CMLVector *vec, double k, bool *out_mask,
                                size_t *out_count, double *out_lower,
                                double *out_upper);

/*
 * Z-score Outlier Detection on vector:
 *   Flags values where |x - mu| / sigma > threshold (typically threshold
 * = 3.0).
 */
CMLStatus cml_prep_outliers_zscore(const CMLVector *vec, double threshold,
                                   bool *out_mask, size_t *out_count);

/* Winsorization: clamp values outside [lower_bound, upper_bound] to the bounds
 */
CMLStatus cml_prep_winsorize(CMLVector *vec, double lower_bound,
                             double upper_bound);

/* Replace outliers indicated by mask with vector median */
CMLStatus cml_prep_outliers_replace_median(CMLVector *vec, const bool *mask);

/* Column-wise Winsorization on matrix using IQR bounds */
CMLStatus cml_prep_matrix_winsorize_cols(CMLMatrix *mat, double k_iqr);

/* ==============================================================================
 * Feature Transformations
 * ------------------------------------------------------------------------------
 */

/* Log transformation: x -> log(x). Returns CML_ERROR_NUMERICAL if any x <= 0 */
CMLStatus cml_prep_log(CMLVector *vec);

/* Log1p transformation: x -> log(1 + x). Returns CML_ERROR_NUMERICAL if any x <
 * -1 */
CMLStatus cml_prep_log1p(CMLVector *vec);

/* Power transformation: x -> x^p */
CMLStatus cml_prep_power(CMLVector *vec, double power);

/* Feature clipping: x -> clamp(x, min_val, max_val) */
CMLStatus cml_prep_clip(CMLVector *vec, double min_val, double max_val);

/* Matrix variants */
CMLStatus cml_prep_matrix_log(CMLMatrix *mat);
CMLStatus cml_prep_matrix_log1p(CMLMatrix *mat);
CMLStatus cml_prep_matrix_clip(CMLMatrix *mat, double min_val, double max_val);

/* ==============================================================================
 * Scalers (Fit & Transform Architecture — Data Leakage Prevention)
 * ------------------------------------------------------------------------------
 */

/* ==============================================================================
 * CMLMinMaxScaler
 * Transforms features to [feature_min, feature_max]:
 *   x' = feature_min + (x - min) / (max - min) * (feature_max - feature_min)
 * ------------------------------------------------------------------------------
 */
typedef struct CMLMinMaxScaler {
  double feature_min;
  double feature_max;
  size_t num_features;
  double *data_min;
  double *data_max;
  bool is_fitted;
} CMLMinMaxScaler;

CMLMinMaxScaler *cml_minmax_scaler_create(double feature_min,
                                          double feature_max);
CMLStatus cml_minmax_scaler_fit(CMLMinMaxScaler *scaler, const CMLMatrix *X);
CMLStatus cml_minmax_scaler_transform(const CMLMinMaxScaler *scaler,
                                      const CMLMatrix *X, CMLMatrix *out);
CMLStatus cml_minmax_scaler_fit_transform(CMLMinMaxScaler *scaler,
                                          const CMLMatrix *X, CMLMatrix *out);
CMLStatus cml_minmax_scaler_inverse_transform(const CMLMinMaxScaler *scaler,
                                              const CMLMatrix *X,
                                              CMLMatrix *out);
void cml_minmax_scaler_destroy(CMLMinMaxScaler *scaler);

/* ==============================================================================
 * CMLStandardScaler
 * Transforms features to zero mean and unit variance:
 *   z = (x - mu) / sigma
 * ------------------------------------------------------------------------------
 */
typedef struct CMLStandardScaler {
  size_t num_features;
  double *mean;
  double *stddev;
  bool is_fitted;
} CMLStandardScaler;

CMLStandardScaler *cml_standard_scaler_create(void);
CMLStatus cml_standard_scaler_fit(CMLStandardScaler *scaler,
                                  const CMLMatrix *X);
CMLStatus cml_standard_scaler_transform(const CMLStandardScaler *scaler,
                                        const CMLMatrix *X, CMLMatrix *out);
CMLStatus cml_standard_scaler_fit_transform(CMLStandardScaler *scaler,
                                            const CMLMatrix *X, CMLMatrix *out);
CMLStatus cml_standard_scaler_inverse_transform(const CMLStandardScaler *scaler,
                                                const CMLMatrix *X,
                                                CMLMatrix *out);
void cml_standard_scaler_destroy(CMLStandardScaler *scaler);

/* ==============================================================================
 * CMLRobustScaler
 * Transforms features using median and Interquartile Range:
 *   x' = (x - median) / IQR
 * ------------------------------------------------------------------------------
 */
typedef struct CMLRobustScaler {
  size_t num_features;
  double *median;
  double *iqr;
  bool is_fitted;
} CMLRobustScaler;

CMLRobustScaler *cml_robust_scaler_create(void);
CMLStatus cml_robust_scaler_fit(CMLRobustScaler *scaler, const CMLMatrix *X);
CMLStatus cml_robust_scaler_transform(const CMLRobustScaler *scaler,
                                      const CMLMatrix *X, CMLMatrix *out);
CMLStatus cml_robust_scaler_fit_transform(CMLRobustScaler *scaler,
                                          const CMLMatrix *X, CMLMatrix *out);
CMLStatus cml_robust_scaler_inverse_transform(const CMLRobustScaler *scaler,
                                              const CMLMatrix *X,
                                              CMLMatrix *out);
void cml_robust_scaler_destroy(CMLRobustScaler *scaler);

/* ==============================================================================
 * Preprocessing Pipeline
 * ------------------------------------------------------------------------------
 */

typedef enum CMLScalerType {
  CML_SCALER_NONE = 0,
  CML_SCALER_MINMAX = 1,
  CML_SCALER_STANDARD = 2,
  CML_SCALER_ROBUST = 3,
} CMLScalerType;

typedef struct CMLPipeline {
  bool impute_mean;
  bool clip_outliers;
  double clip_lower;
  double clip_upper;
  CMLScalerType scaler_type;
  void *scaler;
  bool is_fitted;
} CMLPipeline;

CMLPipeline *cml_pipeline_create(void);
CMLStatus cml_pipeline_set_imputer(CMLPipeline *pipe, bool enable_mean_impute);
CMLStatus cml_pipeline_set_clipper(CMLPipeline *pipe, bool enable_clip,
                                   double lower, double upper);
CMLStatus cml_pipeline_set_scaler(CMLPipeline *pipe, CMLScalerType scaler_type,
                                  double minmax_min, double minmax_max);
CMLStatus cml_pipeline_fit(CMLPipeline *pipe, const CMLMatrix *X);
CMLStatus cml_pipeline_transform(const CMLPipeline *pipe, const CMLMatrix *X,
                                 CMLMatrix *out);
void cml_pipeline_destroy(CMLPipeline *pipe);

#endif /* CML_PREPROCESSING_H */
