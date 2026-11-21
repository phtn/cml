#ifndef CML_STATISTICS_H
#define CML_STATISTICS_H

/* ==============================================================================
 * cml/statistics.h — CML Statistical Engine
 *
 * Core statistics module for descriptive statistics, order statistics,
 * covariance, and Pearson correlation across vectors and matrices.
 *
 * Design conventions:
 *   - Functions return CMLStatus error codes.
 *   - Empty input returns CML_ERROR_INVALID_DIMENSION (except sum which returns
 * 0.0).
 *   - Inputs are never modified; working copies are sorted when needed.
 *   - Numerically stable algorithms (e.g. Welford's two-pass algorithm).
 * ------------------------------------------------------------------------------
 */

#include "cml/error.h"
#include "cml/matrix.h"
#include "cml/vector.h"

#include <stdbool.h>
#include <stddef.h>

/* ==============================================================================
 * Descriptive Statistics on Vectors
 * ------------------------------------------------------------------------------
 */

/* Sum of elements: sum = sum(x_i). Returns 0.0 for empty vectors. */
CMLStatus cml_stat_sum(const CMLVector *vec, double *out);

/* Arithmetic mean: mu = (1/n) * sum(x_i). */
CMLStatus cml_stat_mean(const CMLVector *vec, double *out);

/* Minimum element in vector. */
CMLStatus cml_stat_min(const CMLVector *vec, double *out);

/* Maximum element in vector. */
CMLStatus cml_stat_max(const CMLVector *vec, double *out);

/* Minimum and maximum elements in a single pass. */
CMLStatus cml_stat_min_max(const CMLVector *vec, double *out_min,
                           double *out_max);

/* Range: max - min. */
CMLStatus cml_stat_range(const CMLVector *vec, double *out);

/*
 * Median: middle value of sorted elements.
 * For odd length: middle value.
 * For even length: average of the two middle values.
 * Does not modify `vec`.
 */
CMLStatus cml_stat_median(const CMLVector *vec, double *out);

/*
 * Variance calculated using Welford's numerically stable algorithm.
 * is_sample == 0 -> Population variance (divisor n)
 * is_sample != 0 -> Sample variance (divisor n - 1; requires n >= 2)
 */
CMLStatus cml_stat_variance(const CMLVector *vec, int is_sample, double *out);

/* Standard deviation: sqrt(variance). */
CMLStatus cml_stat_stddev(const CMLVector *vec, int is_sample, double *out);

/*
 * Percentile (0.0 to 100.0) using linear interpolation:
 *   index = (n - 1) * (p / 100.0)
 *   val = y[i] + f * (y[i+1] - y[i])
 * Does not modify `vec`.
 */
CMLStatus cml_stat_percentile(const CMLVector *vec, double p, double *out);

/* Quantile (0.0 to 1.0). Convenience wrapper over cml_stat_percentile. */
CMLStatus cml_stat_quantile(const CMLVector *vec, double q, double *out);

/* Quartiles: Q1 (25%), Q2 (50% / median), Q3 (75%). */
CMLStatus cml_stat_quartiles(const CMLVector *vec, double *out_q1,
                             double *out_q2, double *out_q3);

/* Interquartile Range: IQR = Q3 - Q1. */
CMLStatus cml_stat_iqr(const CMLVector *vec, double *out);

/* ==============================================================================
 * Column-Wise Statistics on Matrices
 * ------------------------------------------------------------------------------
 */

/* Column means (out must have size == mat->cols) */
CMLStatus cml_stat_matrix_col_means(const CMLMatrix *mat, CMLVector *out);

/* Column medians (out must have size == mat->cols) */
CMLStatus cml_stat_matrix_col_medians(const CMLMatrix *mat, CMLVector *out);

/* Column minimums (out must have size == mat->cols) */
CMLStatus cml_stat_matrix_col_min(const CMLMatrix *mat, CMLVector *out);

/* Column maximums (out must have size == mat->cols) */
CMLStatus cml_stat_matrix_col_max(const CMLMatrix *mat, CMLVector *out);

/* Column variances (out must have size == mat->cols) */
CMLStatus cml_stat_matrix_col_variance(const CMLMatrix *mat, int is_sample,
                                       CMLVector *out);

/* Column standard deviations (out must have size == mat->cols) */
CMLStatus cml_stat_matrix_col_stddev(const CMLMatrix *mat, int is_sample,
                                     CMLVector *out);

/* Column Interquartile Ranges (out must have size == mat->cols) */
CMLStatus cml_stat_matrix_col_iqr(const CMLMatrix *mat, CMLVector *out);

/* ==============================================================================
 * Covariance & Pearson Correlation
 * ------------------------------------------------------------------------------
 */

/*
 * Covariance between two vectors:
 *   Cov(X, Y) = (1 / denom) * sum((x_i - mean_x) * (y_i - mean_y))
 * is_sample == 0 -> denom = n
 * is_sample != 0 -> denom = n - 1 (requires n >= 2)
 */
CMLStatus cml_stat_covariance(const CMLVector *x, const CMLVector *y,
                              int is_sample, double *out);

/*
 * Pearson correlation coefficient:
 *   r = Cov(X, Y) / (std_x * std_y)
 * Returns values in [-1.0, 1.0].
 * If either vector has zero variance (constant feature), out is set to 0.0.
 */
CMLStatus cml_stat_correlation(const CMLVector *x, const CMLVector *y,
                               double *out);

/*
 * Covariance matrix of dataset columns:
 *   For an (N x D) matrix, computes (D x D) pairwise covariance matrix.
 * out must have dimensions (D x D).
 */
CMLStatus cml_stat_covariance_matrix(const CMLMatrix *mat, int is_sample,
                                     CMLMatrix *out);
CMLMatrix *cml_stat_covariance_matrix_new(const CMLMatrix *mat, int is_sample);

/*
 * Pearson correlation matrix of dataset columns:
 *   For an (N x D) matrix, computes (D x D) pairwise correlation matrix.
 * Diagonal elements are 1.0, matrix is symmetric.
 * out must have dimensions (D x D).
 */
CMLStatus cml_stat_correlation_matrix(const CMLMatrix *mat, CMLMatrix *out);
CMLMatrix *cml_stat_correlation_matrix_new(const CMLMatrix *mat);

#endif /* CML_STATISTICS_H */
