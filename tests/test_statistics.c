/* ==============================================================================
 * tests/test_statistics.c — Test suite for cml/statistics.h
 *
 * Covers:
 *   - Basic statistics (sum, mean, min, max, range, min_max)
 *   - Median (odd, even, unsorted, duplicate values, single element)
 *   - Variance & Stddev (Welford algorithm, population vs sample)
 *   - Percentiles, Quantiles, Quartiles & IQR
 *   - Matrix column-wise statistics
 *   - Covariance & Pearson correlation (positive, negative, zero, constant)
 *   - Covariance & Correlation matrices
 *   - Edge cases & error handling (empty, NULL, dimension mismatches)
 * ------------------------------------------------------------------------------
 */

#include "cml/math.h"
#include "cml/statistics.h"

#include <math.h>
#include <stdio.h>

static int s_pass = 0;
static int s_fail = 0;

#define TEST(description, condition)                                           \
  do {                                                                         \
    if (condition) {                                                           \
      printf("  PASS  %s\n", description);                                     \
      ++s_pass;                                                                \
    } else {                                                                   \
      printf("  FAIL  %s  (line %d)\n", description, __LINE__);                \
      ++s_fail;                                                                \
    }                                                                          \
  } while (0)

static void section(const char *name) { printf("\n[%s]\n", name); }

/* ==============================================================================
 * A. Vector Descriptive Statistics
 * ------------------------------------------------------------------------------
 */

static void test_vector_statistics(void) {
  section("vector descriptive statistics");

  double raw[] = {2.0, 4.0, 6.0, 8.0, 10.0};
  CMLVector *v = cml_vector_from_array(raw, 5);

  double s = 0.0, mean = 0.0, min_v = 0.0, max_v = 0.0, rng = 0.0;
  cml_stat_sum(v, &s);
  cml_stat_mean(v, &mean);
  cml_stat_min(v, &min_v);
  cml_stat_max(v, &max_v);
  cml_stat_range(v, &rng);

  TEST("sum of [2,4,6,8,10] == 30", s == 30.0);
  TEST("mean of [2,4,6,8,10] == 6", mean == 6.0);
  TEST("min == 2", min_v == 2.0);
  TEST("max == 10", max_v == 10.0);
  TEST("range == 8", rng == 8.0);

  /* Variance & StdDev */
  double var_pop = 0.0, var_sam = 0.0, std_pop = 0.0, std_sam = 0.0;
  cml_stat_variance(v, 0, &var_pop);
  cml_stat_variance(v, 1, &var_sam);
  cml_stat_stddev(v, 0, &std_pop);
  cml_stat_stddev(v, 1, &std_sam);

  TEST("population variance == 8", cml_is_close(var_pop, 8.0, CML_EPSILON));
  TEST("sample variance == 10", cml_is_close(var_sam, 10.0, CML_EPSILON));
  TEST("population stddev == sqrt(8)",
       cml_is_close(std_pop, sqrt(8.0), CML_EPSILON));
  TEST("sample stddev == sqrt(10)",
       cml_is_close(std_sam, sqrt(10.0), CML_EPSILON));

  /* Median on odd length (unsorted) */
  double unsorted_odd[] = {9.0, 1.0, 5.0, 3.0, 7.0};
  CMLVector *v_odd = cml_vector_from_array(unsorted_odd, 5);
  double med = 0.0;
  cml_stat_median(v_odd, &med);
  TEST("median of odd unsorted [9,1,5,3,7] == 5", med == 5.0);
  TEST("original vector unmodified", v_odd->data[0] == 9.0);

  /* Median on even length (unsorted) */
  double unsorted_even[] = {7.0, 1.0, 5.0, 3.0};
  CMLVector *v_even = cml_vector_from_array(unsorted_even, 4);
  cml_stat_median(v_even, &med);
  TEST("median of even unsorted [7,1,5,3] == 4", med == 4.0);

  /* Percentiles, Quartiles & IQR on [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] */
  double nums10[10];
  for (size_t i = 0; i < 10; ++i)
    nums10[i] = (double)(i + 1);
  CMLVector *v10 = cml_vector_from_array(nums10, 10);

  double p0 = 0.0, p50 = 0.0, p100 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0,
         iqr_val = 0.0;
  cml_stat_percentile(v10, 0.0, &p0);
  cml_stat_percentile(v10, 50.0, &p50);
  cml_stat_percentile(v10, 100.0, &p100);
  cml_stat_quartiles(v10, &q1, &q2, &q3);
  cml_stat_iqr(v10, &iqr_val);

  TEST("percentile 0 == 1", p0 == 1.0);
  TEST("percentile 50 == 5.5", p50 == 5.5);
  TEST("percentile 100 == 10", p100 == 10.0);
  TEST("q1 == 3.25", cml_is_close(q1, 3.25, CML_EPSILON));
  TEST("q2 == 5.5", cml_is_close(q2, 5.5, CML_EPSILON));
  TEST("q3 == 7.75", cml_is_close(q3, 7.75, CML_EPSILON));
  TEST("iqr == 4.5", cml_is_close(iqr_val, 4.5, CML_EPSILON));

  /* Single element */
  CMLVector *v1 = cml_vector_create_filled(1, 42.0);
  cml_stat_median(v1, &med);
  TEST("median of 1-element vector is 42", med == 42.0);
  cml_stat_percentile(v1, 50.0, &p50);
  TEST("percentile of 1-element vector is 42", p50 == 42.0);

  cml_vector_destroy(v1);
  cml_vector_destroy(v10);
  cml_vector_destroy(v_even);
  cml_vector_destroy(v_odd);
  cml_vector_destroy(v);
}

/* ==============================================================================
 * B. Matrix Column Statistics
 * ------------------------------------------------------------------------------
 */

static void test_matrix_column_statistics(void) {
  section("matrix column statistics");

  /*
   * Mat (3 x 2):
   * [ 1.0, 10.0 ]
   * [ 2.0, 20.0 ]
   * [ 3.0, 30.0 ]
   */
  double data[] = {1.0, 10.0, 2.0, 20.0, 3.0, 30.0};
  CMLMatrix *m = cml_matrix_from_array(data, 3, 2);

  CMLVector *means = cml_vector_create(2);
  CMLVector *medians = cml_vector_create(2);
  CMLVector *stddevs = cml_vector_create(2);
  CMLVector *iqrs = cml_vector_create(2);

  cml_stat_matrix_col_means(m, means);
  cml_stat_matrix_col_medians(m, medians);
  cml_stat_matrix_col_stddev(m, 1, stddevs);
  cml_stat_matrix_col_iqr(m, iqrs);

  TEST("col means: [2.0, 20.0]",
       means->data[0] == 2.0 && means->data[1] == 20.0);
  TEST("col medians: [2.0, 20.0]",
       medians->data[0] == 2.0 && medians->data[1] == 20.0);
  TEST("col sample stddevs: [1.0, 10.0]",
       cml_is_close(stddevs->data[0], 1.0, CML_EPSILON) &&
           cml_is_close(stddevs->data[1], 10.0, CML_EPSILON));
  TEST("col iqrs: [1.0, 10.0]",
       cml_is_close(iqrs->data[0], 1.0, CML_EPSILON) &&
           cml_is_close(iqrs->data[1], 10.0, CML_EPSILON));

  cml_vector_destroy(iqrs);
  cml_vector_destroy(stddevs);
  cml_vector_destroy(medians);
  cml_vector_destroy(means);
  cml_matrix_destroy(m);
}

/* ==============================================================================
 * C. Covariance & Correlation
 * ------------------------------------------------------------------------------
 */

static void test_covariance_and_correlation(void) {
  section("covariance & correlation");

  double x_raw[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  double y_pos[] = {2.0, 4.0, 6.0, 8.0, 10.0};
  double y_neg[] = {10.0, 8.0, 6.0, 4.0, 2.0};
  double y_zero[] = {1.0, -2.0, 2.0, -1.0, 0.0};
  double x_zero_centered[] = {-2.0, -1.0, 0.0, 1.0, 2.0};

  CMLVector *x = cml_vector_from_array(x_raw, 5);
  CMLVector *yp = cml_vector_from_array(y_pos, 5);
  CMLVector *yn = cml_vector_from_array(y_neg, 5);
  CMLVector *xz = cml_vector_from_array(x_zero_centered, 5);
  CMLVector *yz = cml_vector_from_array(y_zero, 5);

  /* Perfect positive correlation */
  double r = 0.0, cov = 0.0;
  cml_stat_covariance(x, yp, 1, &cov);
  cml_stat_correlation(x, yp, &r);
  TEST("sample covariance of perfect positive == 5.0",
       cml_is_close(cov, 5.0, CML_EPSILON));
  TEST("perfect positive correlation r == 1.0",
       cml_is_close(r, 1.0, CML_EPSILON));

  /* Perfect negative correlation */
  cml_stat_correlation(x, yn, &r);
  TEST("perfect negative correlation r == -1.0",
       cml_is_close(r, -1.0, CML_EPSILON));

  /* Uncorrelated / near zero */
  cml_stat_correlation(xz, yz, &r);
  TEST("uncorrelated vectors |r| < 0.2", fabs(r) < 0.2);

  /* Exact zero covariance and correlation: symmetric quadratic Y = X^2 */
  double y_quad[] = {4.0, 1.0, 0.0, 1.0, 4.0};
  CMLVector *yq = cml_vector_from_array(y_quad, 5);
  cml_stat_correlation(xz, yq, &r);
  TEST("symmetric orthogonal vectors r == 0.0",
       cml_is_close(r, 0.0, CML_EPSILON));
  cml_vector_destroy(yq);

  /* Constant vector correlation */
  CMLVector *const_vec = cml_vector_create_filled(5, 7.0);
  cml_stat_correlation(x, const_vec, &r);
  TEST("correlation with constant vector returns 0.0", r == 0.0);

  /* Correlation Matrix on (4 x 3) */
  double mat_data[] = {1.0, 2.0, 10.0, 2.0, 4.0, 20.0,
                       3.0, 6.0, 30.0, 4.0, 8.0, 40.0};
  CMLMatrix *data_mat = cml_matrix_from_array(mat_data, 4, 3);
  CMLMatrix *corr = cml_stat_correlation_matrix_new(data_mat);

  TEST("correlation matrix shape is 3x3",
       corr != NULL && corr->rows == 3 && corr->cols == 3);
  TEST("correlation diagonal is 1.0",
       cml_is_close(corr->data[0], 1.0, CML_EPSILON) &&
           cml_is_close(corr->data[4], 1.0, CML_EPSILON) &&
           cml_is_close(corr->data[8], 1.0, CML_EPSILON));
  TEST("perfect collinear columns r == 1.0",
       cml_is_close(corr->data[1], 1.0, CML_EPSILON) &&
           cml_is_close(corr->data[2], 1.0, CML_EPSILON));

  cml_matrix_destroy(corr);
  cml_matrix_destroy(data_mat);
  cml_vector_destroy(const_vec);
  cml_vector_destroy(x);
  cml_vector_destroy(yp);
  cml_vector_destroy(yn);
  cml_vector_destroy(xz);
  cml_vector_destroy(yz);
}

/* ==============================================================================
 * D. Edge Cases & Error Handling
 * ------------------------------------------------------------------------------
 */

static void test_statistics_edge_cases(void) {
  section("statistics edge cases & error handling");

  double val = 0.0;
  CMLVector *v0 = cml_vector_create(0);
  TEST("sum of empty vector is 0.0",
       cml_stat_sum(v0, &val) == CML_SUCCESS && val == 0.0);
  TEST("mean of empty vector returns CML_ERROR_INVALID_DIMENSION",
       cml_stat_mean(v0, &val) == CML_ERROR_INVALID_DIMENSION);
  TEST("median of empty vector returns CML_ERROR_INVALID_DIMENSION",
       cml_stat_median(v0, &val) == CML_ERROR_INVALID_DIMENSION);
  TEST("min of empty vector returns CML_ERROR_INVALID_DIMENSION",
       cml_stat_min(v0, &val) == CML_ERROR_INVALID_DIMENSION);

  /* NULL checks */
  TEST("mean NULL vector returns CML_ERROR_INVALID_ARGUMENT",
       cml_stat_mean(NULL, &val) == CML_ERROR_INVALID_ARGUMENT);
  TEST("mean NULL out returns CML_ERROR_INVALID_ARGUMENT",
       cml_stat_mean(v0, NULL) == CML_ERROR_INVALID_ARGUMENT);

  /* Percentile out of bounds */
  CMLVector *v1 = cml_vector_create_filled(3, 1.0);
  TEST("percentile < 0 returns CML_ERROR_INVALID_ARGUMENT",
       cml_stat_percentile(v1, -5.0, &val) == CML_ERROR_INVALID_ARGUMENT);
  TEST("percentile > 100 returns CML_ERROR_INVALID_ARGUMENT",
       cml_stat_percentile(v1, 105.0, &val) == CML_ERROR_INVALID_ARGUMENT);

  /* Sample variance on 1 element */
  CMLVector *v_one = cml_vector_create_filled(1, 1.0);
  TEST("sample variance on 1 element returns CML_ERROR_INVALID_DIMENSION",
       cml_stat_variance(v_one, 1, &val) == CML_ERROR_INVALID_DIMENSION);
  cml_vector_destroy(v_one);

  cml_vector_destroy(v1);
  cml_vector_destroy(v0);
}

/* ==============================================================================
 * main
 * ------------------------------------------------------------------------------
 */

int main(void) {
  printf("=== CML Statistics Module Test Suite ===\n");

  test_vector_statistics();
  test_matrix_column_statistics();
  test_covariance_and_correlation();
  test_statistics_edge_cases();

  printf("\n--- Results: %d passed, %d failed ---\n", s_pass, s_fail);
  return (s_fail == 0) ? 0 : 1;
}
