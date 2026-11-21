/* ==============================================================================
 * tests/test_preprocessing.c — Test suite for cml/preprocessing.h
 *
 * Covers:
 *   - Missing & invalid value detection (NaN, Inf, Nonfinite, finite check)
 *   - Imputation strategies (constant, mean, median, row removal)
 *   - Outlier detection (IQR, Z-score, Winsorization, median replacement)
 *   - Feature transformations (log, log1p, power, clip)
 *   - Scalers & Data Leakage Prevention (MinMax, Standard, Robust)
 *   - Preprocessing Pipeline
 * ------------------------------------------------------------------------------
 */

#include "cml/math.h"
#include "cml/preprocessing.h"
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
 * A. Missing & Invalid Values Detection & Imputation
 * ------------------------------------------------------------------------------
 */

static void test_missing_values(void) {
  section("missing & invalid values");

  /* [1.0, 2.0, NAN, 4.0, INFINITY, 6.0] */
  double raw[] = {1.0, 2.0, NAN, 4.0, INFINITY, 6.0};
  CMLVector *v = cml_vector_from_array(raw, 6);

  size_t nan_cnt = 0, inf_cnt = 0, nonfin_cnt = 0;
  bool fin = false;
  cml_prep_count_nan(v, &nan_cnt);
  cml_prep_count_inf(v, &inf_cnt);
  cml_prep_count_nonfinite(v, &nonfin_cnt);
  cml_prep_is_finite(v, &fin);

  TEST("NaN count == 1", nan_cnt == 1);
  TEST("Inf count == 1", inf_cnt == 1);
  TEST("Nonfinite count == 2", nonfin_cnt == 2);
  TEST("is_finite is false", fin == false);

  /* Replace with constant */
  CMLVector *v_rep = cml_vector_clone(v);
  cml_prep_vector_replace_invalid(v_rep, 0.0);
  cml_prep_count_nonfinite(v_rep, &nonfin_cnt);
  TEST("replace invalid with 0.0 makes vector finite", nonfin_cnt == 0);
  TEST("replaced values are 0.0",
       v_rep->data[2] == 0.0 && v_rep->data[4] == 0.0);
  cml_vector_destroy(v_rep);

  /* Remove invalid */
  CMLVector *v_clean = NULL;
  cml_prep_vector_remove_invalid(v, &v_clean);
  TEST("remove invalid size == 4", v_clean != NULL && v_clean->size == 4);
  TEST("remove invalid elements match",
       v_clean->data[0] == 1.0 && v_clean->data[1] == 2.0 &&
           v_clean->data[2] == 4.0 && v_clean->data[3] == 6.0);
  cml_vector_destroy(v_clean);

  /* Mean imputation: [2.0, 4.0, NAN, 8.0] -> mean of (2,4,8)=14/3=4.6667 */
  double raw_mean[] = {2.0, 4.0, NAN, 8.0};
  CMLVector *v_mean = cml_vector_from_array(raw_mean, 4);
  cml_prep_vector_impute_mean(v_mean);
  TEST("mean imputation replaces NAN with 14/3",
       cml_is_close(v_mean->data[2], 14.0 / 3.0, CML_EPSILON));
  cml_vector_destroy(v_mean);

  /* Matrix column-wise mean imputation */
  /* [1, 10], [3, NAN], [5, 30], [7, 40] -> col 1 mean is (10+30+40)/3 = 80/3 */
  double mat_raw[] = {1.0, 10.0, 3.0, NAN, 5.0, 30.0, 7.0, 40.0};
  CMLMatrix *m_nan = cml_matrix_from_array(mat_raw, 4, 2);
  cml_prep_matrix_impute_mean(m_nan);
  double v11 = 0.0;
  cml_matrix_get(m_nan, 1, 1, &v11);
  TEST("matrix column-wise mean imputation",
       cml_is_close(v11, 80.0 / 3.0, CML_EPSILON));
  cml_matrix_destroy(m_nan);

  cml_vector_destroy(v);
}

/* ==============================================================================
 * B. Outlier Detection & Handling
 * ------------------------------------------------------------------------------
 */

static void test_outliers(void) {
  section("outlier detection & handling");

  /* [10, 11, 12, 13, 14, 15, 1000] */
  double raw[] = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 1000.0};
  CMLVector *v = cml_vector_from_array(raw, 7);

  bool mask[7];
  size_t cnt = 0;
  double lower = 0.0, upper = 0.0;
  cml_prep_outliers_iqr(v, 1.5, mask, &cnt, &lower, &upper);

  TEST("IQR outlier count == 1", cnt == 1);
  TEST("IQR outlier mask flags 1000", mask[6] == true && mask[0] == false);

  /* Z-score outlier detection */
  cml_prep_outliers_zscore(v, 2.0, mask, &cnt);
  TEST("Z-score outlier detected", cnt >= 1 && mask[6] == true);

  /* Winsorize */
  CMLVector *v_win = cml_vector_clone(v);
  cml_prep_winsorize(v_win, 10.0, 20.0);
  TEST("winsorize clamps 1000 to 20", v_win->data[6] == 20.0);
  cml_vector_destroy(v_win);

  cml_vector_destroy(v);
}

/* ==============================================================================
 * C. Feature Transformations
 * ------------------------------------------------------------------------------
 */

static void test_transformations(void) {
  section("feature transformations");

  double raw[] = {1.0, (double)CML_E, 100.0};
  CMLVector *v = cml_vector_from_array(raw, 3);

  /* Log */
  cml_prep_log(v);
  TEST("log(1) == 0", cml_is_close(v->data[0], 0.0, CML_EPSILON));
  TEST("log(E) == 1", cml_is_close(v->data[1], 1.0, CML_EPSILON));
  cml_vector_destroy(v);

  /* Log on negative returns numerical error */
  double neg_raw[] = {-1.0, 2.0};
  CMLVector *v_neg = cml_vector_from_array(neg_raw, 2);
  TEST("log on negative returns CML_ERROR_NUMERICAL",
       cml_prep_log(v_neg) == CML_ERROR_NUMERICAL);
  cml_vector_destroy(v_neg);

  /* Clipping */
  double clip_raw[] = {1.0, 5.0, 10.0, 100.0, 200.0};
  CMLVector *v_clip = cml_vector_from_array(clip_raw, 5);
  cml_prep_clip(v_clip, 0.0, 50.0);
  TEST("clip lower bound intact", v_clip->data[0] == 1.0);
  TEST("clip upper bound clamped to 50",
       v_clip->data[3] == 50.0 && v_clip->data[4] == 50.0);
  cml_vector_destroy(v_clip);
}

/* ==============================================================================
 * D. Scalers & Data Leakage Prevention
 * ------------------------------------------------------------------------------
 */

static void test_scalers(void) {
  section("scalers & data leakage prevention");

  /*
   * Training set:
   * [ 10.0, 100.0 ]
   * [ 20.0, 200.0 ]
   * [ 30.0, 300.0 ]
   * [ 40.0, 400.0 ]
   * [ 50.0, 500.0 ]
   */
  double train_raw[] = {10.0,  100.0, 20.0,  200.0, 30.0,
                        300.0, 40.0,  400.0, 50.0,  500.0};
  CMLMatrix *X_train = cml_matrix_from_array(train_raw, 5, 2);

  /*
   * Test set:
   * [ 15.0, 250.0 ]
   * [ 60.0, 600.0 ]
   */
  double test_raw[] = {15.0, 250.0, 60.0, 600.0};
  CMLMatrix *X_test = cml_matrix_from_array(test_raw, 2, 2);

  /* 1. MinMaxScaler [0, 1] */
  CMLMinMaxScaler *minmax = cml_minmax_scaler_create(0.0, 1.0);
  CMLMatrix *X_train_norm = cml_matrix_create(5, 2);
  CMLMatrix *X_test_norm = cml_matrix_create(2, 2);

  cml_minmax_scaler_fit(minmax, X_train);
  cml_minmax_scaler_transform(minmax, X_train, X_train_norm);
  cml_minmax_scaler_transform(minmax, X_test, X_test_norm);

  /* Train: min should be 0.0, max should be 1.0 */
  TEST("minmax train min == 0.0",
       X_train_norm->data[0] == 0.0 && X_train_norm->data[1] == 0.0);
  TEST("minmax train max == 1.0",
       X_train_norm->data[8] == 1.0 && X_train_norm->data[9] == 1.0);

  /* Test: 60.0 -> (60 - 10) / 40 = 1.25 (evaluated with TRAIN parameters!) */
  TEST("minmax test scaled with train params",
       cml_is_close(X_test_norm->data[2], 1.25, CML_EPSILON));

  /* Inverse transform */
  CMLMatrix *X_train_rec = cml_matrix_create(5, 2);
  cml_minmax_scaler_inverse_transform(minmax, X_train_norm, X_train_rec);
  TEST("minmax inverse transform recovers train",
       cml_matrix_is_equal(X_train, X_train_rec, CML_EPSILON));

  cml_matrix_destroy(X_train_rec);
  cml_matrix_destroy(X_train_norm);
  cml_matrix_destroy(X_test_norm);
  cml_minmax_scaler_destroy(minmax);

  /* 2. StandardScaler */
  CMLStandardScaler *std_scaler = cml_standard_scaler_create();
  X_train_norm = cml_matrix_create(5, 2);
  X_test_norm = cml_matrix_create(2, 2);

  cml_standard_scaler_fit(std_scaler, X_train);
  cml_standard_scaler_transform(std_scaler, X_train, X_train_norm);
  cml_standard_scaler_transform(std_scaler, X_test, X_test_norm);

  /* Verify zero mean and unit variance on train set */
  CMLVector *means = cml_vector_create(2);
  CMLVector *stddevs = cml_vector_create(2);
  cml_stat_matrix_col_means(X_train_norm, means);
  cml_stat_matrix_col_stddev(X_train_norm, 0, stddevs);

  TEST("standard scaler train mean == 0",
       cml_is_close(means->data[0], 0.0, CML_EPSILON) &&
           cml_is_close(means->data[1], 0.0, CML_EPSILON));
  TEST("standard scaler train stddev == 1",
       cml_is_close(stddevs->data[0], 1.0, CML_EPSILON) &&
           cml_is_close(stddevs->data[1], 1.0, CML_EPSILON));

  cml_vector_destroy(means);
  cml_vector_destroy(stddevs);
  cml_matrix_destroy(X_train_norm);
  cml_matrix_destroy(X_test_norm);
  cml_standard_scaler_destroy(std_scaler);

  /* 3. RobustScaler */
  CMLRobustScaler *rob_scaler = cml_robust_scaler_create();
  X_train_norm = cml_matrix_create(5, 2);
  cml_robust_scaler_fit(rob_scaler, X_train);
  cml_robust_scaler_transform(rob_scaler, X_train, X_train_norm);

  CMLVector *meds = cml_vector_create(2);
  cml_stat_matrix_col_medians(X_train_norm, meds);
  TEST("robust scaler train median == 0",
       cml_is_close(meds->data[0], 0.0, CML_EPSILON));

  cml_vector_destroy(meds);
  cml_matrix_destroy(X_train_norm);
  cml_robust_scaler_destroy(rob_scaler);

  /* 4. Preprocessing Pipeline */
  CMLPipeline *pipe = cml_pipeline_create();
  cml_pipeline_set_imputer(pipe, true);
  cml_pipeline_set_scaler(pipe, CML_SCALER_STANDARD, 0.0, 1.0);

  cml_pipeline_fit(pipe, X_train);
  X_train_norm = cml_matrix_create(5, 2);
  cml_pipeline_transform(pipe, X_train, X_train_norm);
  TEST("pipeline execution succeeds",
       X_train_norm->data[0] != 0.0 || X_train_norm->data[1] != 0.0);

  cml_pipeline_destroy(pipe);
  cml_matrix_destroy(X_train_norm);
  cml_matrix_destroy(X_train);
  cml_matrix_destroy(X_test);
}

/* ==============================================================================
 * main
 * ------------------------------------------------------------------------------
 */

int main(void) {
  printf("=== CML Preprocessing Module Test Suite ===\n");

  test_missing_values();
  test_outliers();
  test_transformations();
  test_scalers();

  printf("\n--- Results: %d passed, %d failed ---\n", s_pass, s_fail);
  return (s_fail == 0) ? 0 : 1;
}
