/* ==============================================================================
 * examples/data_analysis_example.c — CML Phase 4 Capstone Data Analysis
 *
 * Demonstrates:
 *   1. Dataset creation & validation (Housing dataset: Size, Bedrooms, Age -> Price)
 *   2. Descriptive feature and target statistics (mean, median, min, max, stddev)
 *   3. Data quality checks (missing/invalid values, outlier detection)
 *   4. Reproducible Fisher-Yates dataset shuffling
 *   5. Train/Test dataset splitting (80% train, 20% test)
 *   6. Data leakage prevention: fit StandardScaler on training set ONLY
 *   7. Transform both training and testing sets with learned parameters
 *   8. Pairwise feature correlation matrix
 *   9. Formatted summary report
 *  10. Memory clean-up
 * ==============================================================================
 */

#include "cml.h"

#include <stdio.h>

int main(void) {
  printf("========================================\n");
  printf("          CML DATA ANALYSIS\n");
  printf("========================================\n\n");

  /* 1. Create housing dataset (10 samples, 3 features, 1 target) */
  const size_t n_samples = 10;
  const size_t n_features = 3;
  CMLDataset *ds = cml_dataset_create(n_samples, n_features);

  const char *feat_names[] = {"Size (sqft)", "Bedrooms", "Age (years)"};
  cml_dataset_set_feature_names(ds, feat_names);
  cml_dataset_set_target_name(ds, "Price ($)");

  /* Data: Size, Bedrooms, Age, Price */
  double raw_data[10][4] = {
    { 850.0, 2.0, 10.0, 150000.0 },
    { 1200.0, 3.0,  8.0, 220000.0 },
    { 1600.0, 3.0,  5.0, 300000.0 },
    { 2100.0, 4.0,  3.0, 420000.0 },
    { 2500.0, 4.0,  2.0, 500000.0 },
    { 3000.0, 5.0,  1.0, 600000.0 },
    { 1100.0, 2.0, 15.0, 180000.0 },
    { 1800.0, 3.0,  7.0, 340000.0 },
    { 2200.0, 4.0,  6.0, 450000.0 },
    { 3500.0, 5.0,  2.0, 750000.0 }
  };

  for (size_t i = 0; i < n_samples; ++i) {
    ds->X->data[i * n_features + 0] = raw_data[i][0];
    ds->X->data[i * n_features + 1] = raw_data[i][1];
    ds->X->data[i * n_features + 2] = raw_data[i][2];
    ds->y->data[i] = raw_data[i][3];
  }

  /* 2. Validate dimensions */
  if (cml_dataset_validate(ds) != CML_SUCCESS) {
    printf("Dataset validation failed!\n");
    cml_dataset_destroy(ds);
    return 1;
  }

  printf("Samples:       %zu\n", ds->num_samples);
  printf("Features:      %zu\n", ds->num_features);
  printf("Target:        %s\n\n", ds->target_name);

  /* 3. Feature statistics */
  printf("FEATURE STATISTICS\n");
  printf("----------------------------------------\n");
  CMLVector *col = cml_vector_create(n_samples);
  for (size_t c = 0; c < n_features; ++c) {
    cml_matrix_get_col(ds->X, c, col);
    double mean = 0.0, med = 0.0, min_v = 0.0, max_v = 0.0, sd = 0.0;
    cml_stat_mean(col, &mean);
    cml_stat_median(col, &med);
    cml_stat_min(col, &min_v);
    cml_stat_max(col, &max_v);
    cml_stat_stddev(col, 0, &sd);

    printf("%s:\n", ds->feature_names[c]);
    printf("  Mean:        %10.2f\n", mean);
    printf("  Median:      %10.2f\n", med);
    printf("  Min:         %10.2f\n", min_v);
    printf("  Max:         %10.2f\n", max_v);
    printf("  Std Dev:     %10.2f\n\n", sd);
  }

  /* 4. Target statistics */
  double y_mean = 0.0, y_med = 0.0, y_min = 0.0, y_max = 0.0, y_sd = 0.0;
  cml_stat_mean(ds->y, &y_mean);
  cml_stat_median(ds->y, &y_med);
  cml_stat_min(ds->y, &y_min);
  cml_stat_max(ds->y, &y_max);
  cml_stat_stddev(ds->y, 0, &y_sd);

  printf("Target (%s):\n", ds->target_name);
  printf("  Mean:        %10.2f\n", y_mean);
  printf("  Median:      %10.2f\n", y_med);
  printf("  Min:         %10.2f\n", y_min);
  printf("  Max:         %10.2f\n", y_max);
  printf("  Std Dev:     %10.2f\n\n", y_sd);

  /* 5. Data quality */
  size_t nonfinite_cnt = 0;
  cml_prep_matrix_count_nonfinite(ds->X, &nonfinite_cnt);
  size_t outlier_cnt = 0;
  for (size_t c = 0; c < n_features; ++c) {
    cml_matrix_get_col(ds->X, c, col);
    size_t o_c = 0;
    cml_prep_outliers_iqr(col, 1.5, NULL, &o_c, NULL, NULL);
    outlier_cnt += o_c;
  }

  printf("DATA QUALITY\n");
  printf("----------------------------------------\n");
  printf("Missing / Non-finite values: %zu\n", nonfinite_cnt);
  printf("Outliers detected (IQR):     %zu\n\n", outlier_cnt);

  /* 6. Shuffle & Split */
  CMLDataset *train = NULL, *test = NULL;
  cml_dataset_split(ds, 0.8, 0.2, 0.0, true, 42, &train, &test, NULL);

  printf("DATA SPLIT (Seed 42)\n");
  printf("----------------------------------------\n");
  printf("Training set:   %zu samples\n", train->num_samples);
  printf("Testing set:    %zu samples\n\n", test->num_samples);

  /* 7. Fit Scaler on Training features ONLY (Prevent Data Leakage) */
  printf("PREPROCESSING (StandardScaler)\n");
  printf("----------------------------------------\n");
  CMLStandardScaler *scaler = cml_standard_scaler_create();
  cml_standard_scaler_fit(scaler, train->X);
  printf("Scaler fitted on:  Training set only\n");

  CMLMatrix *X_train_scaled = cml_matrix_create(train->num_samples, train->num_features);
  CMLMatrix *X_test_scaled = cml_matrix_create(test->num_samples, test->num_features);

  cml_standard_scaler_transform(scaler, train->X, X_train_scaled);
  cml_standard_scaler_transform(scaler, test->X, X_test_scaled);
  printf("Status:            Transformations complete\n\n");

  /* 8. Pairwise feature correlation matrix */
  printf("FEATURE CORRELATION MATRIX (Training)\n");
  printf("----------------------------------------\n");
  CMLMatrix *corr = cml_stat_correlation_matrix_new(train->X);
  cml_matrix_print(corr, "Correlation Matrix");
  printf("\n");

  /* Clean up */
  cml_matrix_destroy(corr);
  cml_matrix_destroy(X_train_scaled);
  cml_matrix_destroy(X_test_scaled);
  cml_standard_scaler_destroy(scaler);
  cml_dataset_destroy(train);
  cml_dataset_destroy(test);
  cml_vector_destroy(col);
  cml_dataset_destroy(ds);

  printf("========================================\n");
  printf("Analysis completed successfully.\n");
  return 0;
}
