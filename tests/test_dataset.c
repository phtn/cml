/* ==============================================================================
 * tests/test_dataset.c — Test suite for cml/dataset.h
 *
 * Covers:
 *   - Dataset lifecycle, metadata, cloning, destruction
 *   - Validation & bounds checking
 *   - Sample get and set operations
 *   - Shuffling (Fisher-Yates reproducibility & feature-target alignment)
 *   - Train / Validation / Test dataset splitting
 *   - CSV export and import round-trip
 * ------------------------------------------------------------------------------
 */

#include "cml/dataset.h"
#include "cml/math.h"

#include <stdio.h>
#include <unistd.h>

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
 * A. Lifecycle & Metadata
 * ------------------------------------------------------------------------------
 */

static void test_lifecycle_and_metadata(void) {
  section("dataset lifecycle & metadata");

  CMLDataset *ds = cml_dataset_create(4, 2);
  TEST("dataset create succeeds", ds != NULL);
  TEST("num_samples == 4", ds->num_samples == 4);
  TEST("num_features == 2", ds->num_features == 2);
  TEST("dataset validation succeeds", cml_dataset_validate(ds) == CML_SUCCESS);

  /* Set feature names and target name */
  const char *feats[] = {"SquareFeet", "Bedrooms"};
  cml_dataset_set_feature_names(ds, feats);
  cml_dataset_set_target_name(ds, "Price");

  TEST("feature names set", ds->feature_names != NULL);
  TEST("target name set", ds->target_name != NULL);

  /* Populate sample data */
  double row_data[2] = {1200.0, 3.0};
  CMLVector *sample_feats = cml_vector_from_array(row_data, 2);
  cml_dataset_set_sample(ds, 0, sample_feats, 250000.0);

  /* Retrieve sample */
  CMLVector *ret_feats = cml_vector_create(2);
  double ret_target = 0.0;
  cml_dataset_get_sample(ds, 0, ret_feats, &ret_target);
  TEST("get sample features match",
       ret_feats->data[0] == 1200.0 && ret_feats->data[1] == 3.0);
  TEST("get sample target matches", ret_target == 250000.0);

  /* Clone */
  CMLDataset *clone = cml_dataset_clone(ds);
  TEST("clone succeeds", clone != NULL);
  TEST("clone target matches", clone->y->data[0] == 250000.0);
  TEST("clone metadata matches", clone->target_name != NULL);

  cml_dataset_destroy(clone);
  cml_vector_destroy(sample_feats);
  cml_vector_destroy(ret_feats);
  cml_dataset_destroy(ds);
}

/* ==============================================================================
 * B. Shuffling & Feature-Target Alignment
 * ------------------------------------------------------------------------------
 */

static void test_shuffling(void) {
  section("dataset shuffling & alignment");

  /* Setup 5 samples */
  CMLDataset *ds = cml_dataset_create(5, 2);
  for (size_t i = 0; i < 5; ++i) {
    ds->X->data[i * 2 + 0] = (double)(i + 1);
    ds->X->data[i * 2 + 1] = (double)(i + 1) * 10.0;
    ds->y->data[i] = (double)(i + 1) * 100.0;
  }

  /* Shuffle with seed 42 */
  cml_dataset_shuffle(ds, 42);

  /* Verify feature-target alignment: X[i][0] * 100 must equal y[i] */
  bool aligned = true;
  for (size_t i = 0; i < 5; ++i) {
    double feat0 = ds->X->data[i * 2 + 0];
    double target = ds->y->data[i];
    if (feat0 * 100.0 != target)
      aligned = false;
  }
  TEST("features and targets remain strictly aligned after shuffle", aligned);

  /* Reproducibility: same seed produces identical order */
  CMLDataset *ds2 = cml_dataset_create(5, 2);
  for (size_t i = 0; i < 5; ++i) {
    ds2->X->data[i * 2 + 0] = (double)(i + 1);
    ds2->X->data[i * 2 + 1] = (double)(i + 1) * 10.0;
    ds2->y->data[i] = (double)(i + 1) * 100.0;
  }
  cml_dataset_shuffle(ds2, 42);
  TEST("reproducible shuffle produces identical order",
       cml_matrix_is_equal(ds->X, ds2->X, CML_EPSILON));

  cml_dataset_destroy(ds2);
  cml_dataset_destroy(ds);
}

/* ==============================================================================
 * C. Dataset Splitting
 * ------------------------------------------------------------------------------
 */

static void test_splitting(void) {
  section("dataset splitting");

  /* 10 samples */
  CMLDataset *ds = cml_dataset_create(10, 2);
  for (size_t i = 0; i < 10; ++i) {
    ds->X->data[i * 2 + 0] = (double)i;
    ds->X->data[i * 2 + 1] = (double)i * 2.0;
    ds->y->data[i] = (double)i * 10.0;
  }

  /* 80% train, 20% test */
  CMLDataset *train = NULL, *test = NULL;
  CMLStatus s =
      cml_dataset_split(ds, 0.8, 0.2, 0.0, true, 1234, &train, &test, NULL);
  TEST("80/20 split status CML_SUCCESS", s == CML_SUCCESS);
  TEST("train sample count == 8", train != NULL && train->num_samples == 8);
  TEST("test sample count == 2", test != NULL && test->num_samples == 2);

  /* Check alignment in train and test */
  bool train_aligned = true;
  for (size_t i = 0; i < train->num_samples; ++i) {
    if (train->X->data[i * 2 + 0] * 10.0 != train->y->data[i])
      train_aligned = false;
  }
  TEST("train subset preserves feature-target alignment", train_aligned);

  bool test_aligned = true;
  for (size_t i = 0; i < test->num_samples; ++i) {
    if (test->X->data[i * 2 + 0] * 10.0 != test->y->data[i])
      test_aligned = false;
  }
  TEST("test subset preserves feature-target alignment", test_aligned);

  cml_dataset_destroy(train);
  cml_dataset_destroy(test);

  /* 70/15/15 split (Train, Val, Test) */
  CMLDataset *val = NULL;
  s = cml_dataset_split(ds, 0.7, 0.15, 0.15, true, 4321, &train, &test, &val);
  TEST("70/15/15 split succeeds", s == CML_SUCCESS);
  TEST("train count + test count + val count == 10",
       train->num_samples + test->num_samples + val->num_samples == 10);

  cml_dataset_destroy(train);
  cml_dataset_destroy(test);
  cml_dataset_destroy(val);
  cml_dataset_destroy(ds);
}

/* ==============================================================================
 * D. CSV Export & Import Round-Trip
 * ------------------------------------------------------------------------------
 */

static void test_csv_round_trip(void) {
  section("csv export and import");

  CMLDataset *ds = cml_dataset_create(3, 2);
  const char *feats[] = {"Age", "Income"};
  cml_dataset_set_feature_names(ds, feats);
  cml_dataset_set_target_name(ds, "Spend");

  ds->X->data[0] = 25.0;
  ds->X->data[1] = 50000.0;
  ds->y->data[0] = 1500.0;
  ds->X->data[2] = 35.0;
  ds->X->data[3] = 75000.0;
  ds->y->data[1] = 2500.0;
  ds->X->data[4] = 45.0;
  ds->X->data[5] = 95000.0;
  ds->y->data[2] = 3500.0;

  const char *csv_path = "/tmp/cml_test_dataset.csv";
  CMLStatus s_export = cml_dataset_to_csv(ds, csv_path);
  TEST("csv export succeeds", s_export == CML_SUCCESS);

  CMLDataset *loaded = NULL;
  CMLStatus s_import = cml_dataset_from_csv(csv_path, true, 2, &loaded);
  TEST("csv import succeeds", s_import == CML_SUCCESS);
  TEST("loaded sample count == 3", loaded != NULL && loaded->num_samples == 3);
  TEST("loaded feature count == 2",
       loaded != NULL && loaded->num_features == 2);
  TEST("loaded values match exported",
       cml_matrix_is_equal(ds->X, loaded->X, 1e-4) &&
           cml_vector_is_equal(ds->y, loaded->y, 1e-4));

  unlink(csv_path);
  cml_dataset_destroy(loaded);
  cml_dataset_destroy(ds);
}

/* ==============================================================================
 * main
 * ------------------------------------------------------------------------------
 */

int main(void) {
  printf("=== CML Dataset Module Test Suite ===\n");

  test_lifecycle_and_metadata();
  test_shuffling();
  test_splitting();
  test_csv_round_trip();

  printf("\n--- Results: %d passed, %d failed ---\n", s_pass, s_fail);
  return (s_fail == 0) ? 0 : 1;
}
