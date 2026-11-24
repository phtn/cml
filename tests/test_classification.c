#include "cml.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void test_binary_labels(void) {
  double values[] = {0.0, -0.0, 0.0, 1.0, 1.0};
  CMLVector labels = {values, 5, 5};
  CMLBinaryClassCounts counts;

  assert(cml_classification_validate_binary_labels(&labels, &counts) ==
         CML_SUCCESS);
  assert(counts.negative_count == 3 && counts.positive_count == 2);
  assert(cml_classification_validate_binary_labels(&labels, NULL) ==
         CML_SUCCESS);

  const double invalid[] = {-1.0, 2.0, 0.5, nextafter(1.0, 0.0)};
  for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
    values[4] = invalid[i];
    assert(cml_classification_validate_binary_labels(&labels, &counts) ==
           CML_ERROR_INVALID_ARGUMENT);
  }
  values[4] = NAN;
  assert(cml_classification_validate_binary_labels(&labels, &counts) ==
         CML_ERROR_NUMERICAL);
  values[4] = INFINITY;
  assert(cml_classification_validate_binary_labels(&labels, &counts) ==
         CML_ERROR_NUMERICAL);
  assert(counts.negative_count == 3 && counts.positive_count == 2);

  labels.size = 3;
  assert(cml_classification_validate_binary_labels(&labels, &counts) ==
         CML_SUCCESS);
  assert(counts.negative_count == 3 && counts.positive_count == 0);
  CMLVector empty = {0};
  assert(cml_classification_validate_binary_labels(&empty, &counts) ==
         CML_ERROR_INVALID_DIMENSION);
  assert(cml_classification_validate_binary_labels(NULL, &counts) ==
         CML_ERROR_INVALID_ARGUMENT);
}

static void test_binary_dataset(void) {
  double x[] = {1.2, 0.0, 2.4, 0.0, 3.1, 1.0, 4.5, 1.0, 5.7, 1.0};
  double y[] = {0.0, 0.0, 0.0, 1.0, 1.0};
  CMLMatrix features = {x, 5, 2, 10};
  CMLVector labels = {y, 5, 5};
  CMLDataset dataset = {
      .X = &features, .y = &labels, .num_samples = 5, .num_features = 2};
  CMLBinaryClassCounts counts;
  assert(cml_classification_validate_binary_dataset(&dataset, &counts) ==
         CML_SUCCESS);
  assert(counts.negative_count == 3 && counts.positive_count == 2);

  x[9] = NAN;
  assert(cml_classification_validate_binary_dataset(&dataset, &counts) ==
         CML_ERROR_NUMERICAL);
  x[9] = -INFINITY;
  assert(cml_classification_validate_binary_dataset(&dataset, &counts) ==
         CML_ERROR_NUMERICAL);
  x[9] = 1.0;
  y[4] = -1.0;
  assert(cml_classification_validate_binary_dataset(&dataset, &counts) ==
         CML_ERROR_INVALID_ARGUMENT);
  y[4] = 1.0;

  labels.size = 4;
  assert(cml_classification_validate_binary_dataset(&dataset, &counts) ==
         CML_ERROR_INVALID_DIMENSION);
  labels.size = 5;
  features.capacity = 9;
  assert(cml_classification_validate_binary_dataset(&dataset, &counts) ==
         CML_ERROR_INVALID_DIMENSION);
  features.capacity = 10;
  features.data = NULL;
  assert(cml_classification_validate_binary_dataset(&dataset, &counts) ==
         CML_ERROR_INVALID_ARGUMENT);
  features.data = x;

  features.cols = dataset.num_features = 0;
  assert(cml_classification_validate_binary_dataset(&dataset, &counts) ==
         CML_ERROR_INVALID_DIMENSION);
  features.cols = dataset.num_features = SIZE_MAX;
  assert(cml_classification_validate_binary_dataset(&dataset, &counts) ==
         CML_ERROR_OVERFLOW);
  CMLMatrix empty_features = {0};
  CMLVector empty_labels = {0};
  CMLDataset empty = {.X = &empty_features, .y = &empty_labels};
  assert(cml_classification_validate_binary_dataset(&empty, &counts) ==
         CML_ERROR_INVALID_DIMENSION);
  assert(counts.negative_count == 3 && counts.positive_count == 2);
}

int main(void) {
  test_binary_labels();
  test_binary_dataset();
  puts("Classification tests passed.");
  return 0;
}
