#include "cml/classification.h"
#include "cml/preprocessing.h"

#include <math.h>
#include <stdint.h>

CMLStatus
cml_classification_validate_binary_labels(const CMLVector *labels,
                                          CMLBinaryClassCounts *out_counts) {
  if (labels == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (labels->size == 0 || labels->capacity < labels->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (labels->size > SIZE_MAX / sizeof(double)) {
    return CML_ERROR_OVERFLOW;
  }
  if (labels->data == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  size_t positives = 0;
  for (size_t i = 0; i < labels->size; ++i) {
    double label = labels->data[i];
    if (!isfinite(label)) {
      return CML_ERROR_NUMERICAL;
    }
    if (label != 0.0 && label != 1.0) {
      return CML_ERROR_INVALID_ARGUMENT;
    }
    positives += (label == 1.0);
  }

  if (out_counts != NULL) {
    *out_counts = (CMLBinaryClassCounts){labels->size - positives, positives};
  }
  return CML_SUCCESS;
}

CMLStatus
cml_classification_validate_binary_dataset(const CMLDataset *dataset,
                                           CMLBinaryClassCounts *out_counts) {
  CMLStatus status = cml_dataset_validate(dataset);
  if (status != CML_SUCCESS) {
    return status;
  }

  const CMLMatrix *features = dataset->X;
  if (features->rows == 0 || features->cols == 0) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (features->rows > SIZE_MAX / sizeof(double) / features->cols) {
    return CML_ERROR_OVERFLOW;
  }
  if (features->capacity < features->rows * features->cols) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (features->data == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  bool finite;
  status = cml_prep_matrix_is_finite(features, &finite);
  if (status != CML_SUCCESS) {
    return status;
  }
  if (!finite) {
    return CML_ERROR_NUMERICAL;
  }

  return cml_classification_validate_binary_labels(dataset->y, out_counts);
}
