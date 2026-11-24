#ifndef CML_CLASSIFICATION_H
#define CML_CLASSIFICATION_H

#include "cml/dataset.h"

typedef struct CMLBinaryClassCounts {
  size_t negative_count;
  size_t positive_count;
} CMLBinaryClassCounts;

/* Validate a nonempty vector of exact 0.0 and 1.0 labels. Finite values
 * outside this encoding return CML_ERROR_INVALID_ARGUMENT. NaN and infinity
 * return CML_ERROR_NUMERICAL. Empty or inconsistent shapes return
 * CML_ERROR_INVALID_DIMENSION. Unaddressable sizes return CML_ERROR_OVERFLOW.
 *
 * A single class is valid. out_counts may be NULL and is written only on
 * success. The input is borrowed for the duration of the call.
 */
CMLStatus
cml_classification_validate_binary_labels(const CMLVector *labels,
                                          CMLBinaryClassCounts *out_counts);

/* Validate the feature matrix and binary targets of an existing dataset.
 * Each sample must have one label and at least one feature. Non-finite
 * features return CML_ERROR_NUMERICAL and dimensions too large to address
 * return CML_ERROR_OVERFLOW. Label and count rules are the same as above.
 */
CMLStatus
cml_classification_validate_binary_dataset(const CMLDataset *dataset,
                                           CMLBinaryClassCounts *out_counts);

#endif /* CML_CLASSIFICATION_H */
