#ifndef CML_DATASET_H
#define CML_DATASET_H

/* ==============================================================================
 * cml/dataset.h — CML Dataset Architecture & Management
 *
 * Provides:
 *   - CMLDataset structure encapsulating features (X) and targets (y).
 *   - Dataset lifecycle, sample access, validation, cloning, and metadata.
 *   - Fisher-Yates reproducible shuffling with custom seed (preventing bias).
 *   - Train / test / validation splitting with guaranteed disjoint subsets and
 *     feature-target alignment.
 *   - Stratified splitting for classification datasets.
 *   - CSV dataset loading and export.
 * ------------------------------------------------------------------------------
 */

#include "cml/error.h"
#include "cml/matrix.h"
#include "cml/vector.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ==============================================================================
 * CMLDataset Structure
 * ------------------------------------------------------------------------------
 */
typedef struct CMLDataset {
  CMLMatrix *X;         /* Feature matrix: (num_samples x num_features) */
  CMLVector *y;         /* Target vector: (num_samples) */
  size_t num_samples;   /* Number of samples */
  size_t num_features;  /* Number of features per sample */
  char **feature_names; /* Array of feature name strings (or NULL) */
  char *target_name;    /* Target name string (or NULL) */
} CMLDataset;

/* ==============================================================================
 * Lifecycle & Management
 * ------------------------------------------------------------------------------
 */

/* Create dataset with allocated internal X (num_samples x num_features) and y
 * (num_samples) */
CMLDataset *cml_dataset_create(size_t num_samples, size_t num_features);

/* Create dataset by cloning external feature matrix X and target vector y */
CMLDataset *cml_dataset_from_matrices(const CMLMatrix *X, const CMLVector *y);

/* Deep copy of dataset including metadata */
CMLDataset *cml_dataset_clone(const CMLDataset *src);

/* Validate internal dimension alignment and non-NULL pointers */
CMLStatus cml_dataset_validate(const CMLDataset *ds);

/* Destroy dataset and release all owned matrices, vectors, and string metadata
 */
void cml_dataset_destroy(CMLDataset *ds);

/* Set feature names (copies strings) */
CMLStatus cml_dataset_set_feature_names(CMLDataset *ds,
                                        const char *const *names);

/* Set target name (copies string) */
CMLStatus cml_dataset_set_target_name(CMLDataset *ds, const char *name);

/* Extract a single sample (features into out_features, target into out_target)
 */
CMLStatus cml_dataset_get_sample(const CMLDataset *ds, size_t index,
                                 CMLVector *out_features, double *out_target);

/* Set a single sample */
CMLStatus cml_dataset_set_sample(CMLDataset *ds, size_t index,
                                 const CMLVector *features, double target);

/* ==============================================================================
 * Shuffling (Fisher-Yates Algorithm with Seed)
 * ------------------------------------------------------------------------------
 */

/* In-place shuffle of vector elements */
CMLStatus cml_vector_shuffle(CMLVector *vec, uint64_t seed);

/* In-place shuffle of matrix rows */
CMLStatus cml_matrix_shuffle_rows(CMLMatrix *mat, uint64_t seed);

/*
 * In-place shuffle of dataset samples.
 * Guarantees features X[i] and targets y[i] move together.
 */
CMLStatus cml_dataset_shuffle(CMLDataset *ds, uint64_t seed);

/* ==============================================================================
 * Dataset Splitting
 * ------------------------------------------------------------------------------
 */

/*
 * Split dataset into training, testing, and optional validation subsets.
 * Ratios must sum to 1.0 within tolerance.
 * If val_ratio == 0.0, out_val can be NULL.
 * Subsets are completely disjoint and preserve feature-target alignment.
 */
CMLStatus cml_dataset_split(const CMLDataset *src, double train_ratio,
                            double test_ratio, double val_ratio, bool shuffle,
                            uint64_t seed, CMLDataset **out_train,
                            CMLDataset **out_test, CMLDataset **out_val);

/*
 * Stratified train/test split for discrete/classification targets.
 * Preserves class distribution between training and test sets.
 */
CMLStatus cml_dataset_split_stratified(const CMLDataset *src,
                                       double train_ratio, double test_ratio,
                                       uint64_t seed, CMLDataset **out_train,
                                       CMLDataset **out_test);

/* ==============================================================================
 * CSV Dataset Loader & Exporter
 * ------------------------------------------------------------------------------
 */

/*
 * Load numerical dataset from CSV.
 *   filepath: path to CSV file.
 *   has_header: whether the first row contains column names.
 *   target_col_idx: 0-based column index of target variable (or -1 if last
 * column).
 */
CMLStatus cml_dataset_from_csv(const char *filepath, bool has_header,
                               int target_col_idx, CMLDataset **out_ds);

/* Export dataset to CSV */
CMLStatus cml_dataset_to_csv(const CMLDataset *ds, const char *filepath);

#endif /* CML_DATASET_H */
