/* ==============================================================================
 * src/dataset.c — CML Dataset Architecture & Management Implementation
 * ------------------------------------------------------------------------------
 */

#include "cml/dataset.h"
#include "cml/math.h"
#include "cml/memory.h"
#include "internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Helper custom PRNG for deterministic, reproducible shuffling across platforms
 */
typedef struct CMLRng {
  uint64_t state;
} CMLRng;

static void cml_rng_init(CMLRng *rng, uint64_t seed) {
  rng->state = (seed == 0) ? 0x853c49e6748fea9bULL : seed;
}

static uint64_t cml_rng_next(CMLRng *rng) {
  uint64_t x = rng->state;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  rng->state = x;
  return x;
}

static size_t cml_rng_uniform(CMLRng *rng, size_t bound) {
  if (bound <= 1)
    return 0;
  return (size_t)(cml_rng_next(rng) % (uint64_t)bound);
}

/* Helper string duplication using cml_malloc */
static char *cml_strdup(const char *str) {
  if (str == NULL)
    return NULL;
  size_t len = strlen(str);
  char *copy = (char *)cml_malloc(len + 1);
  if (copy == NULL)
    return NULL;
  memcpy(copy, str, len + 1);
  return copy;
}

/* ==============================================================================
 * Lifecycle & Management
 * ------------------------------------------------------------------------------
 */

CMLDataset *cml_dataset_create(size_t num_samples, size_t num_features) {
  CMLDataset *ds = (CMLDataset *)cml_malloc(sizeof(CMLDataset));
  if (ds == NULL)
    return NULL;

  ds->num_samples = num_samples;
  ds->num_features = num_features;
  ds->feature_names = NULL;
  ds->target_name = NULL;

  ds->X = cml_matrix_create(num_samples, num_features);
  ds->y = cml_vector_create(num_samples);

  if (ds->X == NULL || ds->y == NULL) {
    cml_dataset_destroy(ds);
    return NULL;
  }

  if (num_samples == 0)
    ds->X->cols = num_features;
  return ds;
}

CMLDataset *cml_dataset_from_matrices(const CMLMatrix *X, const CMLVector *y) {
  if (X == NULL || y == NULL)
    return NULL;
  if (X->rows != y->size)
    return NULL;

  CMLDataset *ds = (CMLDataset *)cml_malloc(sizeof(CMLDataset));
  if (ds == NULL)
    return NULL;

  ds->num_samples = X->rows;
  ds->num_features = X->cols;
  ds->feature_names = NULL;
  ds->target_name = NULL;

  ds->X = cml_matrix_clone(X);
  ds->y = cml_vector_clone(y);

  if (ds->X == NULL || ds->y == NULL) {
    cml_dataset_destroy(ds);
    return NULL;
  }

  return ds;
}

CMLDataset *cml_dataset_clone(const CMLDataset *src) {
  if (src == NULL)
    return NULL;

  CMLDataset *clone = cml_dataset_from_matrices(src->X, src->y);
  if (clone == NULL)
    return NULL;

  if (src->feature_names) {
    cml_dataset_set_feature_names(clone,
                                  (const char *const *)src->feature_names);
  }
  if (src->target_name) {
    cml_dataset_set_target_name(clone, src->target_name);
  }

  return clone;
}

CMLStatus cml_dataset_validate(const CMLDataset *ds) {
  if (ds == NULL || ds->X == NULL || ds->y == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (ds->X->rows != ds->num_samples || ds->X->cols != ds->num_features ||
      ds->y->size != ds->num_samples) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  return CML_SUCCESS;
}

void cml_dataset_destroy(CMLDataset *ds) {
  if (ds == NULL)
    return;

  if (ds->X) {
    cml_matrix_destroy(ds->X);
    ds->X = NULL;
  }
  if (ds->y) {
    cml_vector_destroy(ds->y);
    ds->y = NULL;
  }
  if (ds->feature_names) {
    for (size_t i = 0; i < ds->num_features; ++i) {
      if (ds->feature_names[i]) {
        cml_free(ds->feature_names[i]);
      }
    }
    cml_free(ds->feature_names);
    ds->feature_names = NULL;
  }
  if (ds->target_name) {
    cml_free(ds->target_name);
    ds->target_name = NULL;
  }
  cml_free(ds);
}

CMLStatus cml_dataset_set_feature_names(CMLDataset *ds,
                                        const char *const *names) {
  if (ds == NULL || names == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (ds->num_features == 0)
    return CML_ERROR_INVALID_DIMENSION;
  char **copy = cml_calloc(ds->num_features, sizeof(*copy));
  if (copy == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  for (size_t i = 0; i < ds->num_features; ++i) {
    if (names[i] != NULL && (copy[i] = cml_strdup(names[i])) == NULL) {
      for (size_t j = 0; j < i; ++j)
        cml_free(copy[j]);
      cml_free(copy);
      return CML_ERROR_OUT_OF_MEMORY;
    }
  }
  if (ds->feature_names != NULL) {
    for (size_t i = 0; i < ds->num_features; ++i)
      cml_free(ds->feature_names[i]);
    cml_free(ds->feature_names);
  }
  ds->feature_names = copy;
  return CML_SUCCESS;
}

CMLStatus cml_dataset_set_target_name(CMLDataset *ds, const char *name) {
  if (ds == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  char *copy = name == NULL ? NULL : cml_strdup(name);
  if (name != NULL && copy == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  cml_free(ds->target_name);
  ds->target_name = copy;
  return CML_SUCCESS;
}

CMLStatus cml_dataset_get_sample(const CMLDataset *ds, size_t index,
                                 CMLVector *out_features, double *out_target) {
  if (ds == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (index >= ds->num_samples)
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;

  if (out_features) {
    CMLStatus s = cml_matrix_get_row(ds->X, index, out_features);
    if (s != CML_SUCCESS)
      return s;
  }
  if (out_target) {
    *out_target = ds->y->data[index];
  }

  return CML_SUCCESS;
}

CMLStatus cml_dataset_set_sample(CMLDataset *ds, size_t index,
                                 const CMLVector *features, double target) {
  if (ds == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (index >= ds->num_samples)
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;

  if (features) {
    CMLStatus s = cml_matrix_set_row(ds->X, index, features);
    if (s != CML_SUCCESS)
      return s;
  }
  ds->y->data[index] = target;

  return CML_SUCCESS;
}

/* ==============================================================================
 * Shuffling (Fisher-Yates Algorithm)
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_vector_shuffle(CMLVector *vec, uint64_t seed) {
  if (vec == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (vec->size <= 1)
    return CML_SUCCESS;

  CMLRng rng;
  cml_rng_init(&rng, seed);

  for (size_t i = vec->size - 1; i > 0; --i) {
    size_t j = cml_rng_uniform(&rng, i + 1);
    if (i != j) {
      double tmp = vec->data[i];
      vec->data[i] = vec->data[j];
      vec->data[j] = tmp;
    }
  }

  return CML_SUCCESS;
}

CMLStatus cml_matrix_shuffle_rows(CMLMatrix *mat, uint64_t seed) {
  if (mat == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (mat->rows <= 1)
    return CML_SUCCESS;

  CMLRng rng;
  cml_rng_init(&rng, seed);

  for (size_t i = mat->rows - 1; i > 0; --i) {
    size_t j = cml_rng_uniform(&rng, i + 1);
    if (i != j) {
      cml_matrix_swap_rows(mat, i, j);
    }
  }

  return CML_SUCCESS;
}

CMLStatus cml_dataset_shuffle(CMLDataset *ds, uint64_t seed) {
  if (ds == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (ds->num_samples <= 1)
    return CML_SUCCESS;

  CMLRng rng;
  cml_rng_init(&rng, seed);

  for (size_t i = ds->num_samples - 1; i > 0; --i) {
    size_t j = cml_rng_uniform(&rng, i + 1);
    if (i != j) {
      /* Swap row i and j in X */
      cml_matrix_swap_rows(ds->X, i, j);
      /* Swap target i and j in y */
      double tmp = ds->y->data[i];
      ds->y->data[i] = ds->y->data[j];
      ds->y->data[j] = tmp;
    }
  }

  return CML_SUCCESS;
}

/* ==============================================================================
 * Dataset Splitting
 * ------------------------------------------------------------------------------
 */

static CMLStatus validate_split_data(const CMLDataset *src) {
  CMLStatus status = cml_dataset_validate(src);
  if (status == CML_SUCCESS)
    status = cml_matrix_storage(src->X);
  if (status == CML_SUCCESS)
    status = cml_vector_storage(src->y, src->num_samples);
  return status;
}

static CMLStatus gather_rows(const CMLDataset *src, const size_t *indices,
                             size_t count, CMLDataset **out) {
  CMLDataset *subset = cml_dataset_create(count, src->num_features);
  if (subset == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  size_t cols = src->num_features;
  for (size_t i = 0; i < count; ++i) {
    memcpy(subset->X->data + i * cols, src->X->data + indices[i] * cols,
           cols * sizeof(double));
    subset->y->data[i] = src->y->data[indices[i]];
  }
  CMLStatus status = CML_SUCCESS;
  if (src->feature_names != NULL)
    status = cml_dataset_set_feature_names(
        subset, (const char *const *)src->feature_names);
  if (status == CML_SUCCESS && src->target_name != NULL)
    status = cml_dataset_set_target_name(subset, src->target_name);
  if (status == CML_SUCCESS)
    *out = subset;
  else
    cml_dataset_destroy(subset);
  return status;
}

static size_t *split_indices(size_t count, bool shuffle, uint64_t seed) {
  size_t *indices = cml_calloc(count, sizeof(*indices));
  if (indices != NULL) {
    for (size_t i = 0; i < count; ++i)
      indices[i] = i;
    if (shuffle)
      cml_shuffle_indices(indices, count, &seed);
  }
  return indices;
}

static CMLStatus split_contiguous(const CMLDataset *src, const size_t *indices,
                                  size_t n_train, size_t n_validation,
                                  CMLDataset **out_train, CMLDataset **out_test,
                                  CMLDataset **out_validation) {
  CMLDataset *train = NULL, *test = NULL, *validation = NULL;
  CMLStatus status = gather_rows(src, indices, n_train, &train);
  if (status == CML_SUCCESS && n_validation > 0 && out_validation != NULL)
    status = gather_rows(src, indices + n_train, n_validation, &validation);
  if (status == CML_SUCCESS)
    status = gather_rows(src, indices + n_train + n_validation,
                         src->num_samples - n_train - n_validation, &test);
  if (status == CML_SUCCESS) {
    *out_train = train;
    *out_test = test;
    if (out_validation != NULL)
      *out_validation = validation;
  } else {
    cml_dataset_destroy(train);
    cml_dataset_destroy(test);
    cml_dataset_destroy(validation);
  }
  return status;
}

CMLStatus cml_dataset_split(const CMLDataset *src, double train_ratio,
                            double test_ratio, double val_ratio, bool shuffle,
                            uint64_t seed, CMLDataset **out_train,
                            CMLDataset **out_test, CMLDataset **out_val) {
  if (out_train == NULL || out_test == NULL || out_train == out_test ||
      out_train == out_val || out_test == out_val || !isfinite(train_ratio) ||
      !isfinite(test_ratio) || !isfinite(val_ratio) || train_ratio < 0.0 ||
      test_ratio < 0.0 || val_ratio < 0.0 ||
      !cml_is_close(train_ratio + test_ratio + val_ratio, 1.0, 1e-5))
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = validate_split_data(src);
  if (status != CML_SUCCESS)
    return status;
  size_t n = src->num_samples;
  size_t n_train = (size_t)round((double)n * fmin(train_ratio, 1.0));
  size_t n_val = (size_t)round((double)n * fmin(val_ratio, 1.0));
  if (n_train > n - n_val)
    n_train = n - n_val;
  size_t *indices = split_indices(n, shuffle, seed);
  if (indices == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  status = split_contiguous(src, indices, n_train, n_val, out_train, out_test,
                            out_val);
  cml_free(indices);
  return status;
}

typedef struct {
  size_t index;
  double label;
} LabeledIndex;

static int compare_labels(const void *first, const void *second) {
  const LabeledIndex *a = first, *b = second;
  if (a->label != b->label)
    return (a->label > b->label) - (a->label < b->label);
  return (a->index > b->index) - (a->index < b->index);
}

typedef struct {
  size_t start, count, train;
  double remainder;
} ClassGroup;

static int compare_remainder(const void *first, const void *second) {
  const ClassGroup *a = first, *b = second;
  if (a->remainder != b->remainder)
    return (a->remainder < b->remainder) - (a->remainder > b->remainder);
  return (a->start > b->start) - (a->start < b->start);
}

CMLStatus cml_dataset_split_stratified(const CMLDataset *src,
                                       double train_ratio, double test_ratio,
                                       uint64_t seed, CMLDataset **out_train,
                                       CMLDataset **out_test) {
  if (out_train == NULL || out_test == NULL || out_train == out_test ||
      !isfinite(train_ratio) || !isfinite(test_ratio) || train_ratio <= 0.0 ||
      test_ratio <= 0.0 || !cml_is_close(train_ratio + test_ratio, 1.0, 1e-5))
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = validate_split_data(src);
  if (status != CML_SUCCESS)
    return status;
  size_t n = src->num_samples;
  if (!cml_values_finite(src->y->data, n))
    return CML_ERROR_NUMERICAL;
  if (n > SIZE_MAX / sizeof(ClassGroup) || n > SIZE_MAX / sizeof(LabeledIndex))
    return CML_ERROR_OVERFLOW;
  size_t *indices = cml_calloc(n, sizeof(*indices));
  LabeledIndex *ranked = cml_calloc(n, sizeof(*ranked));
  ClassGroup *groups = cml_calloc(n, sizeof(*groups));
  if (indices == NULL || ranked == NULL || groups == NULL) {
    status = CML_ERROR_OUT_OF_MEMORY;
    goto done;
  }
  for (size_t i = 0; i < n; ++i)
    ranked[i] = (LabeledIndex){i, src->y->data[i]};
  qsort(ranked, n, sizeof(*ranked), compare_labels);
  double fraction = train_ratio / (train_ratio + test_ratio);
  size_t group_count = 0, allocated = 0;
  for (size_t start = 0; start < n;) {
    size_t end = start + 1;
    while (end < n && ranked[end].label == ranked[start].label)
      ++end;
    double quota = (double)(end - start) * fraction;
    size_t train = (size_t)floor(quota);
    groups[group_count++] =
        (ClassGroup){start, end - start, train, quota - (double)train};
    allocated += train;
    start = end;
  }
  size_t target = (size_t)round((double)n * fraction);
  if (allocated > target || target - allocated > group_count) {
    status = CML_ERROR_NUMERICAL;
    goto done;
  }
  qsort(groups, group_count, sizeof(*groups), compare_remainder);
  for (size_t i = 0; i < target - allocated; ++i)
    ++groups[i].train;
  size_t train_at = 0, test_at = target;
  for (size_t i = 0; i < group_count; ++i) {
    ClassGroup group = groups[i];
    for (size_t j = group.count; j > 1; --j) {
      size_t from = group.start + j - 1;
      size_t to = group.start + cml_random_index(&seed, j);
      LabeledIndex saved = ranked[from];
      ranked[from] = ranked[to];
      ranked[to] = saved;
    }
    for (size_t j = 0; j < group.count; ++j)
      indices[j < group.train ? train_at++ : test_at++] =
          ranked[group.start + j].index;
  }
  cml_shuffle_indices(indices, target, &seed);
  cml_shuffle_indices(indices + target, n - target, &seed);
  status = split_contiguous(src, indices, target, 0, out_train, out_test, NULL);
done:
  cml_free(groups);
  cml_free(ranked);
  cml_free(indices);
  return status;
}

CMLStatus cml_dataset_kfold_split(const CMLDataset *src, size_t folds,
                                  size_t fold, bool shuffle, uint64_t seed,
                                  CMLDataset **out_train,
                                  CMLDataset **out_test) {
  if (out_train == NULL || out_test == NULL || out_train == out_test)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = validate_split_data(src);
  if (status != CML_SUCCESS)
    return status;
  size_t n = src->num_samples;
  if (folds < 2 || folds > n || fold >= folds)
    return CML_ERROR_INVALID_DIMENSION;
  size_t base = n / folds, extra = n % folds;
  size_t count = base + (size_t)(fold < extra);
  size_t start = fold * base + (fold < extra ? fold : extra);
  size_t *indices = split_indices(n, shuffle, seed);
  if (indices == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  CMLDataset *train = NULL, *test = NULL;
  status = gather_rows(src, indices + start, count, &test);
  if (status == CML_SUCCESS) {
    memmove(indices + start, indices + start + count,
            (n - start - count) * sizeof(*indices));
    status = gather_rows(src, indices, n - count, &train);
  }
  if (status == CML_SUCCESS) {
    *out_train = train;
    *out_test = test;
  } else {
    cml_dataset_destroy(test);
  }
  cml_free(indices);
  return status;
}

struct CMLBatchIterator {
  size_t *indices;
  size_t samples, batch_size, offset;
};

CMLBatchIterator *cml_batch_iterator_create(size_t samples, size_t batch_size,
                                            bool shuffle, uint64_t seed) {
  if (samples == 0 || batch_size == 0)
    return NULL;
  CMLBatchIterator *iterator = cml_malloc(sizeof(*iterator));
  if (iterator != NULL) {
    *iterator = (CMLBatchIterator){split_indices(samples, shuffle, seed),
                                   samples, batch_size, 0};
    if (iterator->indices == NULL) {
      cml_free(iterator);
      return NULL;
    }
  }
  return iterator;
}

void cml_batch_iterator_destroy(CMLBatchIterator *iterator) {
  if (iterator != NULL) {
    cml_free(iterator->indices);
    cml_free(iterator);
  }
}

CMLStatus cml_batch_iterator_reset(CMLBatchIterator *iterator, bool shuffle,
                                   uint64_t seed) {
  if (iterator == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < iterator->samples; ++i)
    iterator->indices[i] = i;
  if (shuffle)
    cml_shuffle_indices(iterator->indices, iterator->samples, &seed);
  iterator->offset = 0;
  return CML_SUCCESS;
}

CMLStatus cml_batch_iterator_next(CMLBatchIterator *iterator,
                                  const size_t **indices, size_t *count) {
  if (iterator == NULL || indices == NULL || count == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t remaining = iterator->samples - iterator->offset;
  *count = remaining < iterator->batch_size ? remaining : iterator->batch_size;
  *indices = *count == 0 ? NULL : iterator->indices + iterator->offset;
  iterator->offset += *count;
  return CML_SUCCESS;
}

/* ==============================================================================
 * CSV Dataset Loader & Exporter
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_dataset_from_csv(const char *filepath, bool has_header,
                               int target_col_idx, CMLDataset **out_ds) {
  if (filepath == NULL || out_ds == NULL)
    return CML_ERROR_INVALID_ARGUMENT;

  FILE *fp = fopen(filepath, "r");
  if (fp == NULL)
    return CML_ERROR_IO;

  char line[4096];
  size_t total_cols = 0;
  size_t sample_count = 0;
  char **col_names = NULL;

  /* First pass: determine column count and sample count */
  if (has_header) {
    if (fgets(line, sizeof(line), fp)) {
      /* Count columns in header */
      char *dup_line = cml_strdup(line);
      char *tok = strtok(dup_line, ",\r\n");
      while (tok) {
        ++total_cols;
        tok = strtok(NULL, ",\r\n");
      }
      cml_free(dup_line);

      /* Extract column names */
      col_names = (char **)cml_calloc(total_cols, sizeof(char *));
      dup_line = cml_strdup(line);
      tok = strtok(dup_line, ",\r\n");
      size_t c_idx = 0;
      while (tok && c_idx < total_cols) {
        /* trim leading whitespace */
        while (*tok == ' ' || *tok == '\t')
          ++tok;
        col_names[c_idx++] = cml_strdup(tok);
        tok = strtok(NULL, ",\r\n");
      }
      cml_free(dup_line);
    }
  }

  while (fgets(line, sizeof(line), fp)) {
    /* check non-empty line */
    if (strlen(line) > 1) {
      if (total_cols == 0) {
        char *dup_line = cml_strdup(line);
        char *tok = strtok(dup_line, ",\r\n");
        while (tok) {
          ++total_cols;
          tok = strtok(NULL, ",\r\n");
        }
        cml_free(dup_line);
      }
      ++sample_count;
    }
  }

  if (total_cols < 2 || sample_count == 0) {
    if (col_names) {
      for (size_t i = 0; i < total_cols; ++i)
        if (col_names[i])
          cml_free(col_names[i]);
      cml_free(col_names);
    }
    fclose(fp);
    return CML_ERROR_IO;
  }

  size_t t_idx =
      (target_col_idx < 0) ? (total_cols - 1) : (size_t)target_col_idx;
  if (t_idx >= total_cols)
    t_idx = total_cols - 1;

  size_t num_feats = total_cols - 1;
  CMLDataset *ds = cml_dataset_create(sample_count, num_feats);
  if (ds == NULL) {
    if (col_names) {
      for (size_t i = 0; i < total_cols; ++i)
        if (col_names[i])
          cml_free(col_names[i]);
      cml_free(col_names);
    }
    fclose(fp);
    return CML_ERROR_OUT_OF_MEMORY;
  }

  /* Set names */
  if (col_names) {
    char **feat_names = (char **)cml_calloc(num_feats, sizeof(char *));
    size_t f_pos = 0;
    for (size_t i = 0; i < total_cols; ++i) {
      if (i == t_idx) {
        cml_dataset_set_target_name(ds, col_names[i]);
      } else {
        feat_names[f_pos++] = col_names[i];
      }
    }
    cml_dataset_set_feature_names(ds, (const char *const *)feat_names);
    cml_free(feat_names);
    for (size_t i = 0; i < total_cols; ++i)
      if (col_names[i])
        cml_free(col_names[i]);
    cml_free(col_names);
  }

  /* Second pass: load data */
  rewind(fp);
  if (has_header) {
    if (!fgets(line, sizeof(line), fp)) {
      /* ignore */
    }
  }

  size_t row = 0;
  while (fgets(line, sizeof(line), fp) && row < sample_count) {
    char *dup_line = cml_strdup(line);
    char *tok = strtok(dup_line, ",\r\n");
    size_t col = 0;
    size_t feat_idx = 0;
    while (tok && col < total_cols) {
      double val = strtod(tok, NULL);
      if (col == t_idx) {
        ds->y->data[row] = val;
      } else {
        ds->X->data[row * num_feats + feat_idx] = val;
        ++feat_idx;
      }
      ++col;
      tok = strtok(NULL, ",\r\n");
    }
    cml_free(dup_line);
    ++row;
  }

  fclose(fp);
  *out_ds = ds;
  return CML_SUCCESS;
}

CMLStatus cml_dataset_to_csv(const CMLDataset *ds, const char *filepath) {
  if (ds == NULL || filepath == NULL)
    return CML_ERROR_INVALID_ARGUMENT;

  FILE *fp = fopen(filepath, "w");
  if (fp == NULL)
    return CML_ERROR_IO;

  /* Header */
  if (ds->feature_names) {
    for (size_t c = 0; c < ds->num_features; ++c) {
      fprintf(fp, "%s,", ds->feature_names[c]);
    }
    fprintf(fp, "%s\n", ds->target_name ? ds->target_name : "target");
  }

  for (size_t r = 0; r < ds->num_samples; ++r) {
    for (size_t c = 0; c < ds->num_features; ++c) {
      fprintf(fp, "%.6f,", ds->X->data[r * ds->num_features + c]);
    }
    fprintf(fp, "%.6f\n", ds->y->data[r]);
  }

  fclose(fp);
  return CML_SUCCESS;
}
