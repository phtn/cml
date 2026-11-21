/* ==============================================================================
 * src/dataset.c — CML Dataset Architecture & Management Implementation
 * ------------------------------------------------------------------------------
 */

#include "cml/dataset.h"
#include "cml/math.h"
#include "cml/memory.h"

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

  if ((num_samples > 0 && num_features > 0 && ds->X == NULL) ||
      (num_samples > 0 && ds->y == NULL)) {
    cml_dataset_destroy(ds);
    return NULL;
  }

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

  if (ds->feature_names) {
    for (size_t i = 0; i < ds->num_features; ++i) {
      if (ds->feature_names[i])
        cml_free(ds->feature_names[i]);
    }
    cml_free(ds->feature_names);
    ds->feature_names = NULL;
  }

  ds->feature_names = (char **)cml_calloc(ds->num_features, sizeof(char *));
  if (ds->feature_names == NULL)
    return CML_ERROR_OUT_OF_MEMORY;

  for (size_t i = 0; i < ds->num_features; ++i) {
    if (names[i]) {
      ds->feature_names[i] = cml_strdup(names[i]);
    }
  }

  return CML_SUCCESS;
}

CMLStatus cml_dataset_set_target_name(CMLDataset *ds, const char *name) {
  if (ds == NULL)
    return CML_ERROR_INVALID_ARGUMENT;

  if (ds->target_name) {
    cml_free(ds->target_name);
    ds->target_name = NULL;
  }
  if (name) {
    ds->target_name = cml_strdup(name);
  }
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

CMLStatus cml_dataset_split(const CMLDataset *src, double train_ratio,
                            double test_ratio, double val_ratio, bool shuffle,
                            uint64_t seed, CMLDataset **out_train,
                            CMLDataset **out_test, CMLDataset **out_val) {
  if (src == NULL || out_train == NULL || out_test == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (train_ratio < 0.0 || test_ratio < 0.0 || val_ratio < 0.0) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  double sum_ratios = train_ratio + test_ratio + val_ratio;
  if (!cml_is_close(sum_ratios, 1.0, 1e-5)) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  size_t n = src->num_samples;
  size_t n_train = (size_t)round((double)n * train_ratio);
  size_t n_val = (val_ratio > 0.0) ? (size_t)round((double)n * val_ratio) : 0;
  if (n_train + n_val > n) {
    n_train = n - n_val;
  }
  size_t n_test = n - n_train - n_val;

  /* Permutation array */
  size_t *indices = (size_t *)cml_malloc(n * sizeof(size_t));
  if (indices == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  for (size_t i = 0; i < n; ++i)
    indices[i] = i;

  if (shuffle) {
    CMLRng rng;
    cml_rng_init(&rng, seed);
    for (size_t i = n - 1; i > 0; --i) {
      size_t j = cml_rng_uniform(&rng, i + 1);
      if (i != j) {
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
      }
    }
  }

  CMLDataset *d_train = cml_dataset_create(n_train, src->num_features);
  CMLDataset *d_test = cml_dataset_create(n_test, src->num_features);
  CMLDataset *d_val = (n_val > 0 && out_val)
                          ? cml_dataset_create(n_val, src->num_features)
                          : NULL;

  /* Populate train */
  for (size_t i = 0; i < n_train; ++i) {
    size_t idx = indices[i];
    cml_mem_copy(&d_train->X->data[i * src->num_features],
                 &src->X->data[idx * src->num_features],
                 src->num_features * sizeof(double));
    d_train->y->data[i] = src->y->data[idx];
  }
  if (src->feature_names)
    cml_dataset_set_feature_names(d_train,
                                  (const char *const *)src->feature_names);
  if (src->target_name)
    cml_dataset_set_target_name(d_train, src->target_name);

  /* Populate val */
  if (d_val) {
    for (size_t i = 0; i < n_val; ++i) {
      size_t idx = indices[n_train + i];
      cml_mem_copy(&d_val->X->data[i * src->num_features],
                   &src->X->data[idx * src->num_features],
                   src->num_features * sizeof(double));
      d_val->y->data[i] = src->y->data[idx];
    }
    if (src->feature_names)
      cml_dataset_set_feature_names(d_val,
                                    (const char *const *)src->feature_names);
    if (src->target_name)
      cml_dataset_set_target_name(d_val, src->target_name);
  }

  /* Populate test */
  for (size_t i = 0; i < n_test; ++i) {
    size_t idx = indices[n_train + n_val + i];
    cml_mem_copy(&d_test->X->data[i * src->num_features],
                 &src->X->data[idx * src->num_features],
                 src->num_features * sizeof(double));
    d_test->y->data[i] = src->y->data[idx];
  }
  if (src->feature_names)
    cml_dataset_set_feature_names(d_test,
                                  (const char *const *)src->feature_names);
  if (src->target_name)
    cml_dataset_set_target_name(d_test, src->target_name);

  cml_free(indices);
  *out_train = d_train;
  *out_test = d_test;
  if (out_val)
    *out_val = d_val;

  return CML_SUCCESS;
}

CMLStatus cml_dataset_split_stratified(const CMLDataset *src,
                                       double train_ratio, double test_ratio,
                                       uint64_t seed, CMLDataset **out_train,
                                       CMLDataset **out_test) {
  if (src == NULL || out_train == NULL || out_test == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (train_ratio <= 0.0 || test_ratio <= 0.0)
    return CML_ERROR_INVALID_ARGUMENT;

  return cml_dataset_split(src, train_ratio, test_ratio, 0.0, true, seed,
                           out_train, out_test, NULL);
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
