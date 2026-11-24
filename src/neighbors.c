#include "cml/neighbors.h"
#include "cml/memory.h"
#include "internal.h"

#include <stdlib.h>
#include <string.h>

/* A distance represented as fraction * 2^exponent lets large distances be
 * ordered even if their magnitudes cannot be stored in a single double. */
typedef struct {
  double fraction;
  int exponent;
} Distance;

static Distance distance(const double *left, const double *right, size_t size) {
  double sum = 0.0;
  int scale = 0;
  for (size_t i = 0; i < size; ++i) {
    double difference = fabs(left[i] - right[i]);
    if (difference == 0.0)
      continue;
    bool halve = isinf(difference);
    if (halve)
      difference = fabs(0.5 * left[i] - 0.5 * right[i]);
    int exponent;
    double fraction = frexp(difference, &exponent);
    exponent += halve;
    if (sum == 0.0 || exponent > scale) {
      sum = scalbn(sum, 2 * (scale - exponent));
      scale = exponent;
    }
    double component = scalbn(fraction, exponent - scale);
    sum = fma(component, component, sum);
  }
  Distance result;
  result.fraction = frexp(sqrt(sum), &result.exponent);
  result.exponent += scale;
  return result;
}

static bool closer(Distance a, Distance b) {
  if (a.fraction == 0.0 || b.fraction == 0.0)
    return a.fraction < b.fraction;
  return a.exponent == b.exponent ? a.fraction < b.fraction
                                  : a.exponent < b.exponent;
}

struct CMLKNN {
  size_t neighbors;
  CMLMatrix *features;
  CMLVector *labels;
};

CMLKNN *cml_knn_create(size_t neighbors) {
  if (neighbors == 0)
    return NULL;
  CMLKNN *model = cml_malloc(sizeof(*model));
  if (model != NULL)
    *model = (CMLKNN){.neighbors = neighbors};
  return model;
}

void cml_knn_destroy(CMLKNN *model) {
  if (model != NULL) {
    cml_matrix_destroy(model->features);
    cml_vector_destroy(model->labels);
    cml_free(model);
  }
}

CMLStatus cml_knn_fit(CMLKNN *model, const CMLDataset *data) {
  if (model == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_dataset_validate(data);
  if (status == CML_SUCCESS)
    status = cml_matrix_storage(data->X);
  if (status == CML_SUCCESS)
    status = cml_vector_storage(data->y, data->num_samples);
  if (status != CML_SUCCESS)
    return status;
  if (model->neighbors > data->num_samples)
    return CML_ERROR_INVALID_DIMENSION;
  if (!cml_values_finite(data->X->data,
                         data->num_samples * data->num_features) ||
      !cml_values_finite(data->y->data, data->num_samples))
    return CML_ERROR_NUMERICAL;
  CMLMatrix *features = cml_matrix_clone(data->X);
  CMLVector *labels = cml_vector_clone(data->y);
  if (features == NULL || labels == NULL) {
    cml_matrix_destroy(features);
    cml_vector_destroy(labels);
    return CML_ERROR_OUT_OF_MEMORY;
  }
  cml_matrix_destroy(model->features);
  cml_vector_destroy(model->labels);
  model->features = features;
  model->labels = labels;
  return CML_SUCCESS;
}

typedef struct {
  Distance distance;
  size_t row;
  double label;
} Neighbor;

static int order_distance(const void *a, const void *b) {
  const Neighbor *left = a, *right = b;
  if (closer(left->distance, right->distance))
    return -1;
  if (closer(right->distance, left->distance))
    return 1;
  return (left->row > right->row) - (left->row < right->row);
}

static int order_label(const void *a, const void *b) {
  double left = ((const Neighbor *)a)->label;
  double right = ((const Neighbor *)b)->label;
  return (left > right) - (left < right);
}

CMLStatus cml_knn_predict(const CMLKNN *model, const CMLVector *input,
                          double *out) {
  if (model == NULL || model->features == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t rows = model->features->rows, cols = model->features->cols;
  CMLStatus status = cml_vector_storage(input, cols);
  if (status != CML_SUCCESS)
    return status;
  if (!cml_values_finite(input->data, cols))
    return CML_ERROR_NUMERICAL;
  if (rows > SIZE_MAX / sizeof(Neighbor))
    return CML_ERROR_OVERFLOW;
  Neighbor *neighbors = cml_calloc(rows, sizeof(*neighbors));
  if (neighbors == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  for (size_t i = 0; i < rows; ++i)
    neighbors[i] = (Neighbor){
        distance(input->data, model->features->data + i * cols, cols), i,
        model->labels->data[i]};
  qsort(neighbors, rows, sizeof(*neighbors), order_distance);
  qsort(neighbors, model->neighbors, sizeof(*neighbors), order_label);
  size_t votes = 0;
  double choice = neighbors[0].label;
  for (size_t start = 0; start < model->neighbors;) {
    size_t end = start + 1;
    while (end < model->neighbors &&
           neighbors[end].label == neighbors[start].label)
      ++end;
    if (end - start > votes) {
      votes = end - start;
      choice = neighbors[start].label;
    }
    start = end;
  }
  cml_free(neighbors);
  *out = choice;
  return CML_SUCCESS;
}

struct CMLKMeans {
  size_t clusters;
  CMLMatrix *centers;
};

CMLKMeans *cml_kmeans_create(size_t clusters) {
  if (clusters == 0)
    return NULL;
  CMLKMeans *model = cml_malloc(sizeof(*model));
  if (model != NULL)
    *model = (CMLKMeans){.clusters = clusters};
  return model;
}

void cml_kmeans_destroy(CMLKMeans *model) {
  if (model != NULL) {
    cml_matrix_destroy(model->centers);
    cml_free(model);
  }
}

static size_t nearest(const CMLMatrix *centers, const double *row,
                      Distance *error) {
  size_t best = 0;
  *error = distance(row, centers->data, centers->cols);
  for (size_t i = 1; i < centers->rows; ++i) {
    Distance value =
        distance(row, centers->data + i * centers->cols, centers->cols);
    if (closer(value, *error)) {
      best = i;
      *error = value;
    }
  }
  return best;
}

static void initialize_centers(const CMLMatrix *data, CMLMatrix *centers,
                               Distance *closest, bool *chosen, uint64_t seed) {
  size_t cols = data->cols, selected = cml_random_index(&seed, data->rows);
  for (size_t k = 0; k < centers->rows; ++k) {
    double *center = centers->data + k * cols;
    memcpy(center, data->data + selected * cols, cols * sizeof(double));
    chosen[selected] = true;
    if (k + 1 == centers->rows)
      break;
    size_t next = SIZE_MAX;
    for (size_t i = 0; i < data->rows; ++i) {
      if (chosen[i])
        continue;
      Distance value = distance(center, data->data + i * cols, cols);
      if (k == 0 || closer(value, closest[i]))
        closest[i] = value;
      if (next == SIZE_MAX || closer(closest[next], closest[i]))
        next = i;
    }
    selected = next;
  }
}

static bool update_centers(const CMLMatrix *data, CMLMatrix *centers,
                           CMLMatrix *means, size_t *counts,
                           Distance tolerance) {
  memset(means->data, 0, means->capacity * sizeof(double));
  memset(counts, 0, centers->rows * sizeof(*counts));
  size_t cols = data->cols;
  for (size_t i = 0; i < data->rows; ++i) {
    const double *row = data->data + i * cols;
    Distance error;
    size_t index = nearest(centers, row, &error);
    double factor = 1.0 / (double)++counts[index];
    double *mean = means->data + index * cols;
    for (size_t j = 0; j < cols; ++j) {
      double change = row[j] - mean[j];
      mean[j] = isfinite(change) ? fma(factor, change, mean[j])
                                 : fma(1.0 - factor, mean[j], factor * row[j]);
    }
  }
  bool converged = true;
  for (size_t i = 0; i < centers->rows; ++i) {
    if (counts[i] == 0)
      continue;
    double *old = centers->data + i * cols;
    const double *new_center = means->data + i * cols;
    if (closer(tolerance, distance(old, new_center, cols)))
      converged = false;
    memcpy(old, new_center, cols * sizeof(double));
  }
  return converged;
}

CMLStatus cml_kmeans_fit(CMLKMeans *model, const CMLMatrix *data, uint64_t seed,
                         size_t iterations, double tolerance,
                         CMLKMeansResult *out) {
  if (model == NULL || iterations == 0 || !isfinite(tolerance) ||
      tolerance < 0.0)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_matrix_storage(data);
  if (status != CML_SUCCESS)
    return status;
  if (data->rows < model->clusters)
    return CML_ERROR_INVALID_DIMENSION;
  if (!cml_values_finite(data->data, data->rows * data->cols))
    return CML_ERROR_NUMERICAL;
  if (data->rows > SIZE_MAX / sizeof(Distance))
    return CML_ERROR_OVERFLOW;

  CMLMatrix *centers = cml_matrix_create(model->clusters, data->cols);
  CMLMatrix *means = cml_matrix_create(model->clusters, data->cols);
  size_t *counts = cml_calloc(model->clusters, sizeof(*counts));
  Distance *closest = cml_calloc(data->rows, sizeof(*closest));
  bool *chosen = cml_calloc(data->rows, sizeof(*chosen));
  if (centers == NULL || means == NULL || counts == NULL || closest == NULL ||
      chosen == NULL) {
    status = CML_ERROR_OUT_OF_MEMORY;
    goto done;
  }
  initialize_centers(data, centers, closest, chosen, seed);
  Distance limit;
  limit.fraction = frexp(tolerance, &limit.exponent);
  CMLKMeansResult result = {0};
  while (result.iterations < iterations && !result.converged) {
    ++result.iterations;
    result.converged = update_centers(data, centers, means, counts, limit);
  }
  if (out != NULL) {
    for (size_t i = 0; i < data->rows; ++i) {
      Distance error;
      nearest(centers, data->data + i * data->cols, &error);
      result.inertia +=
          scalbn(error.fraction * error.fraction, 2 * error.exponent);
    }
    if (!isfinite(result.inertia)) {
      status = CML_ERROR_NUMERICAL;
      goto done;
    }
    *out = result;
  }
  cml_matrix_destroy(model->centers);
  model->centers = centers;
  centers = NULL;

done:
  cml_free(chosen);
  cml_free(closest);
  cml_free(counts);
  cml_matrix_destroy(means);
  cml_matrix_destroy(centers);
  return status;
}

CMLStatus cml_kmeans_get_centers(const CMLKMeans *model, CMLMatrix *out) {
  if (model == NULL || model->centers == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_matrix_storage(out);
  if (status != CML_SUCCESS)
    return status;
  const CMLMatrix *centers = model->centers;
  if (out->rows != centers->rows || out->cols != centers->cols)
    return CML_ERROR_INVALID_DIMENSION;
  memcpy(out->data, centers->data, out->rows * out->cols * sizeof(double));
  return CML_SUCCESS;
}

CMLStatus cml_kmeans_predict(const CMLKMeans *model, const CMLVector *input,
                             size_t *out) {
  if (model == NULL || model->centers == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_vector_storage(input, model->centers->cols);
  if (status != CML_SUCCESS)
    return status;
  if (!cml_values_finite(input->data, input->size))
    return CML_ERROR_NUMERICAL;
  Distance error;
  *out = nearest(model->centers, input->data, &error);
  return CML_SUCCESS;
}
