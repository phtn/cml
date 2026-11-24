#include "cml.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

static void neighbors(void) {
  double features[] = {-3, -2, -1, 1, 2, 3};
  double labels[] = {4, 4, 4, 9, 9, 9};
  CMLMatrix x = {features, 6, 1, 6};
  CMLVector y = {labels, 6, 6};
  CMLDataset data = {.X = &x, .y = &y, .num_samples = 6, .num_features = 1};
  CMLKNN *model = cml_knn_create(3);
  double sample = -1.5, label;
  CMLVector p = {&sample, 1, 1};
  assert(model != NULL && cml_knn_fit(model, &data) == CML_SUCCESS);
  labels[0] = labels[1] = labels[2] = 9;
  assert(cml_knn_predict(model, &p, &label) == CML_SUCCESS && label == 4.0);
  sample = 1.5;
  assert(cml_knn_predict(model, &p, &label) == CML_SUCCESS && label == 9.0);
  labels[0] = NAN;
  assert(cml_knn_fit(model, &data) == CML_ERROR_NUMERICAL);
  assert(cml_knn_predict(model, &p, &label) == CML_SUCCESS && label == 9.0);
  cml_knn_destroy(model);

  features[0] = -DBL_MAX;
  features[1] = -0.5 * DBL_MAX;
  labels[0] = 0.0;
  labels[1] = 1.0;
  data.num_samples = x.rows = y.size = 2;
  model = cml_knn_create(1);
  assert(cml_knn_fit(model, &data) == CML_SUCCESS);
  sample = DBL_MAX;
  assert(cml_knn_predict(model, &p, &label) == CML_SUCCESS && label == 1.0);
  cml_knn_destroy(model);
}

static void clusters(void) {
  double samples[] = {-4, 1, -2, 1, 3, 4, 5, 4};
  CMLMatrix data = {samples, 4, 2, 8};
  CMLKMeans *model = cml_kmeans_create(2);
  assert(model != NULL);
  CMLKMeansResult result;
  assert(cml_kmeans_fit(model, &data, 5, 50, 1e-10, &result) == CML_SUCCESS);
  assert(result.converged && fabs(result.inertia - 4.0) < 1e-13);
  double centers[4], repeat[4];
  CMLMatrix out = {centers, 2, 2, 4};
  assert(cml_kmeans_get_centers(model, &out) == CML_SUCCESS);
  CMLVector sample = {samples, 2, 2};
  size_t left, right;
  assert(cml_kmeans_predict(model, &sample, &left) == CML_SUCCESS);
  sample.data = samples + 4;
  assert(cml_kmeans_predict(model, &sample, &right) == CML_SUCCESS);
  assert(left != right && centers[2 * left] == -3.0 &&
         centers[2 * right] == 4.0);
  assert(cml_kmeans_fit(model, &data, 5, 50, 1e-10, &result) == CML_SUCCESS);
  out.data = repeat;
  assert(cml_kmeans_get_centers(model, &out) == CML_SUCCESS);
  for (size_t i = 0; i < 4; ++i)
    assert(repeat[i] == centers[i]);
  samples[6] = INFINITY;
  assert(cml_kmeans_fit(model, &data, 5, 50, 1e-10, &result) ==
         CML_ERROR_NUMERICAL);
  assert(cml_kmeans_predict(model, &sample, &right) == CML_SUCCESS);
  cml_kmeans_destroy(model);
}

static void rectangular_multiplication(void) {
  const size_t rows = 67, inner = 73, cols = 111;
  CMLMatrix *a = cml_matrix_create(rows, inner),
            *b = cml_matrix_create(inner, cols);
  CMLMatrix *out = cml_matrix_create(rows, cols);
  assert(a != NULL && b != NULL && out != NULL);
  for (size_t i = 0; i < a->capacity; ++i)
    a->data[i] = (double)(i % 13) - 6.0;
  for (size_t i = 0; i < b->capacity; ++i)
    b->data[i] = (double)(i % 17) - 8.0;
  assert(cml_matrix_matmul(a, b, out) == CML_SUCCESS);
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j) {
      double expected = 0.0;
      for (size_t k = 0; k < inner; ++k)
        expected += a->data[i * inner + k] * b->data[k * cols + j];
      assert(out->data[i * cols + j] == expected);
    }
  cml_matrix_destroy(out);
  cml_matrix_destroy(b);
  cml_matrix_destroy(a);
}

int main(void) {
  neighbors();
  clusters();
  rectangular_multiplication();
  puts("Neighbors and blocked multiplication tests passed.");
}
