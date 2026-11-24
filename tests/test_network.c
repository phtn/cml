#include "cml.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void forward(void) {
  CMLDenseConfig layers[] = {{2, CML_ACTIVATION_RELU},
                             {1, CML_ACTIVATION_SIGMOID}};
  CMLNetwork *network = cml_network_create(2, layers, 2, 17);
  assert(network != NULL && cml_network_parameter_count(network) == 9);
  CMLNetwork *copy = cml_network_create(2, layers, 2, 17);
  assert(copy != NULL);
  double a[9], b[9];
  CMLVector first = {a, 9, 9}, second = {b, 9, 9};
  assert(cml_network_get_parameters(network, &first) == CML_SUCCESS);
  assert(cml_network_get_parameters(copy, &second) == CML_SUCCESS);
  for (size_t i = 0; i < 9; ++i)
    assert(a[i] == b[i]);
  cml_network_destroy(copy);

  double parameters[] = {1, 2, -1, -2, 0, 0, 2, 3, -4};
  CMLVector weights = {parameters, 9, 9};
  assert(cml_network_set_parameters(network, &weights) == CML_SUCCESS);
  double x[] = {2, 1, -1, -1}, y[] = {0, 0};
  CMLVector input = {x, 2, 2}, output = {y, 1, 2};
  assert(cml_network_predict(network, &input, &output) == CML_SUCCESS);
  assert(fabs(y[0] - 0.9820137900379085) < 1e-14);
  CMLMatrix batch = {x, 2, 2, 4}, predictions = {y, 2, 1, 2};
  assert(cml_network_predict_batch(network, &batch, &predictions) ==
         CML_SUCCESS);
  assert(fabs(y[1] - 0.9933071490757153) < 1e-14);
  x[3] = NAN;
  double before = y[0];
  assert(cml_network_predict_batch(network, &batch, &predictions) ==
         CML_ERROR_NUMERICAL);
  assert(y[0] == before);
  input.size = 1;
  assert(cml_network_predict(network, &input, &output) ==
         CML_ERROR_INVALID_DIMENSION);
  cml_network_destroy(network);
}

static void softmax(void) {
  CMLDenseConfig layer = {3, CML_ACTIVATION_SOFTMAX};
  CMLNetwork *network = cml_network_create(1, &layer, 1, 0);
  assert(network != NULL);
  double coefficients[] = {0, 0, 0, 1000, 1001, 1002};
  CMLVector p = {coefficients, 6, 6};
  assert(cml_network_set_parameters(network, &p) == CML_SUCCESS);
  double value = 0, scores[3];
  CMLVector x = {&value, 1, 1}, y = {scores, 3, 3};
  assert(cml_network_predict(network, &x, &y) == CML_SUCCESS);
  assert(fabs(scores[0] + scores[1] + scores[2] - 1.0) < 1e-14);
  assert(fabs(scores[2] - 0.6652409557748218) < 1e-14);
  cml_network_destroy(network);
}

int main(void) {
  forward();
  softmax();
  puts("Network forward tests passed.");
}
