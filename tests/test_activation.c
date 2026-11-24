#include "cml.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void test_scalar_sigmoid(void) {
  const double logits[] = {-10.0, -2.0, -1.0, 0.0, 1.0, 2.0, 10.0};
  const double probabilities[] = {
      0.000045397868702434, 0.119202922022118, 0.268941421369995, 0.5,
      0.731058578630005,    0.880797077977882, 0.999954602131298};
  for (size_t i = 0; i < sizeof(logits) / sizeof(logits[0]); ++i) {
    assert(fabs(cml_sigmoid(logits[i]) - probabilities[i]) < 1e-14);
  }
  assert(cml_sigmoid(-0.0) == 0.5);
  assert(fabs(cml_sigmoid(1e-8) - (0.5 + 2.5e-9)) < 1e-15);
  assert(fabs(cml_sigmoid(-1e-8) - (0.5 - 2.5e-9)) < 1e-15);

  /* The reciprocal formula with exp(710) loses this representable tail. */
  double tail = cml_sigmoid(-710.0);
  assert(tail > 0.0 && tail < DBL_MIN);
  assert(fabs(tail / 4.47628622567513e-309 - 1.0) < 1e-12);
  assert(cml_sigmoid(-DBL_MAX) == 0.0);
  assert(cml_sigmoid(DBL_MAX) == 1.0);
  assert(cml_sigmoid(-INFINITY) == 0.0);
  assert(cml_sigmoid(INFINITY) == 1.0);
  assert(isnan(cml_sigmoid(NAN)));
}

static void test_vector_sigmoid(void) {
  double logits[] = {-INFINITY, -2.0, 0.0, 2.0, INFINITY, NAN};
  double output[6];
  CMLVector input = {logits, 6, 6};
  CMLVector out = {output, 6, 6};

  assert(cml_vector_sigmoid(&input, &out) == CML_SUCCESS);
  assert(logits[1] == -2.0 && logits[2] == 0.0 && logits[3] == 2.0);
  assert(output[0] == 0.0 && output[2] == 0.5 && output[4] == 1.0);
  assert(fabs(output[1] - 0.119202922022118) < 1e-14);
  assert(fabs(output[3] - 0.880797077977882) < 1e-14);
  assert(isnan(output[5]));

  assert(cml_vector_sigmoid_inplace(&input) == CML_SUCCESS);
  for (size_t i = 0; i < 5; ++i) {
    assert(logits[i] == output[i]);
  }
  assert(isnan(logits[5]));
}

static void test_vector_shape(void) {
  double value = 0.0;
  double result = -1.0;
  CMLVector input = {&value, 1, 1};
  CMLVector out = {&result, 0, 1};
  CMLVector empty = {0};
  assert(cml_vector_sigmoid_inplace(&empty) == CML_SUCCESS);
  assert(cml_vector_sigmoid(NULL, &out) == CML_ERROR_INVALID_ARGUMENT);
  assert(cml_vector_sigmoid(&input, NULL) == CML_ERROR_INVALID_ARGUMENT);
  assert(cml_vector_sigmoid(&input, &out) == CML_ERROR_INVALID_DIMENSION);
  out.size = 1;
  input.data = NULL;
  assert(cml_vector_sigmoid(&input, &out) == CML_ERROR_INVALID_ARGUMENT);
  input.data = &value;
  input.size = input.capacity = out.size = out.capacity = SIZE_MAX;
  assert(cml_vector_sigmoid(&input, &out) == CML_ERROR_OVERFLOW);
  assert(result == -1.0);
}

int main(void) {
  test_scalar_sigmoid();
  test_vector_sigmoid();
  test_vector_shape();
  puts("Activation tests passed.");
  return 0;
}
