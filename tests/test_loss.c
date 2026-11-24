#include "cml.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

static void test_metrics(void) {
  double targets[] = {3.0, 5.0, 7.0, 9.0};
  double predictions[] = {2.0, 6.0, 8.0, 10.0};
  CMLVector actual = {targets, 4, 4};
  CMLVector predicted = {predictions, 4, 4};
  double loss;

  assert(cml_loss_sse(&actual, &predicted, &loss) == CML_SUCCESS &&
         loss == 4.0);
  assert(cml_loss_mse(&actual, &predicted, &loss) == CML_SUCCESS &&
         loss == 1.0);
  assert(cml_loss_rmse(&actual, &predicted, &loss) == CML_SUCCESS &&
         loss == 1.0);
  assert(cml_loss_mae(&actual, &predicted, &loss) == CML_SUCCESS &&
         loss == 1.0);

  assert(cml_loss_mse(&actual, &actual, &loss) == CML_SUCCESS && loss == 0.0);
  assert(cml_loss_rmse(&actual, &actual, &loss) == CML_SUCCESS && loss == 0.0);

  actual.size = predicted.size = 1;
  predictions[0] = 6.0;
  assert(cml_loss_sse(&actual, &predicted, &loss) == CML_SUCCESS &&
         loss == 9.0);
  assert(cml_loss_mse(&actual, &predicted, &loss) == CML_SUCCESS &&
         loss == 9.0);
  assert(cml_loss_rmse(&actual, &predicted, &loss) == CML_SUCCESS &&
         loss == 3.0);
  assert(cml_loss_mae(&actual, &predicted, &loss) == CML_SUCCESS &&
         loss == 3.0);
}

static void test_errors(void) {
  double data[] = {1.0, NAN};
  CMLVector actual = {data, 1, 2};
  CMLVector predicted = {data, 2, 2};
  CMLVector empty = {0};
  double loss = -1.0;

  assert(cml_loss_mse(NULL, &predicted, &loss) == CML_ERROR_INVALID_ARGUMENT);
  assert(cml_loss_mse(&actual, &actual, NULL) == CML_ERROR_INVALID_ARGUMENT);
  assert(cml_loss_sse(&empty, &empty, &loss) == CML_ERROR_INVALID_DIMENSION);
  assert(cml_loss_rmse(&actual, &predicted, &loss) ==
         CML_ERROR_INVALID_DIMENSION);
  actual.size = 2;
  assert(cml_loss_mae(&actual, &predicted, &loss) == CML_ERROR_NUMERICAL);
  data[1] = INFINITY;
  assert(cml_loss_mse(&actual, &predicted, &loss) == CML_ERROR_NUMERICAL);
  assert(loss == -1.0);
}

static void test_numerical_scale(void) {
  double targets[] = {0.0, 0.0, 0.0, 0.0};
  double predictions[] = {0x1p512, 0.0, 0.0, 0.0};
  CMLVector actual = {targets, 4, 4};
  CMLVector predicted = {predictions, 4, 4};
  double loss = -1.0;

  assert(cml_loss_sse(&actual, &predicted, &loss) == CML_ERROR_NUMERICAL);
  assert(loss == -1.0);
  assert(cml_loss_mse(&actual, &predicted, &loss) == CML_SUCCESS);
  assert(loss == 0x1p1022);
  assert(cml_loss_rmse(&actual, &predicted, &loss) == CML_SUCCESS);
  assert(loss == 0x1p511);

  for (size_t i = 0; i < predicted.size; ++i) {
    predictions[i] = 0x1p-538;
  }
  assert(cml_loss_sse(&actual, &predicted, &loss) == CML_SUCCESS);
  assert(loss == DBL_TRUE_MIN);
  assert(cml_loss_rmse(&actual, &predicted, &loss) == CML_SUCCESS);
  assert(loss == 0x1p-538);

  targets[0] = -DBL_MAX;
  predictions[0] = DBL_MAX;
  assert(cml_loss_mae(&actual, &predicted, &loss) == CML_SUCCESS);
  assert(loss == DBL_MAX / 2.0);
  assert(cml_loss_rmse(&actual, &predicted, &loss) == CML_SUCCESS);
  assert(loss == DBL_MAX);
}

int main(void) {
  test_metrics();
  test_errors();
  test_numerical_scale();
  puts("Loss tests passed.");
  return 0;
}
