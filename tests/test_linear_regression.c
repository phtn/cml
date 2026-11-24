#include "cml.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void assert_parameters(const CMLLinearRegression *model,
                              double expected_weight,
                              double expected_intercept) {
  double weight, intercept;
  CMLVector weights = {&weight, 1, 1};
  assert(cml_linear_regression_get_parameters(model, &weights, &intercept) ==
         CML_SUCCESS);
  assert(fabs(weight - expected_weight) <= 1e-12 * fabs(expected_weight));
  assert(fabs(intercept - expected_intercept) <=
         1e-12 * fabs(expected_intercept));
}

static void test_fit_and_refit(void) {
  double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  double y[] = {3.0, 5.0, 7.0, 9.0, 11.0};
  CMLMatrix features = {x, 5, 1, 5};
  CMLVector targets = {y, 5, 5};
  CMLLinearRegression *model = cml_linear_regression_create(1);
  assert(model != NULL);
  assert(!cml_linear_regression_is_trained(model));
  assert(cml_linear_regression_fit(model, &features, &targets) == CML_SUCCESS);
  assert(cml_linear_regression_is_trained(model));
  assert_parameters(model, 2.0, 1.0);

  double samples[] = {1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0};
  const double expected[] = {3.0, 5.0, 7.0, 9.0, 11.0, 21.0, 41.0};
  for (size_t i = 0; i < sizeof(samples) / sizeof(samples[0]); ++i) {
    CMLVector sample = {&samples[i], 1, 1};
    double prediction;
    assert(cml_linear_regression_predict(model, &sample, &prediction) ==
           CML_SUCCESS);
    assert(fabs(prediction - expected[i]) <= 1e-12);
  }

  for (size_t i = 0; i < targets.size; ++i) {
    assert(x[i] == (double)(i + 1));
    assert(y[i] == expected[i]);
    y[i] = 12.0 - 2.0 * x[i];
  }
  assert(cml_linear_regression_fit(model, &features, &targets) == CML_SUCCESS);
  assert_parameters(model, -2.0, 12.0);

  /* The fitted line minimizes squared error even when the points are noisy. */
  double noisy[] = {1.0, 4.0, 3.0, 6.0, 6.0};
  targets.data = noisy;
  assert(cml_linear_regression_fit(model, &features, &targets) == CML_SUCCESS);
  assert_parameters(model, 1.2, 0.4);

  cml_linear_regression_destroy(model);
}

static void test_constant_data(void) {
  double x[] = {1.0, 2.0, 3.0, 4.0};
  double y[] = {7.0, 7.0, 7.0, 7.0};
  CMLMatrix features = {x, 4, 1, 4};
  CMLVector targets = {y, 4, 4};
  CMLLinearRegression *model = cml_linear_regression_create(1);
  assert(model != NULL);

  assert(cml_linear_regression_fit(model, &features, &targets) == CML_SUCCESS);
  assert_parameters(model, 0.0, 7.0);

  for (size_t i = 0; i < targets.size; ++i) {
    y[i] = x[i];
    x[i] = 5.0;
  }
  assert(cml_linear_regression_fit(model, &features, &targets) == CML_SUCCESS);
  assert_parameters(model, 0.0, 2.5);

  features.rows = 1;
  targets.size = 1;
  assert(cml_linear_regression_fit(model, &features, &targets) == CML_SUCCESS);
  assert_parameters(model, 0.0, 1.0);
  cml_linear_regression_destroy(model);
}

static void test_errors(void) {
  double x[] = {1.0, 2.0};
  double y[] = {3.0, 5.0};
  CMLMatrix features = {x, 2, 1, 2};
  CMLVector targets = {y, 2, 2};
  CMLVector sample = {x, 1, 2};
  CMLVector empty = {0};
  CMLMatrix empty_matrix = {0};
  double value = -1.0;

  assert(cml_linear_regression_create(0) == NULL);
  assert(cml_linear_regression_create(2) == NULL);
  assert(!cml_linear_regression_is_trained(NULL));
  cml_linear_regression_destroy(NULL);

  CMLLinearRegression *model = cml_linear_regression_create(1);
  assert(model != NULL);
  assert(cml_linear_regression_predict(model, &sample, &value) ==
         CML_ERROR_INVALID_ARGUMENT);
  assert(cml_linear_regression_get_parameters(model, &sample, &value) ==
         CML_ERROR_INVALID_ARGUMENT);
  assert(value == -1.0 && x[0] == 1.0);
  assert(cml_linear_regression_fit(model, &empty_matrix, &empty) ==
         CML_ERROR_INVALID_DIMENSION);
  assert(cml_linear_regression_fit(model, &features, &sample) ==
         CML_ERROR_INVALID_DIMENSION);
  features.cols = 2;
  assert(cml_linear_regression_fit(model, &features, &targets) ==
         CML_ERROR_INVALID_DIMENSION);
  features.cols = 1;
  assert(cml_linear_regression_fit(model, NULL, &targets) ==
         CML_ERROR_INVALID_ARGUMENT);
  assert(!cml_linear_regression_is_trained(model));
  assert(cml_linear_regression_fit(model, &features, &targets) == CML_SUCCESS);

  y[0] = NAN;
  assert(cml_linear_regression_fit(model, &features, &targets) ==
         CML_ERROR_NUMERICAL);
  y[0] = 3.0;
  x[0] = INFINITY;
  assert(cml_linear_regression_fit(model, &features, &targets) ==
         CML_ERROR_NUMERICAL);
  assert(cml_linear_regression_is_trained(model));
  assert_parameters(model, 2.0, 1.0);

  assert(cml_linear_regression_predict(model, &targets, &value) ==
         CML_ERROR_INVALID_DIMENSION);
  assert(cml_linear_regression_predict(model, &sample, &value) ==
         CML_ERROR_NUMERICAL);
  x[0] = DBL_MAX;
  assert(cml_linear_regression_predict(model, &sample, &value) ==
         CML_ERROR_NUMERICAL);
  assert(value == -1.0);

  cml_linear_regression_destroy(model);
}

static void test_numerical_scale(void) {
  double x[] = {0.0, 0.0, 0.0};
  double y[] = {0.0, 0.0, 0.0};
  CMLMatrix features = {x, 3, 1, 3};
  CMLVector targets = {y, 3, 3};
  CMLLinearRegression *model = cml_linear_regression_create(1);
  assert(model != NULL);

  const double scales[] = {0x1p-600, 0x1p600};
  for (size_t i = 0; i < sizeof(scales) / sizeof(scales[0]); ++i) {
    for (size_t j = 0; j < targets.size; ++j) {
      x[j] = (double)(j + 1) * scales[i];
      y[j] = 2.0 * x[j] + scales[i];
    }
    assert(cml_linear_regression_fit(model, &features, &targets) ==
           CML_SUCCESS);
    assert_parameters(model, 2.0, scales[i]);
  }

  const double offset = 0x1p53;
  for (size_t i = 0; i < targets.size; ++i) {
    x[i] = offset + 2.0 * (double)i;
    y[i] = 4.0 * (double)i;
  }
  assert(cml_linear_regression_fit(model, &features, &targets) == CML_SUCCESS);
  assert_parameters(model, 2.0, -2.0 * offset);

  x[0] = -DBL_MIN;
  x[1] = DBL_MIN;
  x[2] = 0.0;
  y[0] = -DBL_MAX;
  y[1] = DBL_MAX;
  y[2] = 0.0;
  assert(cml_linear_regression_fit(model, &features, &targets) ==
         CML_ERROR_NUMERICAL);
  assert_parameters(model, 2.0, -2.0 * offset);

  x[0] = x[1] = x[2] = 5.0;
  y[2] = 3.0;
  assert(cml_linear_regression_fit(model, &features, &targets) == CML_SUCCESS);
  assert_parameters(model, 0.0, 1.0);
  cml_linear_regression_destroy(model);
}

static void test_simple_gradient(void) {
  double x[] = {1.0, 2.0, 3.0};
  double y[] = {3.0, 5.0, 7.0};
  double theta[] = {0.0, 0.0};
  double gradient[2];
  CMLMatrix features = {x, 3, 1, 3};
  CMLVector targets = {y, 3, 3};
  CMLVector parameters = {theta, 2, 2};
  CMLVector out = {gradient, 2, 2};

  assert(cml_linear_regression_mse_gradient(&features, &targets, &parameters,
                                            &out) == CML_SUCCESS);
  assert(fabs(gradient[0] - (-68.0 / 3.0)) < 1e-12);
  assert(fabs(gradient[1] - (-10.0)) < 1e-12);
  assert(theta[0] == 0.0 && theta[1] == 0.0);

  double norm;
  assert(cml_vector_norm_l2(&out, &norm) == CML_SUCCESS);
  assert(fabs(norm - hypot(68.0 / 3.0, 10.0)) < 1e-12);

  theta[0] = 2.0;
  theta[1] = 1.0;
  assert(cml_linear_regression_mse_gradient(&features, &targets, &parameters,
                                            &out) == CML_SUCCESS);
  assert(gradient[0] == 0.0 && gradient[1] == 0.0);
  assert(cml_vector_norm_l2(&out, &norm) == CML_SUCCESS && norm == 0.0);

  const double scales[] = {0x1p-600, 0x1p600};
  for (size_t i = 0; i < sizeof(scales) / sizeof(scales[0]); ++i) {
    gradient[0] = 3.0 * scales[i];
    gradient[1] = 4.0 * scales[i];
    assert(cml_vector_norm_l2(&out, &norm) == CML_SUCCESS);
    assert(fabs(norm / scales[i] - 5.0) < 1e-12);
  }
}

static void test_multiple_feature_gradient(void) {
  double x[] = {1.0, 2.0, -1.0, -1.0, 0.0, 2.0, 2.0, -1.0, 1.0};
  double y[] = {0.0, 10.0, 9.0};
  double theta[] = {2.0, -1.0, 3.0, 4.0};
  double gradient[4];
  CMLMatrix features = {x, 3, 3, 9};
  CMLVector targets = {y, 3, 3};
  CMLVector parameters = {theta, 4, 4};
  CMLVector out = {gradient, 4, 4};

  assert(cml_linear_regression_mse_gradient(&features, &targets, &parameters,
                                            &out) == CML_SUCCESS);
  const double expected[] = {6.0, -2.0 / 3.0, -4.0 / 3.0, 4.0 / 3.0};
  for (size_t i = 0; i < out.size; ++i) {
    assert(fabs(gradient[i] - expected[i]) < 1e-12);
  }
}

static void test_gradient_errors(void) {
  double x[] = {1.0, 2.0};
  double y[] = {3.0, 5.0};
  double theta[] = {0.0, 0.0};
  double gradient[] = {42.0, 17.0};
  CMLMatrix features = {x, 2, 1, 2};
  CMLVector targets = {y, 2, 2};
  CMLVector parameters = {theta, 2, 2};
  CMLVector out = {gradient, 2, 2};

  assert(cml_linear_regression_mse_gradient(
             &features, NULL, &parameters, &out) == CML_ERROR_INVALID_ARGUMENT);
  features.rows = 0;
  assert(cml_linear_regression_mse_gradient(&features, &targets, &parameters,
                                            &out) ==
         CML_ERROR_INVALID_DIMENSION);
  features.rows = 2;
  features.cols = SIZE_MAX;
  assert(cml_linear_regression_mse_gradient(&features, &targets, &parameters,
                                            &out) == CML_ERROR_OVERFLOW);
  features.cols = 1;
  out.size = 1;
  assert(cml_linear_regression_mse_gradient(&features, &targets, &parameters,
                                            &out) ==
         CML_ERROR_INVALID_DIMENSION);
  out.size = 2;

  y[1] = NAN;
  assert(cml_linear_regression_mse_gradient(&features, &targets, &parameters,
                                            &out) == CML_ERROR_NUMERICAL);
  y[1] = 5.0;
  theta[0] = INFINITY;
  assert(cml_linear_regression_mse_gradient(&features, &targets, &parameters,
                                            &out) == CML_ERROR_NUMERICAL);
  theta[0] = 0.0;
  x[1] = DBL_MAX;
  assert(cml_linear_regression_mse_gradient(&features, &targets, &parameters,
                                            &out) == CML_ERROR_NUMERICAL);
  assert(gradient[0] == 42.0 && gradient[1] == 17.0);
}

int main(void) {
  test_fit_and_refit();
  test_constant_data();
  test_errors();
  test_numerical_scale();
  test_simple_gradient();
  test_multiple_feature_gradient();
  test_gradient_errors();
  puts("Linear regression tests passed.");
  return 0;
}
