#include "cml/linear_regression.h"
#include "cml/memory.h"

#include <math.h>

struct CMLLinearRegression {
  CMLVector *weights;
  double intercept;
  bool trained;
};

CMLLinearRegression *cml_linear_regression_create(size_t num_features) {
  if (num_features != 1) {
    return NULL;
  }

  CMLLinearRegression *model = cml_malloc(sizeof(*model));
  if (model == NULL) {
    return NULL;
  }

  model->weights = cml_vector_create(num_features);
  if (model->weights == NULL) {
    cml_free(model);
    return NULL;
  }

  model->intercept = 0.0;
  model->trained = false;
  return model;
}

void cml_linear_regression_destroy(CMLLinearRegression *model) {
  if (model != NULL) {
    cml_vector_destroy(model->weights);
    cml_free(model);
  }
}

bool cml_linear_regression_is_trained(const CMLLinearRegression *model) {
  return model != NULL && model->trained;
}

static double scaled_origin(double min, double max, int exponent) {
  double low = scalbn(min, -exponent);
  double high = scalbn(max, -exponent);
  /* Shift a tight cluster so its mean retains the small offsets. Keep zero
   * as the origin for broad ranges to avoid losing values near zero.
   */
  if (high - low <= fmin(fabs(low), fabs(high))) {
    return low;
  }
  return 0.0;
}

CMLStatus cml_linear_regression_fit(CMLLinearRegression *model,
                                    const CMLMatrix *features,
                                    const CMLVector *targets) {
  if (model == NULL || features == NULL || targets == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t count = features->rows;
  if (count == 0 || features->cols != model->weights->size ||
      targets->size != count || features->capacity < count ||
      targets->capacity < count) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (features->data == NULL || targets->data == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  const double *x = features->data;
  const double *y = targets->data;
  double min_x = x[0], max_x = x[0];
  double min_y = y[0], max_y = y[0];
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(x[i]) || !isfinite(y[i])) {
      return CML_ERROR_NUMERICAL;
    }
    min_x = fmin(min_x, x[i]);
    max_x = fmax(max_x, x[i]);
    min_y = fmin(min_y, y[i]);
    max_y = fmax(max_y, y[i]);
  }

  /* Scale by powers of two before forming the centered products. */
  int x_exponent, y_exponent;
  frexp(fmax(fabs(min_x), fabs(max_x)), &x_exponent);
  frexp(fmax(fabs(min_y), fabs(max_y)), &y_exponent);
  double x_origin = scaled_origin(min_x, max_x, x_exponent);
  double y_origin = scaled_origin(min_y, max_y, y_exponent);
  double x_mean_offset = 0.0;
  double y_mean_offset = 0.0;
  for (size_t i = 0; i < count; ++i) {
    x_mean_offset += scalbn(x[i], -x_exponent) - x_origin;
    y_mean_offset += scalbn(y[i], -y_exponent) - y_origin;
  }
  x_mean_offset /= (double)count;
  y_mean_offset /= (double)count;

  double variance_sum = 0.0;
  double covariance_sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    double dx = (scalbn(x[i], -x_exponent) - x_origin) - x_mean_offset;
    double dy = (scalbn(y[i], -y_exponent) - y_origin) - y_mean_offset;
    variance_sum = fma(dx, dx, variance_sum);
    covariance_sum = fma(dx, dy, covariance_sum);
  }

  double scaled_weight = 0.0;
  if (variance_sum > 0.0) {
    scaled_weight = covariance_sum / variance_sum;
  }
  double weight = scalbn(scaled_weight, y_exponent - x_exponent);
  double scaled_intercept = fma(-scaled_weight, x_origin, y_origin) +
                            fma(-scaled_weight, x_mean_offset, y_mean_offset);
  double intercept = scalbn(scaled_intercept, y_exponent);
  if (!isfinite(weight) || !isfinite(intercept) ||
      (weight == 0.0 && scaled_weight != 0.0) ||
      (intercept == 0.0 && scaled_intercept != 0.0)) {
    return CML_ERROR_NUMERICAL;
  }

  model->weights->data[0] = weight;
  model->intercept = intercept;
  model->trained = true;
  return CML_SUCCESS;
}

CMLStatus cml_linear_regression_predict(const CMLLinearRegression *model,
                                        const CMLVector *features,
                                        double *out) {
  if (!cml_linear_regression_is_trained(model) || features == NULL ||
      out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (features->size != model->weights->size ||
      features->capacity < features->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (features->data == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (!isfinite(features->data[0])) {
    return CML_ERROR_NUMERICAL;
  }

  double prediction =
      fma(model->weights->data[0], features->data[0], model->intercept);
  if (!isfinite(prediction)) {
    return CML_ERROR_NUMERICAL;
  }
  *out = prediction;
  return CML_SUCCESS;
}

CMLStatus cml_linear_regression_get_parameters(const CMLLinearRegression *model,
                                               CMLVector *weights,
                                               double *intercept) {
  if (!cml_linear_regression_is_trained(model) || weights == NULL ||
      intercept == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (weights->size != model->weights->size ||
      weights->capacity < weights->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (weights->data == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  weights->data[0] = model->weights->data[0];
  *intercept = model->intercept;
  return CML_SUCCESS;
}
