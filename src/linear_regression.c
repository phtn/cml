#include "cml/linear_regression.h"
#include "cml/memory.h"
#include "internal.h"

#include <float.h>
#include <string.h>

#include <math.h>
#include <stdint.h>

struct CMLLinearRegression {
  CMLVector *weights;
  double intercept;
  bool trained;
};

CMLLinearRegression *cml_linear_regression_create(size_t num_features) {
  if (num_features == 0) {
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

/* Householder QR on the scaled, augmented design [1 X | y]. */
static CMLStatus fit_multiple(CMLLinearRegression *model,
                              const CMLMatrix *features,
                              const CMLVector *targets) {
  size_t rows = features->rows, cols = features->cols;
  if (rows <= cols)
    return CML_ERROR_INVALID_DIMENSION;
  if (!cml_values_finite(features->data, rows * cols) ||
      !cml_values_finite(targets->data, rows))
    return CML_ERROR_NUMERICAL;
  size_t parameters = cols + 1;
  size_t width = parameters + 1;
  if (rows > SIZE_MAX / sizeof(double) / width)
    return CML_ERROR_OVERFLOW;
  CMLMatrix *design = cml_matrix_create(rows, width);
  CMLVector *scales = cml_vector_create(width);
  CMLVector *solution = cml_vector_create(parameters);
  CMLStatus status = CML_ERROR_OUT_OF_MEMORY;
  if (design == NULL || scales == NULL || solution == NULL)
    goto done;
  scales->data[0] = 1.0;
  for (size_t i = 0; i < rows; ++i) {
    double *row = design->data + i * width;
    row[0] = 1.0;
    memcpy(row + 1, features->data + i * cols, cols * sizeof(double));
    row[parameters] = targets->data[i];
    for (size_t j = 1; j < width; ++j)
      scales->data[j] = fmax(scales->data[j], fabs(row[j]));
  }
  for (size_t j = 1; j < width; ++j) {
    if (scales->data[j] == 0.0)
      scales->data[j] = 1.0;
    for (size_t i = 0; i < rows; ++i)
      design->data[i * width + j] /= scales->data[j];
  }

  status = CML_ERROR_NUMERICAL;
  double tolerance = DBL_EPSILON * (double)rows * sqrt((double)rows);
  for (size_t j = 0; j < parameters; ++j) {
    double norm = 0.0;
    for (size_t i = j; i < rows; ++i)
      norm = hypot(norm, design->data[i * width + j]);
    if (norm <= tolerance)
      goto done;
    double *diagonal = &design->data[j * width + j];
    double replacement = -copysign(norm, *diagonal);
    *diagonal -= replacement;
    double vector_norm = 0.0;
    for (size_t i = j; i < rows; ++i)
      vector_norm = hypot(vector_norm, design->data[i * width + j]);
    for (size_t i = j; i < rows; ++i)
      design->data[i * width + j] /= vector_norm;
    for (size_t k = j + 1; k < width; ++k) {
      double projection = 0.0;
      for (size_t i = j; i < rows; ++i)
        projection += design->data[i * width + j] * design->data[i * width + k];
      for (size_t i = j; i < rows; ++i)
        design->data[i * width + k] -=
            2.0 * projection * design->data[i * width + j];
    }
    *diagonal = replacement;
  }

  for (size_t j = parameters; j-- > 0;) {
    const double *row = design->data + j * width;
    double value = row[parameters];
    for (size_t k = j + 1; k < parameters; ++k)
      value = fma(-row[k], solution->data[k], value);
    solution->data[j] = value / row[j];
  }
  int target_exponent;
  double target_fraction = frexp(scales->data[parameters], &target_exponent);
  for (size_t j = 0; j < parameters; ++j) {
    int feature_exponent;
    double feature_fraction = frexp(scales->data[j], &feature_exponent);
    solution->data[j] =
        scalbn(solution->data[j] * (target_fraction / feature_fraction),
               target_exponent - feature_exponent);
  }
  CMLVector weights = {solution->data + 1, cols, cols};
  status =
      cml_linear_regression_set_parameters(model, &weights, solution->data[0]);
done:
  cml_vector_destroy(solution);
  cml_vector_destroy(scales);
  cml_matrix_destroy(design);
  return status;
}

CMLStatus cml_linear_regression_fit(CMLLinearRegression *model,
                                    const CMLMatrix *features,
                                    const CMLVector *targets) {
  if (model == NULL || features == NULL || targets == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (features->cols != model->weights->size)
    return CML_ERROR_INVALID_DIMENSION;
  CMLStatus status = cml_matrix_storage(features);
  if (status != CML_SUCCESS)
    return status;
  size_t count = features->rows;
  status = cml_vector_storage(targets, count);
  if (status != CML_SUCCESS)
    return status;
  if (features->cols > 1)
    return fit_multiple(model, features, targets);

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
  if (!cml_values_finite(features->data, features->size)) {
    return CML_ERROR_NUMERICAL;
  }

  double prediction = cml_affine(features->data, model->weights->data,
                                 features->size, model->intercept);
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

  memcpy(weights->data, model->weights->data, weights->size * sizeof(double));
  *intercept = model->intercept;
  return CML_SUCCESS;
}

CMLStatus cml_linear_regression_mse_gradient(const CMLMatrix *features,
                                             const CMLVector *targets,
                                             const CMLVector *parameters,
                                             CMLVector *out) {
  if (features == NULL || targets == NULL || parameters == NULL ||
      out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t rows = features->rows;
  size_t cols = features->cols;
  if (rows == 0 || cols == 0 || targets->size != rows) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (cols >= SIZE_MAX / sizeof(double) ||
      rows > SIZE_MAX / sizeof(double) / cols) {
    return CML_ERROR_OVERFLOW;
  }
  size_t parameter_count = cols + 1;
  if (parameters->size != parameter_count || out->size != parameter_count ||
      features->capacity < rows * cols || targets->capacity < rows ||
      parameters->capacity < parameter_count ||
      out->capacity < parameter_count) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (features->data == NULL || targets->data == NULL ||
      parameters->data == NULL || out->data == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  double *gradient = cml_calloc(parameter_count, sizeof(*gradient));
  if (gradient == NULL) {
    return CML_ERROR_OUT_OF_MEMORY;
  }

  const double *weights = parameters->data;
  double intercept = weights[cols];
  double factor = 2.0 / (double)rows;
  CMLStatus status = CML_SUCCESS;
  for (size_t i = 0; i < rows; ++i) {
    const double *row = features->data + i * cols;
    double prediction = intercept;
    for (size_t j = 0; j < cols; ++j) {
      prediction = fma(row[j], weights[j], prediction);
    }
    double target = targets->data[i];
    if (!isfinite(prediction) || !isfinite(target)) {
      status = CML_ERROR_NUMERICAL;
      break;
    }
    double residual = prediction - target;
    /* Average before accumulating the products. Scale the terms separately
     * when the subtraction overflows but the averaged difference may fit.
     */
    double change = isfinite(residual) ? factor * residual
                                       : factor * prediction - factor * target;
    for (size_t j = 0; j < cols; ++j) {
      gradient[j] = fma(row[j], change, gradient[j]);
    }
    gradient[cols] += change;
  }
  if (status == CML_SUCCESS) {
    for (size_t j = 0; j < parameter_count; ++j) {
      if (!isfinite(gradient[j])) {
        status = CML_ERROR_NUMERICAL;
        break;
      }
    }
  }
  if (status == CML_SUCCESS) {
    cml_mem_copy(out->data, gradient, parameter_count * sizeof(*gradient));
  }
  cml_free(gradient);
  return status;
}

size_t cml_linear_regression_feature_count(const CMLLinearRegression *model) {
  return model == NULL ? 0 : model->weights->size;
}

CMLStatus cml_linear_regression_set_parameters(CMLLinearRegression *model,
                                               const CMLVector *weights,
                                               double intercept) {
  if (model == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_vector_storage(weights, model->weights->size);
  if (status != CML_SUCCESS)
    return status;
  if (!isfinite(intercept) || !cml_values_finite(weights->data, weights->size))
    return CML_ERROR_NUMERICAL;
  memcpy(model->weights->data, weights->data, weights->size * sizeof(double));
  model->intercept = intercept;
  model->trained = true;
  return CML_SUCCESS;
}

CMLStatus cml_linear_regression_predict_batch(const CMLLinearRegression *model,
                                              const CMLMatrix *features,
                                              CMLVector *out) {
  if (!cml_linear_regression_is_trained(model))
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_matrix_storage(features);
  if (status != CML_SUCCESS)
    return status;
  if (features->cols != model->weights->size)
    return CML_ERROR_INVALID_DIMENSION;
  status = cml_vector_storage(out, features->rows);
  if (status != CML_SUCCESS)
    return status;

  CMLVector *predictions = cml_vector_create(features->rows);
  if (predictions == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  for (size_t i = 0; i < features->rows && status == CML_SUCCESS; ++i) {
    CMLVector row = {features->data + i * features->cols, features->cols,
                     features->cols};
    status = cml_linear_regression_predict(model, &row, &predictions->data[i]);
  }
  if (status == CML_SUCCESS)
    memcpy(out->data, predictions->data, out->size * sizeof(double));
  cml_vector_destroy(predictions);
  return status;
}

typedef struct {
  const CMLMatrix *features;
  const CMLVector *targets;
} RegressionData;

static CMLStatus regression_batch(const CMLVector *p, const size_t *indices,
                                  size_t count, void *context, double *loss,
                                  CMLVector *gradient) {
  const RegressionData *data = context;
  size_t cols = data->features->cols;
  double square_mean = 0.0;
  if (gradient != NULL)
    cml_vector_fill(gradient, 0.0);
  for (size_t i = 0; i < count; ++i) {
    size_t row_index = indices == NULL ? i : indices[i];
    const double *row = data->features->data + row_index * cols;
    double error = cml_affine(row, p->data, cols, p->data[cols]) -
                   data->targets->data[row_index];
    if (!isfinite(error))
      return CML_ERROR_NUMERICAL;
    if (loss != NULL)
      square_mean = fma(error / (double)count, error, square_mean);
    if (gradient != NULL) {
      double derivative = error * (2.0 / (double)count);
      for (size_t j = 0; j < cols; ++j)
        gradient->data[j] = fma(derivative, row[j], gradient->data[j]);
      gradient->data[cols] += derivative;
    }
  }
  if (loss != NULL)
    *loss = square_mean;
  return CML_SUCCESS;
}

static CMLStatus validate_regression_data(const RegressionData *data,
                                          size_t features) {
  CMLStatus status = cml_matrix_storage(data->features);
  if (status != CML_SUCCESS)
    return status;
  if (data->features->cols != features)
    return CML_ERROR_INVALID_DIMENSION;
  return cml_vector_storage(data->targets, data->features->rows);
}

CMLStatus cml_linear_regression_fit_iterative(
    CMLLinearRegression *model, const CMLMatrix *features,
    const CMLVector *targets, const CMLMatrix *validation_features,
    const CMLVector *validation_targets, const CMLTrainingOptions *options,
    CMLTrainingResult *result) {
  if (model == NULL ||
      (validation_features == NULL) != (validation_targets == NULL))
    return CML_ERROR_INVALID_ARGUMENT;
  size_t cols = model->weights->size;
  RegressionData training = {features, targets};
  RegressionData validation = {validation_features, validation_targets};
  CMLStatus status = validate_regression_data(&training, cols);
  if (status == CML_SUCCESS && validation_features != NULL)
    status = validate_regression_data(&validation, cols);
  if (status != CML_SUCCESS)
    return status;

  CMLVector *parameters = cml_vector_create(cols + 1);
  if (parameters == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  memcpy(parameters->data, model->weights->data, cols * sizeof(double));
  parameters->data[cols] = model->intercept;
  status =
      cml_train(parameters, regression_batch, features->rows, &training,
                validation_features == NULL ? 0 : validation_features->rows,
                &validation, options, result);
  if (status == CML_SUCCESS) {
    CMLVector weights = {parameters->data, cols, cols};
    status = cml_linear_regression_set_parameters(model, &weights,
                                                  parameters->data[cols]);
  }
  cml_vector_destroy(parameters);
  return status;
}
