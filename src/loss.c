#include "cml/loss.h"

#include <float.h>
#include <math.h>

typedef enum {
  LOSS_SSE,
  LOSS_MSE,
  LOSS_RMSE,
  LOSS_MAE,
} LossType;

static CMLStatus calculate_loss(const CMLVector *actual,
                                const CMLVector *predicted, LossType type,
                                double *out) {
  if (actual == NULL || predicted == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  size_t count = actual->size;
  if (count == 0 || predicted->size != count || actual->capacity < count ||
      predicted->capacity < count) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (actual->data == NULL || predicted->data == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  double scale = 0.0;
  for (size_t i = 0; i < count; ++i) {
    double target = actual->data[i];
    double prediction = predicted->data[i];
    if (!isfinite(target) || !isfinite(prediction)) {
      return CML_ERROR_NUMERICAL;
    }
    scale = fmax(scale, fabs(prediction - target));
  }
  if (scale == 0.0) {
    *out = 0.0;
    return CML_SUCCESS;
  }

  /* A difference can overflow even when the averaged loss is representable. */
  bool halve = isinf(scale);
  if (halve) {
    scale = DBL_MAX;
  }

  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    double target = actual->data[i];
    double prediction = predicted->data[i];
    double difference =
        halve ? prediction * 0.5 - target * 0.5 : prediction - target;
    double relative = fabs(difference) / scale;
    sum += type == LOSS_MAE ? relative : relative * relative;
  }
  if (type != LOSS_SSE) {
    sum /= (double)count;
  }

  /* Restore the scale after averaging or square rooting. This keeps the
   * squared losses from overflowing or underflowing during rescaling.
   */
  int exponent;
  double fraction = frexp(scale, &exponent);
  exponent += halve;
  if (type == LOSS_RMSE) {
    sum = sqrt(sum);
  }
  double result = (type == LOSS_SSE || type == LOSS_MSE)
                      ? scalbn(fraction * fraction * sum, 2 * exponent)
                      : scalbn(fraction * sum, exponent);
  if (!isfinite(result)) {
    return CML_ERROR_NUMERICAL;
  }
  *out = result;
  return CML_SUCCESS;
}

CMLStatus cml_loss_sse(const CMLVector *actual, const CMLVector *predicted,
                       double *out) {
  return calculate_loss(actual, predicted, LOSS_SSE, out);
}

CMLStatus cml_loss_mse(const CMLVector *actual, const CMLVector *predicted,
                       double *out) {
  return calculate_loss(actual, predicted, LOSS_MSE, out);
}

CMLStatus cml_loss_rmse(const CMLVector *actual, const CMLVector *predicted,
                        double *out) {
  return calculate_loss(actual, predicted, LOSS_RMSE, out);
}

CMLStatus cml_loss_mae(const CMLVector *actual, const CMLVector *predicted,
                       double *out) {
  return calculate_loss(actual, predicted, LOSS_MAE, out);
}
