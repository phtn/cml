#include "cml/activation.h"
#include "internal.h"

#include <math.h>
#include <stdint.h>

double cml_sigmoid(double value) {
  if (value >= 0.0) {
    return 1.0 / (1.0 + exp(-value));
  }
  double exponential = exp(value);
  return exponential / (1.0 + exponential);
}

CMLStatus cml_vector_sigmoid(const CMLVector *input, CMLVector *out) {
  if (input == NULL || out == NULL) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  if (input->size != out->size || input->capacity < input->size ||
      out->capacity < out->size) {
    return CML_ERROR_INVALID_DIMENSION;
  }
  if (input->size > SIZE_MAX / sizeof(double)) {
    return CML_ERROR_OVERFLOW;
  }
  if (input->size > 0 && (input->data == NULL || out->data == NULL)) {
    return CML_ERROR_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < input->size; ++i) {
    out->data[i] = cml_sigmoid(input->data[i]);
  }
  return CML_SUCCESS;
}

CMLStatus cml_vector_sigmoid_inplace(CMLVector *vec) {
  return cml_vector_sigmoid(vec, vec);
}

CMLStatus cml_softmax(const CMLVector *logits, CMLVector *out) {
  if (logits == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t size = logits->size;
  if (size == 0)
    return CML_ERROR_INVALID_DIMENSION;
  CMLStatus status = cml_vector_storage(logits, size);
  if (status == CML_SUCCESS)
    status = cml_vector_storage(out, size);
  if (status != CML_SUCCESS)
    return status;
  if (!cml_values_finite(logits->data, size))
    return CML_ERROR_NUMERICAL;
  double maximum = logits->data[0];
  for (size_t i = 1; i < size; ++i)
    maximum = fmax(maximum, logits->data[i]);
  double sum = 0.0;
  for (size_t i = 0; i < size; ++i) {
    out->data[i] = exp(logits->data[i] - maximum);
    sum += out->data[i];
  }
  for (size_t i = 0; i < size; ++i)
    out->data[i] /= sum;
  return CML_SUCCESS;
}
