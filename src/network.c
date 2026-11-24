#include "cml/activation.h"
#include "cml/memory.h"
#include "internal.h"
#include "network_internal.h"

#include <string.h>

CMLNetwork *cml_network_create(size_t inputs, const CMLDenseConfig *layers,
                               size_t count, uint64_t seed) {
  if (inputs == 0 || layers == NULL || count == 0 ||
      count > SIZE_MAX / sizeof(CMLDenseLayer))
    return NULL;
  CMLNetwork *network = cml_calloc(1, sizeof(*network));
  if (network == NULL)
    return NULL;
  network->inputs = inputs;
  network->layer_count = count;
  network->layers = cml_calloc(count, sizeof(*network->layers));
  if (network->layers == NULL)
    goto fail;
  size_t parameters = 0, units = 0, previous = inputs;
  const size_t max_values = SIZE_MAX / sizeof(double);
  for (size_t i = 0; i < count; ++i) {
    CMLDenseConfig layer = layers[i];
    if (layer.units == 0 || layer.activation < CML_ACTIVATION_NONE ||
        layer.activation > CML_ACTIVATION_SOFTMAX || previous >= max_values ||
        layer.units > (max_values - parameters) / (previous + 1))
      goto fail;
    network->layers[i] = (CMLDenseLayer){previous, layer, parameters, units};
    parameters += layer.units * (previous + 1);
    units += layer.units;
    previous = layer.units;
  }
  network->unit_count = units;
  network->parameters = cml_vector_create(parameters);
  if (network->parameters == NULL)
    goto fail;
  for (size_t i = 0; i < count; ++i) {
    CMLDenseLayer layer = network->layers[i];
    size_t weights = layer.inputs * layer.config.units;
    double divisor = (double)layer.inputs;
    if (layer.config.activation != CML_ACTIVATION_RELU)
      divisor += (double)layer.config.units;
    double bound = sqrt(6.0 / divisor);
    for (size_t j = 0; j < weights; ++j)
      network->parameters->data[layer.parameter_offset + j] =
          (2.0 * cml_random_unit(&seed) - 1.0) * bound;
  }
  return network;
fail:
  cml_network_destroy(network);
  return NULL;
}

void cml_network_destroy(CMLNetwork *network) {
  if (network != NULL) {
    cml_vector_destroy(network->parameters);
    cml_free(network->layers);
    cml_free(network);
  }
}

size_t cml_network_input_size(const CMLNetwork *network) {
  return network == NULL ? 0 : network->inputs;
}

size_t cml_network_output_size(const CMLNetwork *network) {
  return network == NULL
             ? 0
             : network->layers[network->layer_count - 1].config.units;
}

size_t cml_network_layer_count(const CMLNetwork *network) {
  return network == NULL ? 0 : network->layer_count;
}

size_t cml_network_parameter_count(const CMLNetwork *network) {
  return network == NULL ? 0 : network->parameters->size;
}

CMLStatus cml_network_layer_config(const CMLNetwork *network, size_t layer,
                                   CMLDenseConfig *out) {
  if (network == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (layer >= network->layer_count)
    return CML_ERROR_INDEX_OUT_OF_BOUNDS;
  *out = network->layers[layer].config;
  return CML_SUCCESS;
}

CMLStatus cml_network_get_parameters(const CMLNetwork *network,
                                     CMLVector *out) {
  if (network == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_vector_storage(out, network->parameters->size);
  if (status == CML_SUCCESS)
    memcpy(out->data, network->parameters->data, out->size * sizeof(double));
  return status;
}

CMLStatus cml_network_set_parameters(CMLNetwork *network,
                                     const CMLVector *values) {
  if (network == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_vector_storage(values, network->parameters->size);
  if (status != CML_SUCCESS)
    return status;
  if (!cml_values_finite(values->data, values->size))
    return CML_ERROR_NUMERICAL;
  memcpy(network->parameters->data, values->data,
         values->size * sizeof(double));
  return CML_SUCCESS;
}

static double activate(double value, CMLActivation activation) {
  switch (activation) {
  case CML_ACTIVATION_RELU:
    return fmax(value, 0.0);
  case CML_ACTIVATION_SIGMOID:
    return cml_sigmoid(value);
  case CML_ACTIVATION_TANH:
    return tanh(value);
  case CML_ACTIVATION_NONE:
  case CML_ACTIVATION_SOFTMAX:
    return value;
  }
  return NAN;
}

CMLStatus cml_network_forward(const CMLNetwork *network, const double *params,
                              const double *input, double *values,
                              double *logits) {
  if (!cml_values_finite(input, network->inputs))
    return CML_ERROR_NUMERICAL;
  for (size_t i = 0; i < network->layer_count; ++i) {
    const CMLDenseLayer *layer = &network->layers[i];
    size_t outputs = layer->config.units;
    const double *weights = params + layer->parameter_offset;
    const double *bias = weights + outputs * layer->inputs;
    double *result = values + layer->unit_offset;
    for (size_t j = 0; j < outputs; ++j) {
      double value = cml_affine(input, weights + j * layer->inputs,
                                layer->inputs, bias[j]);
      if (!isfinite(value))
        return CML_ERROR_NUMERICAL;
      if (logits != NULL)
        logits[layer->unit_offset + j] = value;
      result[j] = activate(value, layer->config.activation);
    }
    if (layer->config.activation == CML_ACTIVATION_SOFTMAX) {
      CMLVector row = {result, outputs, outputs};
      CMLStatus status = cml_softmax(&row, &row);
      if (status != CML_SUCCESS)
        return status;
    }
    input = result;
  }
  return CML_SUCCESS;
}

CMLStatus cml_network_predict(const CMLNetwork *network, const CMLVector *input,
                              CMLVector *out) {
  if (network == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_vector_storage(input, network->inputs);
  if (status == CML_SUCCESS)
    status = cml_vector_storage(out, cml_network_output_size(network));
  if (status != CML_SUCCESS)
    return status;
  CMLVector *values = cml_vector_create(network->unit_count);
  if (values == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  status = cml_network_forward(network, network->parameters->data, input->data,
                               values->data, NULL);
  if (status == CML_SUCCESS)
    memcpy(out->data, values->data + network->unit_count - out->size,
           out->size * sizeof(double));
  cml_vector_destroy(values);
  return status;
}

CMLStatus cml_network_predict_batch(const CMLNetwork *network,
                                    const CMLMatrix *inputs, CMLMatrix *out) {
  if (network == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_matrix_storage(inputs);
  if (status == CML_SUCCESS)
    status = cml_matrix_storage(out);
  if (status != CML_SUCCESS)
    return status;
  if (inputs->cols != network->inputs || out->rows != inputs->rows ||
      out->cols != cml_network_output_size(network))
    return CML_ERROR_INVALID_DIMENSION;
  CMLVector *values = cml_vector_create(network->unit_count);
  CMLMatrix *results = cml_matrix_create(out->rows, out->cols);
  if (values == NULL || results == NULL) {
    status = CML_ERROR_OUT_OF_MEMORY;
  } else {
    for (size_t i = 0; i < inputs->rows; ++i) {
      status = cml_network_forward(network, network->parameters->data,
                                   inputs->data + i * inputs->cols,
                                   values->data, NULL);
      if (status != CML_SUCCESS)
        break;
      memcpy(results->data + i * out->cols,
             values->data + network->unit_count - out->cols,
             out->cols * sizeof(double));
    }
  }
  if (status == CML_SUCCESS)
    memcpy(out->data, results->data, out->rows * out->cols * sizeof(double));
  cml_matrix_destroy(results);
  cml_vector_destroy(values);
  return status;
}
