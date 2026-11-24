#include "cml/activation.h"
#include "cml/memory.h"
#include "internal.h"
#include "network_internal.h"

#include <string.h>

typedef struct {
  const CMLNetwork *network;
  const CMLMatrix *inputs, *targets;
  CMLNetworkLoss loss;
  double *values, *logits, *deltas;
} BackpropData;

static CMLStatus validate_targets(const BackpropData *data) {
  if (data->loss < CML_NETWORK_MSE ||
      data->loss > CML_NETWORK_CATEGORICAL_CROSS_ENTROPY)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_matrix_storage(data->inputs);
  if (status == CML_SUCCESS)
    status = cml_matrix_storage(data->targets);
  if (status != CML_SUCCESS)
    return status;
  const CMLNetwork *network = data->network;
  CMLDenseConfig output = network->layers[network->layer_count - 1].config;
  const CMLMatrix *targets = data->targets;
  if (data->inputs->cols != network->inputs ||
      targets->rows != data->inputs->rows || targets->cols != output.units)
    return CML_ERROR_INVALID_DIMENSION;
  if ((data->loss == CML_NETWORK_BINARY_CROSS_ENTROPY &&
       output.activation != CML_ACTIVATION_SIGMOID) ||
      (data->loss == CML_NETWORK_CATEGORICAL_CROSS_ENTROPY &&
       output.activation != CML_ACTIVATION_SOFTMAX))
    return CML_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < targets->rows; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < targets->cols; ++j) {
      double value = targets->data[i * targets->cols + j];
      if (!isfinite(value))
        return CML_ERROR_NUMERICAL;
      if (data->loss == CML_NETWORK_BINARY_CROSS_ENTROPY && value != 0.0 &&
          value != 1.0)
        return CML_ERROR_INVALID_ARGUMENT;
      if (data->loss == CML_NETWORK_CATEGORICAL_CROSS_ENTROPY &&
          (value < 0.0 || value > 1.0))
        return CML_ERROR_INVALID_ARGUMENT;
      sum += value;
    }
    if (data->loss == CML_NETWORK_CATEGORICAL_CROSS_ENTROPY &&
        fabs(sum - 1.0) > 1e-10)
      return CML_ERROR_INVALID_ARGUMENT;
  }
  return CML_SUCCESS;
}

static CMLStatus prepare_workspace(BackpropData *data) {
  size_t units = data->network->unit_count;
  if (units > SIZE_MAX / sizeof(double) / 3)
    return CML_ERROR_OVERFLOW;
  data->values = cml_calloc(units * 3, sizeof(double));
  if (data->values == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  data->logits = data->values + units;
  data->deltas = data->logits + units;
  return CML_SUCCESS;
}

static void activation_backward(CMLActivation activation, size_t units,
                                const double *values, const double *logits,
                                double *deltas) {
  if (activation == CML_ACTIVATION_SOFTMAX) {
    double projection = 0.0;
    for (size_t j = 0; j < units; ++j)
      projection = fma(values[j], deltas[j], projection);
    for (size_t j = 0; j < units; ++j)
      deltas[j] = values[j] * (deltas[j] - projection);
    return;
  }
  for (size_t j = 0; j < units; ++j) {
    switch (activation) {
    case CML_ACTIVATION_RELU:
      if (logits[j] <= 0.0)
        deltas[j] = 0.0;
      break;
    case CML_ACTIVATION_TANH:
      deltas[j] *= fma(-values[j], values[j], 1.0);
      break;
    case CML_ACTIVATION_SIGMOID:
      deltas[j] *= values[j] * cml_sigmoid(-logits[j]);
      break;
    case CML_ACTIVATION_NONE:
    case CML_ACTIVATION_SOFTMAX:
      break;
    }
  }
}

static double output_derivative(const BackpropData *data, const double *targets,
                                double batch_scale, bool need_gradient) {
  const CMLNetwork *network = data->network;
  const CMLDenseLayer *last = &network->layers[network->layer_count - 1];
  size_t units = last->config.units;
  const double *values = data->values + last->unit_offset;
  const double *logits = data->logits + last->unit_offset;
  double *deltas = data->deltas + last->unit_offset;
  double loss = 0.0;
  if (data->loss == CML_NETWORK_CATEGORICAL_CROSS_ENTROPY) {
    double maximum = logits[0], total = 0.0, target_sum = 0.0;
    for (size_t j = 1; j < units; ++j)
      maximum = fmax(maximum, logits[j]);
    for (size_t j = 0; j < units; ++j) {
      total += exp(logits[j] - maximum);
      target_sum += targets[j];
      if (targets[j] != 0.0) {
        double difference = maximum - logits[j];
        double scale = batch_scale * targets[j];
        loss += isfinite(difference) ? scale * difference
                                     : scale * maximum - scale * logits[j];
      }
    }
    if (need_gradient)
      for (size_t j = 0; j < units; ++j)
        deltas[j] = batch_scale * fma(target_sum, values[j], -targets[j]);
    return loss + (batch_scale * target_sum) * log(total);
  }
  double scale = batch_scale / (double)units;
  for (size_t j = 0; j < units; ++j) {
    if (data->loss == CML_NETWORK_BINARY_CROSS_ENTROPY) {
      loss = fma(scale, cml_bce_logit(targets[j], logits[j]), loss);
      if (need_gradient)
        deltas[j] =
            scale * (targets[j] == 1.0 ? -cml_sigmoid(-logits[j]) : values[j]);
    } else {
      double error = values[j] - targets[j];
      loss = fma(error * scale, error, loss);
      if (need_gradient)
        deltas[j] = (2.0 * scale) * error;
    }
  }
  if (need_gradient && data->loss == CML_NETWORK_MSE)
    activation_backward(last->config.activation, units, values, logits, deltas);
  return loss;
}

static void propagate(const BackpropData *data, const double *input,
                      const double *parameters, double *gradient) {
  const CMLNetwork *network = data->network;
  for (size_t i = network->layer_count; i-- > 0;) {
    const CMLDenseLayer *layer = &network->layers[i];
    size_t cols = layer->inputs, rows = layer->config.units;
    const double *weights = parameters + layer->parameter_offset;
    double *gradient_weights = gradient + layer->parameter_offset;
    double *gradient_biases = gradient_weights + rows * cols;
    const double *deltas = data->deltas + layer->unit_offset;
    const double *activations =
        i == 0 ? input : data->values + layer->unit_offset - cols;
    for (size_t j = 0; j < rows; ++j) {
      gradient_biases[j] += deltas[j];
      for (size_t k = 0; k < cols; ++k)
        gradient_weights[j * cols + k] =
            fma(deltas[j], activations[k], gradient_weights[j * cols + k]);
    }
    if (i > 0) {
      const CMLDenseLayer *previous = &network->layers[i - 1];
      double *incoming = data->deltas + previous->unit_offset;
      for (size_t k = 0; k < cols; ++k) {
        incoming[k] = 0.0;
        for (size_t j = 0; j < rows; ++j)
          incoming[k] = fma(weights[j * cols + k], deltas[j], incoming[k]);
      }
      activation_backward(previous->config.activation, cols, activations,
                          data->logits + previous->unit_offset, incoming);
    }
  }
}

static CMLStatus network_batch(const CMLVector *parameters,
                               const size_t *indices, size_t count,
                               void *context, double *loss,
                               CMLVector *gradient) {
  const BackpropData *data = context;
  if (gradient != NULL)
    cml_vector_fill(gradient, 0.0);
  double mean = 0.0;
  for (size_t i = 0; i < count; ++i) {
    size_t row = indices == NULL ? i : indices[i];
    const double *input = data->inputs->data + row * data->inputs->cols;
    const double *target = data->targets->data + row * data->targets->cols;
    CMLStatus status = cml_network_forward(data->network, parameters->data,
                                           input, data->values, data->logits);
    if (status != CML_SUCCESS)
      return status;
    mean +=
        output_derivative(data, target, 1.0 / (double)count, gradient != NULL);
    if (gradient != NULL)
      propagate(data, input, parameters->data, gradient->data);
  }
  if ((loss != NULL && !isfinite(mean)) ||
      (gradient != NULL && !cml_values_finite(gradient->data, gradient->size)))
    return CML_ERROR_NUMERICAL;
  if (loss != NULL)
    *loss = mean;
  return CML_SUCCESS;
}

CMLStatus cml_network_objective(const CMLNetwork *network,
                                const CMLMatrix *inputs,
                                const CMLMatrix *targets, CMLNetworkLoss loss,
                                const CMLVector *parameters, double *out_loss,
                                CMLVector *out_gradient) {
  if (network == NULL || (out_loss == NULL && out_gradient == NULL))
    return CML_ERROR_INVALID_ARGUMENT;
  if (parameters == NULL)
    parameters = network->parameters;
  CMLStatus status = cml_vector_storage(parameters, network->parameters->size);
  if (status == CML_SUCCESS && out_gradient != NULL)
    status = cml_vector_storage(out_gradient, parameters->size);
  if (status != CML_SUCCESS)
    return status;
  if (!cml_values_finite(parameters->data, parameters->size))
    return CML_ERROR_NUMERICAL;

  BackpropData data = {
      .network = network, .inputs = inputs, .targets = targets, .loss = loss};
  status = validate_targets(&data);
  if (status == CML_SUCCESS)
    status = prepare_workspace(&data);
  if (status != CML_SUCCESS)
    return status;
  CMLVector *gradient =
      out_gradient == NULL ? NULL : cml_vector_create(parameters->size);
  if (out_gradient != NULL && gradient == NULL)
    status = CML_ERROR_OUT_OF_MEMORY;
  else
    status = network_batch(parameters, NULL, inputs->rows, &data, out_loss,
                           gradient);
  if (status == CML_SUCCESS && gradient != NULL)
    memcpy(out_gradient->data, gradient->data, gradient->size * sizeof(double));
  cml_vector_destroy(gradient);
  cml_free(data.values);
  return status;
}

static CMLStatus checker_loss(const CMLVector *params, void *context,
                              double *out) {
  const BackpropData *data = context;
  return network_batch(params, NULL, data->inputs->rows, context, out, NULL);
}

static CMLStatus checker_gradient(const CMLVector *params, void *context,
                                  CMLVector *out) {
  const BackpropData *data = context;
  return network_batch(params, NULL, data->inputs->rows, context, NULL, out);
}

CMLStatus cml_network_check_gradient(const CMLNetwork *network,
                                     const CMLMatrix *inputs,
                                     const CMLMatrix *targets,
                                     CMLNetworkLoss loss, double step,
                                     double abs_tolerance, double rel_tolerance,
                                     CMLGradientCheckResult *out) {
  if (network == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  BackpropData data = {
      .network = network, .inputs = inputs, .targets = targets, .loss = loss};
  CMLStatus status = validate_targets(&data);
  if (status == CML_SUCCESS)
    status = prepare_workspace(&data);
  if (status == CML_SUCCESS)
    status =
        cml_gradient_check(network->parameters, checker_loss, checker_gradient,
                           &data, step, abs_tolerance, rel_tolerance, out);
  cml_free(data.values);
  return status;
}

CMLStatus cml_network_train(CMLNetwork *network, const CMLMatrix *inputs,
                            const CMLMatrix *targets,
                            const CMLMatrix *validation_inputs,
                            const CMLMatrix *validation_targets,
                            CMLNetworkLoss loss,
                            const CMLTrainingOptions *options,
                            CMLTrainingResult *result) {
  if (network == NULL ||
      (validation_inputs == NULL) != (validation_targets == NULL))
    return CML_ERROR_INVALID_ARGUMENT;
  BackpropData train = {
      .network = network, .inputs = inputs, .targets = targets, .loss = loss};
  BackpropData valid = {.network = network,
                        .inputs = validation_inputs,
                        .targets = validation_targets,
                        .loss = loss};
  CMLStatus status = validate_targets(&train);
  if (status == CML_SUCCESS && validation_inputs != NULL)
    status = validate_targets(&valid);
  if (status == CML_SUCCESS)
    status = prepare_workspace(&train);
  if (status == CML_SUCCESS) {
    valid.values = train.values;
    valid.logits = train.logits;
    valid.deltas = train.deltas;
    status = cml_train(network->parameters, network_batch, inputs->rows, &train,
                       validation_inputs == NULL ? 0 : validation_inputs->rows,
                       &valid, options, result);
  }
  cml_free(train.values);
  return status;
}
