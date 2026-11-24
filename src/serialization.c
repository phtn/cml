#define _POSIX_C_SOURCE 200809L

#include "cml/serialization.h"
#include "cml/memory.h"
#include "internal.h"
#include "multiclass_internal.h"
#include "network_internal.h"

#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

_Static_assert(CHAR_BIT == 8 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   sizeof(double) == sizeof(uint64_t),
               "CML persistence requires IEEE 754 binary64");

enum { MODEL_LINEAR = 1, MODEL_LOGISTIC, MODEL_NETWORK, MODEL_MULTICLASS };
enum { FORMAT_VERSION = 1, HEADER_BYTES = 4 + 6 * 8 };

typedef struct {
  uint64_t kind, metadata;
  size_t inputs, groups;
  CMLDenseConfig *layers;
  CMLVector *parameters;
} ModelFile;

static uint64_t double_bits(double value) {
  uint64_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static double bits_double(uint64_t bits) {
  double value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static bool write_u64(FILE *file, uint64_t value) {
  unsigned char bytes[8];
  for (size_t i = 0; i < sizeof(bytes); ++i)
    bytes[i] = (unsigned char)(value >> (8 * i));
  return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes);
}

static bool read_u64(FILE *file, uint64_t *out) {
  unsigned char bytes[8];
  if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes))
    return false;
  uint64_t value = 0;
  for (size_t i = 0; i < sizeof(bytes); ++i)
    value |= (uint64_t)bytes[i] << (8 * i);
  *out = value;
  return true;
}

static bool write_model(FILE *file, const ModelFile *model) {
  if (fwrite("CMLM", 1, 4, file) != 4 || !write_u64(file, FORMAT_VERSION) ||
      !write_u64(file, model->kind) || !write_u64(file, model->inputs) ||
      !write_u64(file, model->groups) || !write_u64(file, model->metadata) ||
      !write_u64(file, model->parameters->size))
    return false;
  if (model->kind == MODEL_NETWORK)
    for (size_t i = 0; i < model->groups; ++i)
      if (!write_u64(file, model->layers[i].units) ||
          !write_u64(file, (uint64_t)model->layers[i].activation))
        return false;
  for (size_t i = 0; i < model->parameters->size; ++i)
    if (!write_u64(file, double_bits(model->parameters->data[i])))
      return false;
  return true;
}

static CMLStatus save(const ModelFile *model, const char *path) {
  if (path == NULL || *path == '\0')
    return CML_ERROR_INVALID_ARGUMENT;
  if (!cml_values_finite(model->parameters->data, model->parameters->size))
    return CML_ERROR_NUMERICAL;
  const char suffix[] = ".tmp.XXXXXX";
  size_t length = strlen(path);
  if (length > SIZE_MAX - sizeof(suffix))
    return CML_ERROR_OVERFLOW;
  char *temporary = cml_malloc(length + sizeof(suffix));
  if (temporary == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  memcpy(temporary, path, length);
  memcpy(temporary + length, suffix, sizeof(suffix));
  int fd = mkstemp(temporary);
  CMLStatus status = CML_ERROR_IO;
  if (fd >= 0) {
    FILE *file = fdopen(fd, "wb");
    if (file == NULL) {
      close(fd);
    } else {
      bool written = write_model(file, model);
      if (fclose(file) == 0 && written && rename(temporary, path) == 0)
        status = CML_SUCCESS;
    }
    if (status != CML_SUCCESS)
      unlink(temporary);
  }
  cml_free(temporary);
  return status;
}

static void release_model(ModelFile *model) {
  cml_free(model->layers);
  cml_vector_destroy(model->parameters);
}

static CMLStatus read_model(FILE *file, uint64_t wanted, ModelFile *model) {
  if (fseek(file, 0, SEEK_END) != 0)
    return CML_ERROR_IO;
  long length = ftell(file);
  if (length < HEADER_BYTES || fseek(file, 0, SEEK_SET) != 0)
    return CML_ERROR_IO;
  char magic[4];
  uint64_t version, kind, inputs, groups, metadata, count;
  if (fread(magic, 1, 4, file) != 4 || memcmp(magic, "CMLM", 4) != 0 ||
      !read_u64(file, &version) || !read_u64(file, &kind) ||
      !read_u64(file, &inputs) || !read_u64(file, &groups) ||
      !read_u64(file, &metadata) || !read_u64(file, &count))
    return CML_ERROR_IO;
  if (version != FORMAT_VERSION)
    return CML_ERROR_NOT_IMPLEMENTED;
  if (kind != wanted)
    return CML_ERROR_INVALID_ARGUMENT;
  uint64_t entries = (uint64_t)(length - HEADER_BYTES);
  if (entries % 8 != 0)
    return CML_ERROR_IO;
  entries /= 8;
  if (inputs == 0 || inputs >= SIZE_MAX / sizeof(double) || groups > SIZE_MAX ||
      count == 0 || count > SIZE_MAX / sizeof(double))
    return CML_ERROR_INVALID_DIMENSION;

  if (kind == MODEL_NETWORK) {
    if (groups == 0 || groups > SIZE_MAX / sizeof(CMLDenseConfig) ||
        groups > entries / 2 || count != entries - 2 * groups || metadata != 0)
      return CML_ERROR_INVALID_DIMENSION;
  } else if (kind == MODEL_MULTICLASS) {
    if (groups < 2 || groups > SIZE_MAX / sizeof(double) / (inputs + 1) ||
        count != groups * (inputs + 1) || count != entries ||
        metadata > CML_MULTICLASS_ONE_VS_REST)
      return CML_ERROR_INVALID_DIMENSION;
  } else {
    if (groups != 0 || count != inputs + 1 || count != entries ||
        (kind == MODEL_LINEAR && metadata != 0))
      return CML_ERROR_INVALID_DIMENSION;
    if (kind == MODEL_LOGISTIC) {
      double threshold = bits_double(metadata);
      if (!isfinite(threshold) || threshold < 0.0 || threshold > 1.0)
        return CML_ERROR_INVALID_ARGUMENT;
    }
  }

  model->kind = kind;
  model->inputs = (size_t)inputs;
  model->groups = (size_t)groups;
  model->metadata = metadata;
  if (kind == MODEL_NETWORK) {
    model->layers = cml_calloc(model->groups, sizeof(*model->layers));
    if (model->layers == NULL)
      return CML_ERROR_OUT_OF_MEMORY;
    uint64_t total = 0, previous = inputs;
    for (size_t i = 0; i < model->groups; ++i) {
      uint64_t units, activation;
      if (!read_u64(file, &units) || !read_u64(file, &activation))
        return CML_ERROR_IO;
      if (units == 0 || units > (count - total) / (previous + 1) ||
          activation > CML_ACTIVATION_SOFTMAX)
        return CML_ERROR_INVALID_DIMENSION;
      model->layers[i] =
          (CMLDenseConfig){(size_t)units, (CMLActivation)activation};
      total += units * (previous + 1);
      previous = units;
    }
    if (total != count)
      return CML_ERROR_INVALID_DIMENSION;
  }

  model->parameters = cml_vector_create((size_t)count);
  if (model->parameters == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  for (size_t i = 0; i < model->parameters->size; ++i) {
    uint64_t bits;
    if (!read_u64(file, &bits))
      return CML_ERROR_IO;
    double value = bits_double(bits);
    if (!isfinite(value))
      return CML_ERROR_NUMERICAL;
    model->parameters->data[i] = value;
  }
  return fgetc(file) == EOF && !ferror(file) ? CML_SUCCESS : CML_ERROR_IO;
}

static CMLStatus load(const char *path, uint64_t kind, ModelFile *model) {
  if (path == NULL || *path == '\0')
    return CML_ERROR_INVALID_ARGUMENT;
  FILE *file = fopen(path, "rb");
  if (file == NULL)
    return CML_ERROR_IO;
  CMLStatus status = read_model(file, kind, model);
  if (fclose(file) != 0 && status == CML_SUCCESS)
    status = CML_ERROR_IO;
  if (status != CML_SUCCESS)
    release_model(model);
  return status;
}

CMLStatus cml_linear_regression_save(const CMLLinearRegression *model,
                                     const char *path) {
  if (!cml_linear_regression_is_trained(model))
    return CML_ERROR_INVALID_ARGUMENT;
  size_t inputs = cml_linear_regression_feature_count(model);
  CMLVector *parameters = cml_vector_create(inputs + 1);
  if (parameters == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  CMLVector weights = {parameters->data, inputs, inputs};
  CMLStatus status = cml_linear_regression_get_parameters(
      model, &weights, &parameters->data[inputs]);
  if (status == CML_SUCCESS) {
    ModelFile file = {
        .kind = MODEL_LINEAR, .inputs = inputs, .parameters = parameters};
    status = save(&file, path);
  }
  cml_vector_destroy(parameters);
  return status;
}

CMLStatus cml_linear_regression_load(const char *path,
                                     CMLLinearRegression **out) {
  if (out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  ModelFile file = {0};
  CMLStatus status = load(path, MODEL_LINEAR, &file);
  if (status != CML_SUCCESS)
    return status;
  CMLLinearRegression *model = cml_linear_regression_create(file.inputs);
  if (model == NULL) {
    status = CML_ERROR_OUT_OF_MEMORY;
  } else {
    CMLVector weights = {file.parameters->data, file.inputs, file.inputs};
    status = cml_linear_regression_set_parameters(
        model, &weights, file.parameters->data[file.inputs]);
  }
  release_model(&file);
  if (status == CML_SUCCESS)
    *out = model;
  else
    cml_linear_regression_destroy(model);
  return status;
}

CMLStatus cml_logistic_regression_save(const CMLLogisticRegression *model,
                                       const char *path) {
  if (!cml_logistic_regression_is_trained(model))
    return CML_ERROR_INVALID_ARGUMENT;
  size_t inputs = cml_logistic_regression_feature_count(model);
  CMLVector *parameters = cml_vector_create(inputs + 1);
  if (parameters == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  CMLStatus status = cml_logistic_regression_get_parameters(model, parameters);
  if (status == CML_SUCCESS) {
    ModelFile file = {
        .kind = MODEL_LOGISTIC,
        .inputs = inputs,
        .parameters = parameters,
        .metadata = double_bits(cml_logistic_regression_threshold(model))};
    status = save(&file, path);
  }
  cml_vector_destroy(parameters);
  return status;
}

CMLStatus cml_logistic_regression_load(const char *path,
                                       CMLLogisticRegression **out) {
  if (out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  ModelFile file = {0};
  CMLStatus status = load(path, MODEL_LOGISTIC, &file);
  if (status != CML_SUCCESS)
    return status;
  CMLLogisticRegression *model = cml_logistic_regression_create(file.inputs);
  if (model == NULL) {
    status = CML_ERROR_OUT_OF_MEMORY;
  } else {
    status = cml_logistic_regression_set_parameters(model, file.parameters);
    if (status == CML_SUCCESS)
      status = cml_logistic_regression_set_threshold(
          model, bits_double(file.metadata));
  }
  release_model(&file);
  if (status == CML_SUCCESS)
    *out = model;
  else
    cml_logistic_regression_destroy(model);
  return status;
}

CMLStatus cml_multiclass_save(const CMLMulticlassClassifier *model,
                              const char *path) {
  if (model == NULL || !model->trained)
    return CML_ERROR_INVALID_ARGUMENT;
  ModelFile file = {.kind = MODEL_MULTICLASS,
                    .inputs = model->features,
                    .groups = model->classes,
                    .metadata = (uint64_t)model->method,
                    .parameters = model->parameters};
  return save(&file, path);
}

CMLStatus cml_multiclass_load(const char *path, CMLMulticlassClassifier **out) {
  if (out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  ModelFile file = {0};
  CMLStatus status = load(path, MODEL_MULTICLASS, &file);
  if (status != CML_SUCCESS)
    return status;
  CMLMulticlassClassifier *model = cml_multiclass_create(
      file.inputs, file.groups, (CMLMulticlassMethod)file.metadata);
  if (model == NULL) {
    release_model(&file);
    return CML_ERROR_OUT_OF_MEMORY;
  }
  cml_vector_destroy(model->parameters);
  model->parameters = file.parameters;
  model->trained = true;
  *out = model;
  return CML_SUCCESS;
}

static CMLStatus save_network(const CMLNetwork *network,
                              const CMLVector *parameters, const char *path) {
  if (network == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_vector_storage(parameters, network->parameters->size);
  if (status != CML_SUCCESS)
    return status;
  CMLDenseConfig *layers = cml_calloc(network->layer_count, sizeof(*layers));
  if (layers == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  for (size_t i = 0; i < network->layer_count; ++i)
    layers[i] = network->layers[i].config;
  CMLVector view = *parameters;
  ModelFile file = {.kind = MODEL_NETWORK,
                    .inputs = network->inputs,
                    .groups = network->layer_count,
                    .layers = layers,
                    .parameters = &view};
  status = save(&file, path);
  cml_free(layers);
  return status;
}

CMLStatus cml_network_save(const CMLNetwork *network, const char *path) {
  return save_network(network, network == NULL ? NULL : network->parameters,
                      path);
}

CMLStatus cml_network_checkpoint(const CMLTrainingRecord *record,
                                 const CMLVector *parameters, void *context) {
  (void)record;
  if (context == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  const CMLNetworkCheckpoint *checkpoint = context;
  return save_network(checkpoint->network, parameters, checkpoint->path);
}

CMLStatus cml_network_load(const char *path, CMLNetwork **out) {
  if (out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  ModelFile file = {0};
  CMLStatus status = load(path, MODEL_NETWORK, &file);
  if (status != CML_SUCCESS)
    return status;
  CMLNetwork *network =
      cml_network_create(file.inputs, file.layers, file.groups, 0);
  if (network == NULL) {
    release_model(&file);
    return CML_ERROR_OUT_OF_MEMORY;
  }
  cml_vector_destroy(network->parameters);
  network->parameters = file.parameters;
  cml_free(file.layers);
  *out = network;
  return CML_SUCCESS;
}
