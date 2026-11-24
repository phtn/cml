#include "cml/activation.h"
#include "cml/logistic_regression.h"
#include "cml/memory.h"
#include "internal.h"
#include "multiclass_internal.h"

#include <string.h>

CMLStatus cml_classification_validate_classes(const CMLDataset *data,
                                              size_t classes) {
  if (classes < 2)
    return CML_ERROR_INVALID_DIMENSION;
  CMLStatus status = cml_dataset_validate(data);
  if (status == CML_SUCCESS)
    status = cml_matrix_storage(data->X);
  if (status == CML_SUCCESS)
    status = cml_vector_storage(data->y, data->num_samples);
  if (status != CML_SUCCESS)
    return status;
  if (!cml_values_finite(data->X->data, data->num_samples * data->num_features))
    return CML_ERROR_NUMERICAL;
  for (size_t i = 0; i < data->num_samples; ++i) {
    double label = data->y->data[i];
    if (!isfinite(label))
      return CML_ERROR_NUMERICAL;
    if (label < 0.0 || label >= (double)classes || label != floor(label))
      return CML_ERROR_INVALID_ARGUMENT;
  }
  return CML_SUCCESS;
}

CMLMulticlassClassifier *cml_multiclass_create(size_t features, size_t classes,
                                               CMLMulticlassMethod method) {
  if (features == 0 || features == SIZE_MAX || classes < 2 ||
      classes > SIZE_MAX / sizeof(double) / (features + 1) ||
      (method != CML_MULTICLASS_SOFTMAX &&
       method != CML_MULTICLASS_ONE_VS_REST))
    return NULL;
  CMLMulticlassClassifier *model = cml_malloc(sizeof(*model));
  if (model == NULL)
    return NULL;
  *model = (CMLMulticlassClassifier){
      .features = features, .classes = classes, .method = method};
  model->parameters = cml_vector_create(classes * (features + 1));
  if (model->parameters == NULL) {
    cml_free(model);
    return NULL;
  }
  return model;
}

void cml_multiclass_destroy(CMLMulticlassClassifier *model) {
  if (model != NULL) {
    cml_vector_destroy(model->parameters);
    cml_free(model);
  }
}

static double class_logit(const double *sample, const CMLVector *params,
                          size_t cols, size_t class_index) {
  const double *weights = params->data + class_index * (cols + 1);
  return cml_affine(sample, weights, cols, weights[cols]);
}

typedef struct {
  const CMLDataset *data;
  CMLVector *scores;
} SoftmaxData;

static CMLStatus softmax_batch(const CMLVector *params, const size_t *indices,
                               size_t count, void *context, double *loss,
                               CMLVector *gradient) {
  const SoftmaxData *data = context;
  size_t cols = data->data->num_features;
  CMLVector *scores = data->scores;
  double mean = 0.0;
  if (gradient != NULL)
    cml_vector_fill(gradient, 0.0);
  for (size_t i = 0; i < count; ++i) {
    size_t index = indices == NULL ? i : indices[i];
    const double *row = data->data->X->data + index * cols;
    size_t label = (size_t)data->data->y->data[index];
    double maximum = -INFINITY;
    for (size_t k = 0; k < scores->size; ++k) {
      scores->data[k] = class_logit(row, params, cols, k);
      if (!isfinite(scores->data[k]))
        return CML_ERROR_NUMERICAL;
      maximum = fmax(maximum, scores->data[k]);
    }
    double penalty = maximum - scores->data[label];
    double total = 0.0;
    for (size_t k = 0; k < scores->size; ++k) {
      scores->data[k] = exp(scores->data[k] - maximum);
      total += scores->data[k];
    }
    if (loss != NULL)
      mean += penalty / (double)count + log(total) / (double)count;
    if (gradient != NULL)
      for (size_t k = 0; k < scores->size; ++k) {
        double error =
            (scores->data[k] / total - (double)(k == label)) / (double)count;
        double *weights = gradient->data + k * (cols + 1);
        for (size_t j = 0; j < cols; ++j)
          weights[j] = fma(error, row[j], weights[j]);
        weights[cols] += error;
      }
  }
  if (loss != NULL)
    *loss = mean;
  return CML_SUCCESS;
}

static CMLStatus fit_ovr(const CMLMulticlassClassifier *model,
                         CMLVector *candidate, const CMLDataset *training,
                         const CMLDataset *validation,
                         const CMLTrainingOptions *options,
                         CMLTrainingResult *result) {
  CMLVector *labels = cml_vector_create(training->num_samples);
  CMLVector *valid_labels =
      validation == NULL ? NULL : cml_vector_create(validation->num_samples);
  CMLLogisticRegression *binary =
      cml_logistic_regression_create(model->features);
  CMLStatus status = CML_ERROR_OUT_OF_MEMORY;
  if (labels == NULL || binary == NULL ||
      (validation != NULL && valid_labels == NULL))
    goto done;
  CMLDataset train = *training, valid = {0};
  train.y = labels;
  if (validation != NULL) {
    valid = *validation;
    valid.y = valid_labels;
  }
  CMLLogisticOptions config = cml_logistic_options_default();
  config.training = *options;
  status = CML_SUCCESS;
  size_t stride = model->features + 1;
  for (size_t k = 0; k < model->classes && status == CML_SUCCESS; ++k) {
    for (size_t i = 0; i < labels->size; ++i)
      labels->data[i] = (double)(training->y->data[i] == (double)k);
    if (valid_labels != NULL)
      for (size_t i = 0; i < valid_labels->size; ++i)
        valid_labels->data[i] = (double)(validation->y->data[i] == (double)k);
    CMLVector part = {candidate->data + k * stride, stride, stride};
    status = cml_logistic_regression_set_parameters(binary, &part);
    if (status == CML_SUCCESS)
      status = cml_logistic_regression_fit(
          binary, &train, validation == NULL ? NULL : &valid, &config, result);
    if (status == CML_SUCCESS)
      status = cml_logistic_regression_get_parameters(binary, &part);
  }
done:
  cml_logistic_regression_destroy(binary);
  cml_vector_destroy(valid_labels);
  cml_vector_destroy(labels);
  return status;
}

CMLStatus cml_multiclass_fit(CMLMulticlassClassifier *model,
                             const CMLDataset *training,
                             const CMLDataset *validation,
                             const CMLTrainingOptions *options,
                             CMLTrainingResult *result) {
  if (model == NULL || options == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status =
      cml_classification_validate_classes(training, model->classes);
  if (status == CML_SUCCESS && validation != NULL)
    status = cml_classification_validate_classes(validation, model->classes);
  if (status != CML_SUCCESS)
    return status;
  if (training->num_features != model->features ||
      (validation != NULL && validation->num_features != model->features))
    return CML_ERROR_INVALID_DIMENSION;

  if (model->method == CML_MULTICLASS_SOFTMAX) {
    CMLVector *scores = cml_vector_create(model->classes);
    if (scores == NULL)
      return CML_ERROR_OUT_OF_MEMORY;
    SoftmaxData train = {training, scores}, valid = {validation, scores};
    status = cml_train(model->parameters, softmax_batch, training->num_samples,
                       &train, validation == NULL ? 0 : validation->num_samples,
                       &valid, options, result);
    cml_vector_destroy(scores);
  } else {
    CMLVector *candidate = cml_vector_clone(model->parameters);
    if (candidate == NULL)
      return CML_ERROR_OUT_OF_MEMORY;
    status = fit_ovr(model, candidate, training, validation, options, result);
    if (status == CML_SUCCESS) {
      cml_vector_destroy(model->parameters);
      model->parameters = candidate;
    } else {
      cml_vector_destroy(candidate);
    }
  }
  if (status == CML_SUCCESS)
    model->trained = true;
  return status;
}

static CMLStatus validate_input(const CMLMulticlassClassifier *model,
                                const CMLVector *features) {
  if (model == NULL || !model->trained)
    return CML_ERROR_INVALID_ARGUMENT;
  return cml_vector_storage(features, model->features);
}

CMLStatus cml_multiclass_predict_scores(const CMLMulticlassClassifier *model,
                                        const CMLVector *features,
                                        CMLVector *out) {
  CMLStatus status = validate_input(model, features);
  if (status == CML_SUCCESS)
    status = cml_vector_storage(out, model->classes);
  if (status != CML_SUCCESS)
    return status;
  CMLVector *scores = cml_vector_create(model->classes);
  if (scores == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  for (size_t k = 0; k < model->classes; ++k)
    scores->data[k] =
        class_logit(features->data, model->parameters, model->features, k);
  if (!cml_values_finite(scores->data, scores->size)) {
    status = CML_ERROR_NUMERICAL;
  } else if (model->method == CML_MULTICLASS_SOFTMAX) {
    status = cml_softmax(scores, out);
  } else {
    status = cml_vector_sigmoid(scores, out);
  }
  cml_vector_destroy(scores);
  return status;
}

CMLStatus cml_multiclass_predict(const CMLMulticlassClassifier *model,
                                 const CMLVector *features, size_t *out) {
  if (out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = validate_input(model, features);
  if (status != CML_SUCCESS)
    return status;
  double maximum = -INFINITY;
  size_t best = 0;
  for (size_t k = 0; k < model->classes; ++k) {
    double logit =
        class_logit(features->data, model->parameters, model->features, k);
    if (!isfinite(logit))
      return CML_ERROR_NUMERICAL;
    if (logit > maximum) {
      maximum = logit;
      best = k;
    }
  }
  *out = best;
  return CML_SUCCESS;
}
