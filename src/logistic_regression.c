#include "cml/logistic_regression.h"
#include "cml/activation.h"
#include "cml/memory.h"
#include "internal.h"

#include <string.h>

struct CMLLogisticRegression {
  CMLVector *parameters;
  double threshold;
  bool trained;
};

CMLLogisticOptions cml_logistic_options_default(void) {
  return (CMLLogisticOptions){.training = cml_training_options_default(),
                              .negative_weight = 1.0,
                              .positive_weight = 1.0};
}

CMLLogisticRegression *cml_logistic_regression_create(size_t features) {
  if (features == 0 || features >= SIZE_MAX / sizeof(double))
    return NULL;
  CMLLogisticRegression *model = cml_malloc(sizeof(*model));
  if (model == NULL)
    return NULL;
  model->parameters = cml_vector_create(features + 1);
  if (model->parameters == NULL) {
    cml_free(model);
    return NULL;
  }
  model->threshold = 0.5;
  model->trained = false;
  return model;
}

void cml_logistic_regression_destroy(CMLLogisticRegression *model) {
  if (model != NULL) {
    cml_vector_destroy(model->parameters);
    cml_free(model);
  }
}

CMLLogisticRegression *
cml_logistic_regression_clone(const CMLLogisticRegression *model) {
  if (model == NULL)
    return NULL;
  CMLLogisticRegression *copy =
      cml_logistic_regression_create(model->parameters->size - 1);
  if (copy != NULL) {
    memcpy(copy->parameters->data, model->parameters->data,
           model->parameters->size * sizeof(double));
    copy->threshold = model->threshold;
    copy->trained = model->trained;
  }
  return copy;
}

CMLStatus cml_logistic_regression_reset(CMLLogisticRegression *model) {
  if (model == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  cml_vector_fill(model->parameters, 0.0);
  model->trained = false;
  model->threshold = 0.5;
  return CML_SUCCESS;
}

bool cml_logistic_regression_is_trained(const CMLLogisticRegression *model) {
  return model != NULL && model->trained;
}

size_t
cml_logistic_regression_feature_count(const CMLLogisticRegression *model) {
  return model == NULL ? 0 : model->parameters->size - 1;
}

CMLStatus cml_logistic_regression_set_parameters(CMLLogisticRegression *model,
                                                 const CMLVector *parameters) {
  if (model == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_vector_storage(parameters, model->parameters->size);
  if (status != CML_SUCCESS)
    return status;
  if (!cml_values_finite(parameters->data, parameters->size))
    return CML_ERROR_NUMERICAL;
  memcpy(model->parameters->data, parameters->data,
         parameters->size * sizeof(double));
  model->trained = true;
  return CML_SUCCESS;
}

CMLStatus
cml_logistic_regression_get_parameters(const CMLLogisticRegression *model,
                                       CMLVector *out) {
  if (!cml_logistic_regression_is_trained(model))
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_vector_storage(out, model->parameters->size);
  if (status == CML_SUCCESS)
    memcpy(out->data, model->parameters->data, out->size * sizeof(double));
  return status;
}

CMLStatus cml_logistic_regression_set_threshold(CMLLogisticRegression *model,
                                                double threshold) {
  if (model == NULL || !isfinite(threshold) || threshold < 0.0 ||
      threshold > 1.0)
    return CML_ERROR_INVALID_ARGUMENT;
  model->threshold = threshold;
  return CML_SUCCESS;
}

double cml_logistic_regression_threshold(const CMLLogisticRegression *model) {
  return model == NULL ? NAN : model->threshold;
}

typedef struct {
  const CMLDataset *dataset;
  double negative_weight, positive_weight, l1, l2;
} LogisticData;

static CMLStatus logistic_batch(const CMLVector *p, const size_t *indices,
                                size_t count, void *context, double *loss,
                                CMLVector *gradient) {
  const LogisticData *data = context;
  size_t cols = data->dataset->num_features;
  double mean_loss = 0.0;
  if (gradient != NULL)
    cml_vector_fill(gradient, 0.0);
  for (size_t i = 0; i < count; ++i) {
    size_t index = indices == NULL ? i : indices[i];
    const double *row = data->dataset->X->data + index * cols;
    double logit = cml_affine(row, p->data, cols, p->data[cols]);
    if (!isfinite(logit))
      return CML_ERROR_NUMERICAL;
    double label = data->dataset->y->data[index];
    double scale =
        (label == 1.0 ? data->positive_weight : data->negative_weight) /
        (double)count;
    if (loss != NULL)
      mean_loss = fma(scale, cml_bce_logit(label, logit), mean_loss);
    if (gradient != NULL) {
      double error = label == 1.0 ? -cml_sigmoid(-logit) : cml_sigmoid(logit);
      double derivative = scale * error;
      for (size_t j = 0; j < cols; ++j)
        gradient->data[j] = fma(derivative, row[j], gradient->data[j]);
      gradient->data[cols] += derivative;
    }
  }
  if (data->l1 > 0.0 || data->l2 > 0.0) {
    for (size_t j = 0; j < cols; ++j) {
      double weight = p->data[j];
      double quadratic = data->l2 * weight;
      if (loss != NULL)
        mean_loss += data->l1 * fabs(weight) + (0.5 * weight) * quadratic;
      if (gradient != NULL)
        gradient->data[j] +=
            quadratic + data->l1 * ((weight > 0) - (weight < 0));
    }
  }
  if (loss != NULL) {
    if (!isfinite(mean_loss))
      return CML_ERROR_NUMERICAL;
    *loss = mean_loss;
  }
  if (gradient != NULL && !cml_values_finite(gradient->data, gradient->size))
    return CML_ERROR_NUMERICAL;
  return CML_SUCCESS;
}

static bool valid_penalty(const CMLLogisticOptions *options) {
  return options != NULL && isfinite(options->positive_weight) &&
         options->positive_weight > 0.0 && isfinite(options->negative_weight) &&
         options->negative_weight > 0.0 && isfinite(options->l1) &&
         options->l1 >= 0.0 && isfinite(options->l2) && options->l2 >= 0.0;
}

CMLStatus cml_logistic_regression_fit(CMLLogisticRegression *model,
                                      const CMLDataset *training,
                                      const CMLDataset *validation,
                                      const CMLLogisticOptions *options,
                                      CMLTrainingResult *result) {
  if (model == NULL || !valid_penalty(options))
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_classification_validate_binary_dataset(training, NULL);
  if (status == CML_SUCCESS && validation != NULL)
    status = cml_classification_validate_binary_dataset(validation, NULL);
  if (status != CML_SUCCESS)
    return status;
  size_t cols = cml_logistic_regression_feature_count(model);
  if (training->num_features != cols ||
      (validation != NULL && validation->num_features != cols))
    return CML_ERROR_INVALID_DIMENSION;
  LogisticData train = {training, options->negative_weight,
                        options->positive_weight, options->l1, options->l2};
  LogisticData valid = {validation, 1.0, 1.0, 0.0, 0.0};
  status = cml_train(model->parameters, logistic_batch, training->num_samples,
                     &train, validation == NULL ? 0 : validation->num_samples,
                     &valid, &options->training, result);
  if (status == CML_SUCCESS)
    model->trained = true;
  return status;
}

static CMLStatus validate_objective(const CMLDataset *dataset,
                                    const CMLVector *parameters) {
  CMLStatus status = cml_classification_validate_binary_dataset(dataset, NULL);
  if (status != CML_SUCCESS)
    return status;
  status = cml_vector_storage(parameters, dataset->num_features + 1);
  if (status != CML_SUCCESS)
    return status;
  return cml_values_finite(parameters->data, parameters->size)
             ? CML_SUCCESS
             : CML_ERROR_NUMERICAL;
}

CMLStatus cml_logistic_regression_bce(const CMLDataset *dataset,
                                      const CMLVector *parameters,
                                      double *out) {
  if (out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = validate_objective(dataset, parameters);
  if (status != CML_SUCCESS)
    return status;
  LogisticData data = {dataset, 1.0, 1.0, 0.0, 0.0};
  return logistic_batch(parameters, NULL, dataset->num_samples, &data, out,
                        NULL);
}

CMLStatus cml_logistic_regression_bce_gradient(const CMLDataset *dataset,
                                               const CMLVector *parameters,
                                               CMLVector *out) {
  CMLStatus status = validate_objective(dataset, parameters);
  if (status != CML_SUCCESS)
    return status;
  status = cml_vector_storage(out, parameters->size);
  if (status != CML_SUCCESS)
    return status;
  CMLVector *scratch = cml_vector_create(parameters->size);
  if (scratch == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  LogisticData data = {dataset, 1.0, 1.0, 0.0, 0.0};
  status = logistic_batch(parameters, NULL, dataset->num_samples, &data, NULL,
                          scratch);
  if (status == CML_SUCCESS)
    memcpy(out->data, scratch->data, out->size * sizeof(double));
  cml_vector_destroy(scratch);
  return status;
}

CMLStatus
cml_logistic_regression_predict_proba(const CMLLogisticRegression *model,
                                      const CMLVector *features, double *out) {
  if (!cml_logistic_regression_is_trained(model) || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  size_t cols = cml_logistic_regression_feature_count(model);
  CMLStatus status = cml_vector_storage(features, cols);
  if (status != CML_SUCCESS)
    return status;
  double logit = cml_affine(features->data, model->parameters->data, cols,
                            model->parameters->data[cols]);
  if (!isfinite(logit))
    return CML_ERROR_NUMERICAL;
  *out = cml_sigmoid(logit);
  return CML_SUCCESS;
}

CMLStatus cml_logistic_regression_predict(const CMLLogisticRegression *model,
                                          const CMLVector *features,
                                          double *out) {
  if (out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  double probability;
  CMLStatus status =
      cml_logistic_regression_predict_proba(model, features, &probability);
  if (status == CML_SUCCESS)
    *out = probability >= model->threshold ? 1.0 : 0.0;
  return status;
}

static CMLStatus predict_batch(const CMLLogisticRegression *model,
                               const CMLMatrix *features, CMLVector *out,
                               bool classes) {
  if (!cml_logistic_regression_is_trained(model))
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_matrix_storage(features);
  if (status != CML_SUCCESS)
    return status;
  if (features->cols != cml_logistic_regression_feature_count(model))
    return CML_ERROR_INVALID_DIMENSION;
  status = cml_vector_storage(out, features->rows);
  if (status != CML_SUCCESS)
    return status;
  CMLVector *scratch = cml_vector_create(out->size);
  if (scratch == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  for (size_t i = 0; i < features->rows; ++i) {
    CMLVector row = {features->data + i * features->cols, features->cols,
                     features->cols};
    status =
        cml_logistic_regression_predict_proba(model, &row, &scratch->data[i]);
    if (status != CML_SUCCESS)
      break;
    if (classes)
      scratch->data[i] = scratch->data[i] >= model->threshold ? 1.0 : 0.0;
  }
  if (status == CML_SUCCESS)
    memcpy(out->data, scratch->data, out->size * sizeof(double));
  cml_vector_destroy(scratch);
  return status;
}

CMLStatus
cml_logistic_regression_predict_proba_batch(const CMLLogisticRegression *model,
                                            const CMLMatrix *features,
                                            CMLVector *out) {
  return predict_batch(model, features, out, false);
}

CMLStatus
cml_logistic_regression_predict_batch(const CMLLogisticRegression *model,
                                      const CMLMatrix *features,
                                      CMLVector *out) {
  return predict_batch(model, features, out, true);
}
