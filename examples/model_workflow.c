#define _POSIX_C_SOURCE 200809L

#include "cml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Fold standardization into the affine model for inference on raw features. */
static CMLStatus deploy_on_raw_features(CMLLogisticRegression *model,
                                        const CMLStandardScaler *scaler) {
  double values[3];
  CMLVector parameters = {values, 3, 3};
  CMLStatus status = cml_logistic_regression_get_parameters(model, &parameters);
  if (status != CML_SUCCESS)
    return status;
  for (size_t i = 0; i < scaler->num_features; ++i) {
    values[i] =
        scaler->stddev[i] <= 1e-15 ? 0.0 : values[i] / scaler->stddev[i];
    values[2] = fma(-values[i], scaler->mean[i], values[2]);
  }
  return cml_logistic_regression_set_parameters(model, &parameters);
}

static CMLStatus run_workflow(const char *path) {
  CMLStatus status = CML_ERROR_OUT_OF_MEMORY;
  CMLDataset *all = cml_dataset_create(100, 2), *train = NULL, *test = NULL,
             *valid = NULL;
  CMLStandardScaler *scaler = cml_standard_scaler_create();
  CMLLogisticRegression *model = cml_logistic_regression_create(2),
                        *loaded = NULL;
  CMLVector *scores = NULL, *before = NULL, *after = NULL, *classes = NULL;
  CMLRocCurve *validation_roc = NULL, *test_roc = NULL;
  CMLTrainingHistory history = {0};
  if (all == NULL || scaler == NULL || model == NULL)
    goto done;

  /* A strong feature and a differently scaled nuisance feature. */
  for (size_t i = 0; i < all->num_samples; ++i) {
    double sign = i % 2 == 0 ? -1.0 : 1.0;
    all->X->data[2 * i] = 1500.0 + sign * (350.0 + 13.0 * (double)(i % 10));
    all->X->data[2 * i + 1] = (double)(i % 7);
    all->y->data[i] = sign > 0.0 ? 1.0 : 0.0;
  }

  CMLBinaryClassCounts counts;
  status = cml_classification_validate_binary_dataset(all, &counts);
  if (status == CML_SUCCESS)
    status =
        cml_dataset_split(all, 0.6, 0.2, 0.2, true, 42, &train, &test, &valid);
  if (status == CML_SUCCESS)
    status = cml_standard_scaler_fit_transform(scaler, train->X, train->X);
  if (status == CML_SUCCESS)
    status = cml_standard_scaler_transform(scaler, valid->X, valid->X);
  if (status != CML_SUCCESS)
    goto done;

  scores = cml_vector_create(valid->num_samples);
  before = cml_vector_create(test->num_samples);
  after = cml_vector_create(test->num_samples);
  classes = cml_vector_create(test->num_samples);
  if (scores == NULL || before == NULL || after == NULL || classes == NULL) {
    status = CML_ERROR_OUT_OF_MEMORY;
    goto done;
  }
  CMLLogisticOptions options = cml_logistic_options_default();
  options.training.method = CML_OPTIMIZER_ADAM;
  options.training.optimizer.learning_rate = 0.05;
  options.training.batch_size = 8;
  options.training.shuffle = true;
  options.training.seed = 17;
  options.training.epochs = 200;
  options.training.patience = 20;
  options.training.min_improvement = 1e-5;
  options.training.record_every = 10;
  options.training.history = &history;
  options.l2 = 0.005;
  CMLTrainingResult result;
  status = cml_logistic_regression_fit(model, train, valid, &options, &result);
  if (status == CML_SUCCESS)
    status =
        cml_logistic_regression_predict_proba_batch(model, valid->X, scores);
  if (status == CML_SUCCESS)
    status = cml_roc_curve(valid->y, scores, &validation_roc);
  if (status == CML_SUCCESS)
    status = cml_logistic_regression_set_threshold(
        model, validation_roc->best_threshold);
  if (status == CML_SUCCESS)
    status = deploy_on_raw_features(model, scaler);
  if (status == CML_SUCCESS)
    status =
        cml_logistic_regression_predict_proba_batch(model, test->X, before);
  if (status == CML_SUCCESS)
    status = cml_logistic_regression_save(model, path);
  if (status != CML_SUCCESS)
    goto done;
  cml_logistic_regression_destroy(model);
  model = NULL;

  status = cml_logistic_regression_load(path, &loaded);
  if (status == CML_SUCCESS)
    status =
        cml_logistic_regression_predict_proba_batch(loaded, test->X, after);
  if (status == CML_SUCCESS)
    status = cml_logistic_regression_predict_batch(loaded, test->X, classes);
  CMLBinaryMetrics metrics;
  if (status == CML_SUCCESS)
    status = cml_binary_metrics(test->y, classes, &metrics);
  if (status == CML_SUCCESS)
    status = cml_roc_curve(test->y, after, &test_roc);
  if (status != CML_SUCCESS)
    goto done;
  for (size_t i = 0; i < before->size; ++i) {
    if (before->data[i] != after->data[i]) {
      status = CML_ERROR_NUMERICAL;
      goto done;
    }
  }

  printf("Class counts  negative %zu  positive %zu\n", counts.negative_count,
         counts.positive_count);
  printf("Best epoch %zu  validation BCE %.6f  history records %zu\n",
         result.best_epoch, result.validation_loss, history.size);
  printf("Test accuracy %.3f  F1 %.3f  ROC-AUC %.3f\n", metrics.accuracy,
         metrics.f1, test_roc->auc);
  printf("Confusion matrix  TN %zu  FP %zu  FN %zu  TP %zu\n",
         metrics.confusion.true_negative, metrics.confusion.false_positive,
         metrics.confusion.false_negative, metrics.confusion.true_positive);
  printf("Reloaded probabilities match all %zu held-out examples\n",
         after->size);

done:
  cml_training_history_destroy(&history);
  cml_roc_curve_destroy(test_roc);
  cml_roc_curve_destroy(validation_roc);
  cml_vector_destroy(classes);
  cml_vector_destroy(after);
  cml_vector_destroy(before);
  cml_vector_destroy(scores);
  cml_logistic_regression_destroy(loaded);
  cml_logistic_regression_destroy(model);
  cml_standard_scaler_destroy(scaler);
  cml_dataset_destroy(valid);
  cml_dataset_destroy(test);
  cml_dataset_destroy(train);
  cml_dataset_destroy(all);
  return status;
}

int main(void) {
  char path[] = "/tmp/cml-workflow-XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0)
    return EXIT_FAILURE;
  if (close(fd) != 0) {
    unlink(path);
    return EXIT_FAILURE;
  }
  CMLStatus status = run_workflow(path);
  unlink(path);
  if (status != CML_SUCCESS)
    fprintf(stderr, "Model workflow failed (%s)\n", cml_status_string(status));
  return status == CML_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
