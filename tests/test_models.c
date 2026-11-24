#define _POSIX_C_SOURCE 200809L

#include "cml.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void metrics(void) {
  double actual[] = {1, 2, 3}, predicted[] = {-1, 4, 3};
  CMLVector y = {actual, 3, 3}, p = {predicted, 3, 3};
  CMLRegressionMetrics r;
  assert(cml_regression_metrics(&y, &p, &r) == CML_SUCCESS);
  assert(fabs(r.r_squared + 3.0) < 1e-14 && r.samples == 3);
  assert(fabs(r.mse - 8.0 / 3.0) < 1e-14);
  actual[0] = actual[1] = actual[2] = 1;
  assert(cml_r_squared(&y, &p, &r.r_squared) == CML_SUCCESS &&
         r.r_squared == 0.0);

  double a[] = {0, 0, 1, 2}, b[] = {0, 1, 1, 1};
  CMLVector labels = {a, 4, 4}, guesses = {b, 4, 4};
  CMLMulticlassMetrics *report = NULL;
  assert(cml_multiclass_metrics(&labels, &guesses, 3, &report) == CML_SUCCESS);
  assert(report->confusion[0] == 1 && report->confusion[1] == 1 &&
         report->confusion[4] == 1 && report->confusion[7] == 1);
  assert(report->accuracy == 0.5 && report->micro.f1 == 0.5);
  assert(fabs(report->weighted.f1 - 11.0 / 24.0) < 1e-14);
  assert(report->per_class[2].recall == 0.0);
  cml_multiclass_metrics_destroy(report);
}

static void splits(void) {
  CMLDataset *data = cml_dataset_create(12, 1);
  assert(data != NULL);
  for (size_t i = 0; i < data->num_samples; ++i) {
    data->X->data[i] = (double)i;
    data->y->data[i] = (double)(i >= 9);
  }
  assert(cml_dataset_set_target_name(data, "label") == CML_SUCCESS);
  CMLDataset *train = NULL, *test = NULL;
  assert(cml_dataset_split_stratified(data, 0.67, 0.33, 7, &train, &test) ==
         CML_SUCCESS);
  CMLBinaryClassCounts counts;
  assert(cml_classification_validate_binary_dataset(train, &counts) ==
         CML_SUCCESS);
  assert(counts.negative_count == 6 && counts.positive_count == 2);
  assert(cml_classification_validate_binary_dataset(test, &counts) ==
         CML_SUCCESS);
  assert(counts.negative_count == 3 && counts.positive_count == 1);
  assert(train->target_name != NULL);
  cml_dataset_destroy(train);
  cml_dataset_destroy(test);

  unsigned union_bits = 0;
  for (size_t fold = 0; fold < 5; ++fold) {
    assert(cml_dataset_kfold_split(data, 5, fold, true, 42, &train, &test) ==
           CML_SUCCESS);
    unsigned bits = 0, training_bits = 0;
    for (size_t i = 0; i < test->num_samples; ++i)
      bits |= 1u << (unsigned)test->X->data[i];
    for (size_t i = 0; i < train->num_samples; ++i)
      training_bits |= 1u << (unsigned)train->X->data[i];
    assert((bits & training_bits) == 0 && (bits | training_bits) == 4095);
    assert((union_bits & bits) == 0);
    union_bits |= bits;
    cml_dataset_destroy(train);
    cml_dataset_destroy(test);
  }
  assert(union_bits == 4095);
  cml_dataset_destroy(data);

  CMLBatchIterator *iter = cml_batch_iterator_create(5, 2, false, 0);
  assert(iter != NULL);
  const size_t *indices;
  size_t count;
  assert(cml_batch_iterator_next(iter, &indices, &count) == CML_SUCCESS &&
         count == 2 && indices[0] == 0 && indices[1] == 1);
  assert(cml_batch_iterator_next(iter, &indices, &count) == CML_SUCCESS &&
         count == 2 && indices[0] == 2 && indices[1] == 3);
  assert(cml_batch_iterator_next(iter, &indices, &count) == CML_SUCCESS &&
         count == 1 && indices[0] == 4);
  assert(cml_batch_iterator_next(iter, &indices, &count) == CML_SUCCESS &&
         count == 0 && indices == NULL);
  assert(cml_batch_iterator_reset(iter, false, 0) == CML_SUCCESS);
  assert(cml_batch_iterator_next(iter, &indices, &count) == CML_SUCCESS &&
         indices[0] == 0);
  cml_batch_iterator_destroy(iter);
}

static void regression_files(const char *path) {
  double input_value = 4.0, parameters[] = {2.0, 1.0};
  CMLVector input = {&input_value, 1, 1}, w = {parameters, 1, 2};
  CMLLinearRegression *linear = cml_linear_regression_create(1);
  assert(linear != NULL);
  assert(cml_linear_regression_set_parameters(linear, &w, 1.0) == CML_SUCCESS);
  assert(cml_linear_regression_save(linear, path) == CML_SUCCESS);
  cml_linear_regression_destroy(linear);
  linear = NULL;
  assert(cml_linear_regression_load(path, &linear) == CML_SUCCESS);
  double value;
  assert(cml_linear_regression_predict(linear, &input, &value) == CML_SUCCESS &&
         value == 9.0);
  cml_linear_regression_destroy(linear);

  CMLLogisticRegression *logistic = cml_logistic_regression_create(1);
  assert(logistic != NULL);
  w.size = 2;
  assert(cml_logistic_regression_set_parameters(logistic, &w) == CML_SUCCESS);
  assert(cml_logistic_regression_set_threshold(logistic, 0.87) == CML_SUCCESS);
  assert(cml_logistic_regression_save(logistic, path) == CML_SUCCESS);
  double probability;
  assert(cml_logistic_regression_predict_proba(logistic, &input,
                                               &probability) == CML_SUCCESS);
  cml_logistic_regression_destroy(logistic);
  logistic = NULL;
  assert(cml_logistic_regression_load(path, &logistic) == CML_SUCCESS);
  assert(cml_logistic_regression_threshold(logistic) == 0.87);
  assert(cml_logistic_regression_predict_proba(logistic, &input, &value) ==
             CML_SUCCESS &&
         value == probability);
  cml_logistic_regression_destroy(logistic);
}

static void network_files(const char *path) {
  double samples[] = {0, 0, 0, 1, 1, 0, 1, 1};
  double labels[] = {0, 1, 1, 0};
  CMLMatrix inputs = {samples, 4, 2, 8}, targets = {labels, 4, 1, 4};
  CMLDenseConfig layers[] = {{3, CML_ACTIVATION_TANH},
                             {1, CML_ACTIVATION_SIGMOID}};
  CMLNetwork *network = cml_network_create(2, layers, 2, 42);
  assert(network != NULL);
  CMLTrainingOptions config = cml_training_options_default();
  config.epochs = 8;
  config.checkpoint = cml_network_checkpoint;
  CMLNetworkCheckpoint checkpoint = {network, path};
  config.checkpoint_context = &checkpoint;
  assert(cml_network_train(network, &inputs, &targets, &inputs, &targets,
                           CML_NETWORK_BINARY_CROSS_ENTROPY, &config,
                           NULL) == CML_SUCCESS);
  double before[4], after[4];
  CMLMatrix out = {before, 4, 1, 4};
  assert(cml_network_predict_batch(network, &inputs, &out) == CML_SUCCESS);
  cml_network_destroy(network);
  network = NULL;
  assert(cml_network_load(path, &network) == CML_SUCCESS);
  assert(cml_network_layer_count(network) == 2 &&
         cml_network_parameter_count(network) == 13);
  out.data = after;
  assert(cml_network_predict_batch(network, &inputs, &out) == CML_SUCCESS);
  for (size_t i = 0; i < 4; ++i)
    assert(before[i] == after[i]);
  FILE *file = fopen(path, "r+b");
  assert(file != NULL && fseek(file, 4, SEEK_SET) == 0);
  assert(fputc(2, file) == 2 && fclose(file) == 0);
  CMLNetwork *saved = network;
  assert(cml_network_load(path, &network) == CML_ERROR_NOT_IMPLEMENTED &&
         network == saved);
  assert(cml_network_save(network, path) == CML_SUCCESS);
  assert(truncate(path, 55) == 0);
  assert(cml_network_load(path, &network) != CML_SUCCESS && network == saved);
  cml_network_destroy(network);
}

static void multiclass_file(const char *path) {
  double x[] = {-1, 0, 1}, y[] = {0, 1, 2};
  CMLMatrix features = {x, 3, 1, 3};
  CMLVector labels = {y, 3, 3};
  CMLDataset data = {
      .X = &features, .y = &labels, .num_samples = 3, .num_features = 1};
  CMLMulticlassClassifier *model =
      cml_multiclass_create(1, 3, CML_MULTICLASS_SOFTMAX);
  assert(model != NULL);
  CMLTrainingOptions options = cml_training_options_default();
  assert(cml_multiclass_fit(model, &data, NULL, &options, NULL) == CML_SUCCESS);
  CMLVector sample = {x, 1, 1};
  size_t before, after;
  assert(cml_multiclass_predict(model, &sample, &before) == CML_SUCCESS);
  assert(cml_multiclass_save(model, path) == CML_SUCCESS);
  cml_multiclass_destroy(model);
  model = NULL;
  assert(cml_multiclass_load(path, &model) == CML_SUCCESS);
  assert(cml_multiclass_predict(model, &sample, &after) == CML_SUCCESS &&
         before == after && after == 0);
  cml_multiclass_destroy(model);
}

int main(void) {
  metrics();
  splits();
  char path[] = "/tmp/cml-model-test-XXXXXX";
  int file = mkstemp(path);
  assert(file >= 0 && close(file) == 0);
  regression_files(path);
  network_files(path);
  multiclass_file(path);
  assert(unlink(path) == 0);
  puts("Model workflow tests passed.");
}
