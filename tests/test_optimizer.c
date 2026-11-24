#include "cml/linear_regression.h"
#include "cml/optimizer.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static CMLStatus objective(const CMLVector *p, void *context, double *out) {
  double error = p->data[0] - *(const double *)context;
  *out = error * error;
  return CML_SUCCESS;
}

static CMLStatus gradient(const CMLVector *p, void *context, CMLVector *out) {
  out->data[0] = 2.0 * (p->data[0] - *(const double *)context);
  return CML_SUCCESS;
}

static void quadratic(void) {
  double target = 4.0, value = -1.0;
  CMLVector p = {&value, 1, 1};
  CMLOptimizerConfig config = {
      .learning_rate = 0.2, .max_iterations = 200, .gradient_tolerance = 1e-8};
  CMLOptimizer *optimizer = cml_optimizer_create(&config);
  assert(optimizer != NULL);
  CMLGradientCheckResult check;
  assert(cml_gradient_check(&p, objective, gradient, &target, 1e-5, 1e-8, 1e-7,
                            &check) == CML_SUCCESS &&
         check.passed);
  CMLOptimizationResult result;
  assert(cml_optimizer_minimize(optimizer, &p, objective, gradient, &target,
                                &result) == CML_SUCCESS);
  assert(result.status == CML_OPTIMIZATION_CONVERGED);
  assert(fabs(value - target) < 1e-8);
  assert(result.initial_loss == 25.0 && result.final_loss < 1e-15);
  cml_optimizer_destroy(optimizer);

  config.learning_rate = 1e200;
  optimizer = cml_optimizer_create(&config);
  value = -1.0;
  assert(cml_optimizer_minimize(optimizer, &p, objective, gradient, &target,
                                &result) == CML_ERROR_NUMERICAL);
  assert(result.status == CML_OPTIMIZATION_DIVERGED && value == -1.0);
  cml_optimizer_destroy(optimizer);
}

typedef struct {
  size_t saved[5];
  size_t seen;
  double target;
} Batch;

static CMLStatus batch_loss(const CMLVector *p, const size_t *indices,
                            size_t count, void *context, double *loss,
                            CMLVector *out) {
  Batch *batch = context;
  if (indices != NULL)
    for (size_t i = 0; i < count && batch->seen < 5; ++i)
      batch->saved[batch->seen++] = indices[i];
  if (loss != NULL)
    objective(p, &batch->target, loss);
  if (out != NULL)
    gradient(p, &batch->target, out);
  return CML_SUCCESS;
}

static bool observe(const CMLTrainingRecord *record,
                    const CMLVector *parameters, void *context) {
  (void)parameters;
  size_t *count = context;
  ++*count;
  assert(record->epoch >= 1 && record->updates >= record->epoch);
  return true;
}

static bool cancel(const CMLTrainingRecord *record, const CMLVector *parameters,
                   void *context) {
  (void)record;
  (void)parameters;
  (void)context;
  return false;
}

static void batches(void) {
  CMLTrainingOptions options = cml_training_options_default();
  options.batch_size = 2;
  options.epochs = 1;
  options.seed = 44;
  options.shuffle = true;
  Batch first = {.target = 2.0}, second = {.target = 2.0};
  double a = 0.0, b = 0.0;
  CMLVector p = {&a, 1, 1}, q = {&b, 1, 1};
  CMLTrainingResult result;
  assert(cml_train(&p, batch_loss, 5, &first, 0, NULL, &options, &result) ==
         CML_SUCCESS);
  assert(result.optimization.iterations == 3 && first.seen == 5);
  assert(cml_train(&q, batch_loss, 5, &second, 0, NULL, &options, NULL) ==
             CML_SUCCESS &&
         a == b);
  size_t used = 0;
  for (size_t i = 0; i < first.seen; ++i) {
    assert(first.saved[i] < 5 && first.saved[i] == second.saved[i]);
    used |= (size_t)1 << first.saved[i];
  }
  assert(used == 31);

  options = cml_training_options_default();
  size_t observed = 0;
  options.observer = observe;
  options.observer_context = &observed;
  options.patience = 2;
  Batch validation = {.target = 0.0};
  a = 0.0;
  assert(cml_train(&p, batch_loss, 5, &first, 5, &validation, &options,
                   &result) == CML_SUCCESS);
  assert(a == 0.0 && result.best_epoch == 0 && result.epochs == 2 &&
         observed == 2);
  assert(result.validation_loss == 0.0);

  options.observer = cancel;
  assert(cml_train(&p, batch_loss, 5, &first, 0, NULL, &options, &result) ==
         CML_ERROR_CANCELLED);
  assert(a == 0.0 && result.optimization.status == CML_OPTIMIZATION_CANCELLED);
}

static void regression(void) {
  double x[] = {-1, -1, -1, 1, 1, -1, 1, 1};
  double y[] = {5, -1, 9, 3};
  CMLMatrix features = {x, 4, 2, 8};
  CMLVector targets = {y, 4, 4};
  CMLLinearRegression *closed = cml_linear_regression_create(2);
  CMLLinearRegression *trained = cml_linear_regression_create(2);
  assert(closed != NULL && trained != NULL);
  assert(cml_linear_regression_fit(closed, &features, &targets) == CML_SUCCESS);
  CMLTrainingOptions options = cml_training_options_default();
  options.optimizer.gradient_tolerance = 1e-9;
  options.momentum = 0.5;
  CMLTrainingResult result;
  assert(cml_linear_regression_fit_iterative(trained, &features, &targets, NULL,
                                             NULL, &options,
                                             &result) == CML_SUCCESS);
  assert(result.optimization.final_loss < 1e-12);
  double predicted[4];
  CMLVector out = {predicted, 4, 4};
  assert(cml_linear_regression_predict_batch(trained, &features, &out) ==
         CML_SUCCESS);
  for (size_t i = 0; i < 4; ++i)
    assert(fabs(predicted[i] - y[i]) < 1e-8);
  double w[2], bias;
  CMLVector weights = {w, 2, 2};
  assert(cml_linear_regression_get_parameters(closed, &weights, &bias) ==
         CML_SUCCESS);
  assert(fabs(w[0] - 2.0) < 1e-12 && fabs(w[1] + 3.0) < 1e-12 &&
         fabs(bias - 4.0) < 1e-12);
  for (size_t i = 0; i < 4; ++i)
    x[2 * i + 1] = x[2 * i];
  assert(cml_linear_regression_fit(closed, &features, &targets) ==
         CML_ERROR_NUMERICAL);
  cml_linear_regression_destroy(trained);
  cml_linear_regression_destroy(closed);
}

int main(void) {
  quadratic();
  batches();
  regression();
  puts("Optimizer tests passed.");
}
