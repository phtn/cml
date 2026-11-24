#include "cml.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

static CMLStatus objective(const CMLVector *p, void *data, double *loss) {
  return cml_logistic_regression_bce(data, p, loss);
}
static CMLStatus derivative(const CMLVector *p, void *data, CMLVector *out) {
  return cml_logistic_regression_bce_gradient(data, p, out);
}

static void binary(void) {
  double x[] = {1, 2, 3, 7, 8, 9}, y[] = {0, 0, 0, 1, 1, 1};
  CMLMatrix features = {x, 6, 1, 6};
  CMLVector labels = {y, 6, 6};
  CMLDataset data = {
      .X = &features, .y = &labels, .num_samples = 6, .num_features = 1};
  CMLLogisticRegression *model = cml_logistic_regression_create(1);
  assert(model != NULL);
  double coefficients[] = {2, -4}, sample_value = 2, prediction;
  CMLVector p = {coefficients, 2, 2}, sample = {&sample_value, 1, 1};
  assert(cml_logistic_regression_predict(model, &sample, &prediction) ==
         CML_ERROR_INVALID_ARGUMENT);
  assert(cml_logistic_regression_set_parameters(model, &p) == CML_SUCCESS);
  assert(cml_logistic_regression_predict_proba(model, &sample, &prediction) ==
             CML_SUCCESS &&
         prediction == 0.5);
  assert(cml_logistic_regression_predict(model, &sample, &prediction) ==
             CML_SUCCESS &&
         prediction == 1.0);
  assert(cml_logistic_regression_set_threshold(model, 0.6) == CML_SUCCESS);
  assert(cml_logistic_regression_predict(model, &sample, &prediction) ==
             CML_SUCCESS &&
         prediction == 0.0);

  CMLGradientCheckResult gradient_check;
  assert(cml_gradient_check(&p, objective, derivative, &data, 1e-5, 1e-7, 1e-6,
                            &gradient_check) == CML_SUCCESS);
  assert(gradient_check.passed);
  assert(cml_logistic_regression_reset(model) == CML_SUCCESS);
  CMLLogisticOptions opts = cml_logistic_options_default();
  opts.training.optimizer.learning_rate = 0.05;
  opts.training.epochs = 3000;
  opts.training.shuffle = true;
  opts.training.seed = 7;
  opts.training.batch_size = 2;
  opts.training.momentum = 0.3;
  opts.l2 = 0.001;
  CMLTrainingResult result;
  assert(cml_logistic_regression_fit(model, &data, NULL, &opts, &result) ==
         CML_SUCCESS);
  assert(result.optimization.final_loss < 0.1);
  sample_value = 4;
  assert(cml_logistic_regression_predict(model, &sample, &prediction) ==
             CML_SUCCESS &&
         prediction == 0.0);
  sample_value = 6;
  assert(cml_logistic_regression_predict(model, &sample, &prediction) ==
             CML_SUCCESS &&
         prediction == 1.0);
  CMLLogisticRegression *clone = cml_logistic_regression_clone(model);
  assert(clone != NULL);
  double scores[6];
  CMLVector out = {scores, 6, 6};
  assert(cml_logistic_regression_predict_batch(clone, &features, &out) ==
         CML_SUCCESS);
  CMLBinaryMetrics metrics;
  assert(cml_binary_metrics(&labels, &out, &metrics) == CML_SUCCESS);
  assert(metrics.accuracy == 1.0 && metrics.confusion.true_positive == 3);
  assert(cml_logistic_regression_predict_proba_batch(clone, &features, &out) ==
         CML_SUCCESS);
  CMLRocCurve *curve = NULL;
  assert(cml_roc_curve(&labels, &out, &curve) == CML_SUCCESS);
  assert(curve->auc == 1.0 && curve->best_f1 == 1.0);
  cml_roc_curve_destroy(curve);

  y[0] = NAN;
  assert(cml_logistic_regression_fit(model, &data, NULL, &opts, &result) ==
             CML_ERROR_NUMERICAL &&
         cml_logistic_regression_is_trained(model));
  cml_logistic_regression_destroy(clone);
  cml_logistic_regression_destroy(model);
}

static void numeric_cases(void) {
  double y[] = {0.0, 1.0, 0.0, 1.0}, p[] = {0.5, 0.5, 0.5, 0.5};
  CMLVector labels = {y, 4, 4}, scores = {p, 4, 4};
  CMLRocCurve *curve = NULL;
  assert(cml_roc_curve(&labels, &scores, &curve) == CML_SUCCESS);
  assert(curve->size == 2 && curve->auc == 0.5);
  cml_roc_curve_destroy(curve);
  double loss;
  assert(cml_loss_bce(&labels, &scores, DBL_TRUE_MIN, &loss) == CML_SUCCESS);
  assert(fabs(loss - log(2.0)) < 1e-15);
  p[0] = DBL_MAX;
  p[1] = -DBL_MAX;
  p[2] = DBL_MAX;
  p[3] = -DBL_MAX;
  assert(cml_loss_bce_logits(&labels, &scores, &loss) == CML_SUCCESS &&
         loss == DBL_MAX);
  p[0] = 1;
  p[1] = 0;
  p[2] = 1;
  p[3] = 0;
  assert(cml_loss_bce(&labels, &scores, DBL_TRUE_MIN, &loss) == CML_SUCCESS &&
         isfinite(loss) && loss > 700.0);
}

static void multiclass(void) {
  double x[] = {-2, 0, -3, 0, 2, 2, 3, 3, 0, -2, 0, -3};
  double y[] = {0, 0, 1, 1, 2, 2};
  CMLMatrix features = {x, 6, 2, 12};
  CMLVector labels = {y, 6, 6};
  CMLDataset data = {
      .X = &features, .y = &labels, .num_samples = 6, .num_features = 2};
  CMLTrainingOptions opts = cml_training_options_default();
  const CMLMulticlassMethod methods[] = {CML_MULTICLASS_SOFTMAX,
                                         CML_MULTICLASS_ONE_VS_REST};
  for (size_t m = 0; m < 2; ++m) {
    CMLMulticlassClassifier *model = cml_multiclass_create(2, 3, methods[m]);
    assert(model != NULL);
    assert(cml_multiclass_fit(model, &data, NULL, &opts, NULL) == CML_SUCCESS);
    for (size_t i = 0; i < 6; ++i) {
      CMLVector row = {x + 2 * i, 2, 2};
      size_t actual;
      assert(cml_multiclass_predict(model, &row, &actual) == CML_SUCCESS);
      assert(actual == (size_t)y[i]);
      double scores[3];
      CMLVector out = {scores, 3, 3};
      assert(cml_multiclass_predict_scores(model, &row, &out) == CML_SUCCESS);
      assert(scores[actual] > 0.9);
      if (methods[m] == CML_MULTICLASS_SOFTMAX)
        assert(fabs(scores[0] + scores[1] + scores[2] - 1.0) < 1e-14);
    }
    cml_multiclass_destroy(model);
  }
}

int main(void) {
  binary();
  numeric_cases();
  multiclass();
  puts("Logistic regression tests passed.");
}
