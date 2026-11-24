#ifndef CML_LOGISTIC_REGRESSION_H
#define CML_LOGISTIC_REGRESSION_H

#include "cml/classification.h"
#include "cml/optimizer.h"

typedef struct CMLLogisticRegression CMLLogisticRegression;

typedef struct CMLLogisticOptions {
  CMLTrainingOptions training;
  double negative_weight;
  double positive_weight;
  double l1;
  double l2;
} CMLLogisticOptions;

CMLLogisticOptions cml_logistic_options_default(void);
CMLLogisticRegression *cml_logistic_regression_create(size_t features);
CMLLogisticRegression *
cml_logistic_regression_clone(const CMLLogisticRegression *model);
void cml_logistic_regression_destroy(CMLLogisticRegression *model);
CMLStatus cml_logistic_regression_reset(CMLLogisticRegression *model);
bool cml_logistic_regression_is_trained(const CMLLogisticRegression *model);
size_t
cml_logistic_regression_feature_count(const CMLLogisticRegression *model);

/* Parameters are the weights followed by the bias. Setting valid parameters
 * makes the model ready for prediction, and copies caller-owned storage.
 */
CMLStatus cml_logistic_regression_set_parameters(CMLLogisticRegression *model,
                                                 const CMLVector *parameters);
CMLStatus
cml_logistic_regression_get_parameters(const CMLLogisticRegression *model,
                                       CMLVector *out);
CMLStatus cml_logistic_regression_set_threshold(CMLLogisticRegression *model,
                                                double threshold);
double cml_logistic_regression_threshold(const CMLLogisticRegression *model);

/* Refit from current parameters. A failed or cancelled fit preserves them.
 * Class weights are finite and positive. Regularization applies to weights
 * only, with loss l1 * sum(|w|) + 0.5 * l2 * sum(w^2).
 */
CMLStatus cml_logistic_regression_fit(CMLLogisticRegression *model,
                                      const CMLDataset *training,
                                      const CMLDataset *validation,
                                      const CMLLogisticOptions *options,
                                      CMLTrainingResult *result);

/* Compute unregularized BCE or its analytical gradient at supplied parameters.
 */
CMLStatus cml_logistic_regression_bce(const CMLDataset *dataset,
                                      const CMLVector *parameters, double *out);
CMLStatus cml_logistic_regression_bce_gradient(const CMLDataset *dataset,
                                               const CMLVector *parameters,
                                               CMLVector *out);

/* Probability is for class 1. A tie with the threshold predicts class 1.
 * Batch outputs are caller owned and must have one element per feature row.
 */
CMLStatus
cml_logistic_regression_predict_proba(const CMLLogisticRegression *model,
                                      const CMLVector *features, double *out);
CMLStatus cml_logistic_regression_predict(const CMLLogisticRegression *model,
                                          const CMLVector *features,
                                          double *out);
CMLStatus
cml_logistic_regression_predict_proba_batch(const CMLLogisticRegression *model,
                                            const CMLMatrix *features,
                                            CMLVector *out);
CMLStatus
cml_logistic_regression_predict_batch(const CMLLogisticRegression *model,
                                      const CMLMatrix *features,
                                      CMLVector *out);

#endif
