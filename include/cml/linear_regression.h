#ifndef CML_LINEAR_REGRESSION_H
#define CML_LINEAR_REGRESSION_H

#include "cml/matrix.h"
#include "cml/optimizer.h"

typedef struct CMLLinearRegression CMLLinearRegression;

/* Create an untrained model that owns its coefficients. Returns NULL for
 * zero features or failed allocation.
 */
CMLLinearRegression *cml_linear_regression_create(size_t num_features);

/* Destroy the model and its coefficients. NULL is safe. */
void cml_linear_regression_destroy(CMLLinearRegression *model);

/* Return whether a fit has succeeded. Returns false for NULL. */
bool cml_linear_regression_is_trained(const CMLLinearRegression *model);

/* Fit by ordinary least squares with an intercept. Each row of features is a
 * sample, and targets must contain one value per row. At least one sample is
 * required. A constant single feature yields a zero weight and the target
 * mean. Multiple features require a full-rank design with an intercept.
 * The model borrows the training inputs for this call. A failed fit preserves
 * the previous model state. Non-finite data or coefficients that cannot be
 * represented return CML_ERROR_NUMERICAL.
 */
CMLStatus cml_linear_regression_fit(CMLLinearRegression *model,
                                    const CMLMatrix *features,
                                    const CMLVector *targets);

/* Predict a single sample into out. An untrained model returns
 * CML_ERROR_INVALID_ARGUMENT. Non-finite data or predictions return
 * CML_ERROR_NUMERICAL. On error, out is unchanged.
 */
CMLStatus cml_linear_regression_predict(const CMLLinearRegression *model,
                                        const CMLVector *features, double *out);

/* Copy fitted coefficients into a caller-owned vector of the model's feature
 * count, and copy the intercept into a separate output. An untrained model
 * returns CML_ERROR_INVALID_ARGUMENT. On error, both outputs are unchanged.
 */
CMLStatus cml_linear_regression_get_parameters(const CMLLinearRegression *model,
                                               CMLVector *weights,
                                               double *intercept);

/* Compute the analytical MSE gradient at the supplied parameters. For a
 * matrix with d feature columns, parameters and out each contain d weights
 * followed by the intercept component. Each feature row has one target.
 * This operation accepts one or more rows and feature columns and requires
 * no fitted model. On error, out is unchanged. Non-finite data or arithmetic
 * return CML_ERROR_NUMERICAL.
 */
CMLStatus cml_linear_regression_mse_gradient(const CMLMatrix *features,
                                             const CMLVector *targets,
                                             const CMLVector *parameters,
                                             CMLVector *out);

/* Fit by minimizing MSE. A failed or cancelled run preserves the model. */
CMLStatus cml_linear_regression_fit_iterative(
    CMLLinearRegression *model, const CMLMatrix *features,
    const CMLVector *targets, const CMLMatrix *validation_features,
    const CMLVector *validation_targets, const CMLTrainingOptions *options,
    CMLTrainingResult *result);

size_t cml_linear_regression_feature_count(const CMLLinearRegression *model);
/* Install a copy of the fitted coefficients after validating all values. */
CMLStatus cml_linear_regression_set_parameters(CMLLinearRegression *model,
                                               const CMLVector *weights,
                                               double intercept);
CMLStatus cml_linear_regression_predict_batch(const CMLLinearRegression *model,
                                              const CMLMatrix *features,
                                              CMLVector *out);

#endif /* CML_LINEAR_REGRESSION_H */
