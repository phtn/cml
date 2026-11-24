#ifndef CML_LOSS_H
#define CML_LOSS_H

#include "cml/vector.h"

/* Losses over paired, nonempty vectors of equal length. Inputs are borrowed.
 * NULL pointers return CML_ERROR_INVALID_ARGUMENT. Invalid sizes return
 * CML_ERROR_INVALID_DIMENSION. Non-finite data or a result that overflows
 * double return CML_ERROR_NUMERICAL. On error, out is unchanged.
 */

/* Sum of squared errors. */
CMLStatus cml_loss_sse(const CMLVector *actual, const CMLVector *predicted,
                       double *out);

/* Mean squared error. */
CMLStatus cml_loss_mse(const CMLVector *actual, const CMLVector *predicted,
                       double *out);

/* Root mean squared error. */
CMLStatus cml_loss_rmse(const CMLVector *actual, const CMLVector *predicted,
                        double *out);

/* Mean absolute error. */
CMLStatus cml_loss_mae(const CMLVector *actual, const CMLVector *predicted,
                       double *out);

/* BCE over exact binary labels. For probabilities in [0,1], epsilon in
 * (0,0.5] clips both tails. The logit variant accepts finite logits.
 */
CMLStatus cml_loss_bce(const CMLVector *actual, const CMLVector *probabilities,
                       double epsilon, double *out);
CMLStatus cml_loss_bce_logits(const CMLVector *actual, const CMLVector *logits,
                              double *out);

#endif /* CML_LOSS_H */
