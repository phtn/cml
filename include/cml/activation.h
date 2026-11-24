#ifndef CML_ACTIVATION_H
#define CML_ACTIVATION_H

#include "cml/vector.h"

/* Logistic sigmoid. NaN propagates and negative and positive infinity map to
 * 0.0 and 1.0, respectively. Finite tails can round to either endpoint.
 */
double cml_sigmoid(double value);

/* Apply cml_sigmoid element by element to a caller-owned output vector of
 * equal size. Empty vectors are valid. Input and output can share the same
 * buffer, or use disjoint buffers. Missing vectors or storage return
 * CML_ERROR_INVALID_ARGUMENT, and inconsistent sizes return
 * CML_ERROR_INVALID_DIMENSION. Unaddressable sizes return CML_ERROR_OVERFLOW.
 * Outputs are unchanged on error. NaN and infinity follow the scalar rules.
 */
CMLStatus cml_vector_sigmoid(const CMLVector *input, CMLVector *out);

/* Apply cml_sigmoid in place. */
CMLStatus cml_vector_sigmoid_inplace(CMLVector *vec);

/* Stable softmax over a nonempty vector of finite logits. Allows in-place use.
 */
CMLStatus cml_softmax(const CMLVector *logits, CMLVector *out);

#endif /* CML_ACTIVATION_H */
