#ifndef CML_MULTICLASS_H
#define CML_MULTICLASS_H

#include "cml/classification.h"
#include "cml/optimizer.h"

/* Targets are integer-valued doubles in [0, classes). */
CMLStatus cml_classification_validate_classes(const CMLDataset *data,
                                              size_t classes);

typedef enum CMLMulticlassMethod {
  CML_MULTICLASS_SOFTMAX,
  CML_MULTICLASS_ONE_VS_REST,
} CMLMulticlassMethod;

typedef struct CMLMulticlassClassifier CMLMulticlassClassifier;

CMLMulticlassClassifier *cml_multiclass_create(size_t features, size_t classes,
                                               CMLMulticlassMethod method);
void cml_multiclass_destroy(CMLMulticlassClassifier *model);
/* The same options apply to all OvR fits. OvR observers see each fit separately
 * and the result describes the last binary fit.
 */
CMLStatus cml_multiclass_fit(CMLMulticlassClassifier *model,
                             const CMLDataset *training,
                             const CMLDataset *validation,
                             const CMLTrainingOptions *options,
                             CMLTrainingResult *result);
/* Softmax scores sum to one. One-vs-rest scores are independent probabilities.
 */
CMLStatus cml_multiclass_predict_scores(const CMLMulticlassClassifier *model,
                                        const CMLVector *features,
                                        CMLVector *out);
CMLStatus cml_multiclass_predict(const CMLMulticlassClassifier *model,
                                 const CMLVector *features, size_t *out);

#endif
