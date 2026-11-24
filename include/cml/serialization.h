#ifndef CML_SERIALIZATION_H
#define CML_SERIALIZATION_H

#include "cml/linear_regression.h"
#include "cml/logistic_regression.h"
#include "cml/multiclass.h"
#include "cml/network.h"

/* Save to a temporary file in the destination directory and replace the
 * destination after a complete write. Loaders return a new owned model;
 * on error they leave the output pointer unchanged.
 */
CMLStatus cml_linear_regression_save(const CMLLinearRegression *model,
                                     const char *path);
CMLStatus cml_linear_regression_load(const char *path,
                                     CMLLinearRegression **out);
CMLStatus cml_logistic_regression_save(const CMLLogisticRegression *model,
                                       const char *path);
CMLStatus cml_logistic_regression_load(const char *path,
                                       CMLLogisticRegression **out);
CMLStatus cml_multiclass_save(const CMLMulticlassClassifier *model,
                              const char *path);
CMLStatus cml_multiclass_load(const char *path, CMLMulticlassClassifier **out);
CMLStatus cml_network_save(const CMLNetwork *network, const char *path);
CMLStatus cml_network_load(const char *path, CMLNetwork **out);

/* Save a training candidate with the supplied network topology. The context
 * must be a CMLNetworkCheckpoint and may be passed directly to cml_train. */
typedef struct CMLNetworkCheckpoint {
  const CMLNetwork *network;
  const char *path;
} CMLNetworkCheckpoint;
CMLStatus cml_network_checkpoint(const CMLTrainingRecord *record,
                                 const CMLVector *parameters, void *context);

#endif
