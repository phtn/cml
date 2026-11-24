#ifndef CML_MULTICLASS_INTERNAL_H
#define CML_MULTICLASS_INTERNAL_H

#include "cml/multiclass.h"

struct CMLMulticlassClassifier {
  size_t features, classes;
  CMLMulticlassMethod method;
  CMLVector *parameters;
  bool trained;
};

#endif
