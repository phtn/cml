#ifndef CML_NETWORK_INTERNAL_H
#define CML_NETWORK_INTERNAL_H

#include "cml/network.h"

typedef struct CMLDenseLayer {
  size_t inputs;
  CMLDenseConfig config;
  size_t parameter_offset;
  size_t unit_offset;
} CMLDenseLayer;

struct CMLNetwork {
  size_t inputs, layer_count, unit_count;
  CMLDenseLayer *layers;
  CMLVector *parameters;
};

CMLStatus cml_network_forward(const CMLNetwork *network, const double *params,
                              const double *input, double *values,
                              double *logits);

#endif
