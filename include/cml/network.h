#ifndef CML_NETWORK_H
#define CML_NETWORK_H

#include "cml/matrix.h"

#include <stdint.h>

typedef enum CMLActivation {
  CML_ACTIVATION_NONE,
  CML_ACTIVATION_RELU,
  CML_ACTIVATION_SIGMOID,
  CML_ACTIVATION_TANH,
  CML_ACTIVATION_SOFTMAX,
} CMLActivation;

typedef struct CMLDenseConfig {
  size_t units;
  CMLActivation activation;
} CMLDenseConfig;

typedef struct CMLNetwork CMLNetwork;

/* Copy the topology and initialize weights reproducibly from seed. */
CMLNetwork *cml_network_create(size_t inputs, const CMLDenseConfig *layers,
                               size_t count, uint64_t seed);
void cml_network_destroy(CMLNetwork *network);
size_t cml_network_input_size(const CMLNetwork *network);
size_t cml_network_output_size(const CMLNetwork *network);
size_t cml_network_layer_count(const CMLNetwork *network);
size_t cml_network_parameter_count(const CMLNetwork *network);
CMLStatus cml_network_layer_config(const CMLNetwork *network, size_t layer,
                                   CMLDenseConfig *out);

/* For each layer in order, store weights in output-major order, then biases. */
CMLStatus cml_network_get_parameters(const CMLNetwork *network, CMLVector *out);
CMLStatus cml_network_set_parameters(CMLNetwork *network,
                                     const CMLVector *values);
CMLStatus cml_network_predict(const CMLNetwork *network, const CMLVector *input,
                              CMLVector *out);
CMLStatus cml_network_predict_batch(const CMLNetwork *network,
                                    const CMLMatrix *inputs, CMLMatrix *out);

#endif
