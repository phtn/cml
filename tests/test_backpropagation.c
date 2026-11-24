#include "cml.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void check_gradients(void) {
  const CMLActivation activations[] = {
      CML_ACTIVATION_NONE, CML_ACTIVATION_RELU, CML_ACTIVATION_SIGMOID,
      CML_ACTIVATION_TANH, CML_ACTIVATION_SOFTMAX};
  double samples[] = {0.2, -1.0, 0.7, 0.3, -0.4, 0.6};
  double targets[] = {1, 0, 0, 1, 0.25, 0.75};
  CMLMatrix x = {samples, 3, 2, 6}, y = {targets, 3, 2, 6};
  for (size_t i = 0; i < sizeof(activations) / sizeof(*activations); ++i) {
    CMLDenseConfig layers[] = {{3, activations[i]},
                               {2, CML_ACTIVATION_SOFTMAX}};
    CMLNetwork *network = cml_network_create(2, layers, 2, 23);
    assert(network != NULL);
    CMLGradientCheckResult result;
    assert(cml_network_check_gradient(network, &x, &y, CML_NETWORK_MSE, 1e-5,
                                      1e-8, 1e-6, &result) == CML_SUCCESS);
    assert(result.passed);
    assert(cml_network_check_gradient(
               network, &x, &y, CML_NETWORK_CATEGORICAL_CROSS_ENTROPY, 1e-5,
               1e-8, 1e-6, &result) == CML_SUCCESS);
    assert(result.passed);
    cml_network_destroy(network);
  }
}

static void xor_training(void) {
  double samples[] = {0, 0, 0, 1, 1, 0, 1, 1};
  double targets[] = {0, 1, 1, 0};
  CMLMatrix x = {samples, 4, 2, 8}, y = {targets, 4, 1, 4};
  CMLDenseConfig layers[] = {{4, CML_ACTIVATION_TANH},
                             {1, CML_ACTIVATION_SIGMOID}};
  CMLNetwork *network = cml_network_create(2, layers, 2, 42);
  assert(network != NULL);
  CMLGradientCheckResult checked;
  assert(cml_network_check_gradient(network, &x, &y,
                                    CML_NETWORK_BINARY_CROSS_ENTROPY, 1e-5,
                                    1e-8, 1e-6, &checked) == CML_SUCCESS &&
         checked.passed);

  CMLTrainingOptions options = cml_training_options_default();
  options.epochs = 2000;
  options.optimizer.learning_rate = 0.35;
  CMLTrainingResult result;
  assert(cml_network_train(network, &x, &y, NULL, NULL,
                           CML_NETWORK_BINARY_CROSS_ENTROPY, &options,
                           &result) == CML_SUCCESS);
  assert(result.optimization.final_loss < 0.02);
  assert(result.optimization.final_loss < result.optimization.initial_loss);
  double predictions[4];
  CMLMatrix out = {predictions, 4, 1, 4};
  assert(cml_network_predict_batch(network, &x, &out) == CML_SUCCESS);
  for (size_t i = 0; i < 4; ++i)
    assert(fabs(predictions[i] - targets[i]) < 0.05);
  targets[3] = 0.5;
  assert(cml_network_train(network, &x, &y, NULL, NULL,
                           CML_NETWORK_BINARY_CROSS_ENTROPY, &options,
                           &result) == CML_ERROR_INVALID_ARGUMENT);
  cml_network_destroy(network);
}

int main(void) {
  check_gradients();
  xor_training();
  puts("Backpropagation tests passed.");
}
