#include "cml.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  double samples[] = {0, 0, 0, 1, 1, 0, 1, 1}, targets[] = {0, 1, 1, 0};
  CMLMatrix x = {samples, 4, 2, 8}, y = {targets, 4, 1, 4};
  CMLDenseConfig layers[] = {{4, CML_ACTIVATION_TANH},
                             {1, CML_ACTIVATION_SIGMOID}};
  CMLNetwork *network = cml_network_create(2, layers, 2, 42);
  if (network == NULL)
    return EXIT_FAILURE;
  CMLTrainingOptions options = cml_training_options_default();
  options.epochs = 2000;
  options.optimizer.learning_rate = 0.35;
  CMLTrainingResult result;
  CMLStatus status =
      cml_network_train(network, &x, &y, NULL, NULL,
                        CML_NETWORK_BINARY_CROSS_ENTROPY, &options, &result);
  double predictions[4];
  CMLMatrix out = {predictions, 4, 1, 4};
  if (status == CML_SUCCESS)
    status = cml_network_predict_batch(network, &x, &out);
  if (status == CML_SUCCESS) {
    printf("XOR loss %.6f -> %.6f\n", result.optimization.initial_loss,
           result.optimization.final_loss);
    for (size_t i = 0; i < 4; ++i)
      printf("%.0f XOR %.0f = %d (p=%.6f)\n", samples[2 * i],
             samples[2 * i + 1], predictions[i] >= 0.5, predictions[i]);
  } else {
    fprintf(stderr, "XOR training failed (%s)\n", cml_status_string(status));
  }
  cml_network_destroy(network);
  return status == CML_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
