#include "cml.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static CMLStatus quadratic(const CMLVector *parameters, const size_t *indices,
                           size_t count, void *context, double *loss,
                           CMLVector *gradient) {
  (void)indices;
  (void)count;
  (void)context;
  double residual = parameters->data[0] - 4.0;
  if (loss != NULL)
    *loss = residual * residual;
  if (gradient != NULL)
    gradient->data[0] = 2.0 * residual;
  return CML_SUCCESS;
}

static CMLStatus checkpoint(const CMLTrainingRecord *record,
                            const CMLVector *parameters, void *context) {
  size_t *calls = context;
  assert(record->best && record->epoch == *calls);
  assert(parameters->size == 1);
  ++*calls;
  return CML_SUCCESS;
}

static void update_formulas(void) {
  const CMLOptimizerMethod methods[] = {
      CML_OPTIMIZER_SGD, CML_OPTIMIZER_ADAGRAD, CML_OPTIMIZER_RMSPROP,
      CML_OPTIMIZER_ADAM};
  CMLTrainingOptions options = cml_training_options_default();
  options.beta2 = 0.75;
  options.epochs = 1;
  const double expected[] = {0.8, 0.8 / (8.0 + options.epsilon),
                             0.8 / (4.0 + options.epsilon),
                             0.8 / (8.0 + options.epsilon)};
  for (size_t i = 0; i < sizeof(methods) / sizeof(*methods); ++i) {
    options.method = methods[i];
    double value = 0.0;
    CMLVector p = {&value, 1, 1};
    assert(cml_train(&p, quadratic, 1, NULL, 0, NULL, &options, NULL) ==
           CML_SUCCESS);
    assert(fabs(value - expected[i]) < 1e-13);
  }

  size_t saved = 0;
  CMLTrainingHistory history = {0};
  options = cml_training_options_default();
  options.epochs = 5;
  options.clip_norm = 1.0;
  options.checkpoint = checkpoint;
  options.checkpoint_context = &saved;
  options.history = &history;
  options.record_every = 2;
  options.schedule = CML_RATE_STEP;
  options.decay = 0.5;
  options.decay_step = 2;
  double value = 0.0;
  CMLVector p = {&value, 1, 1};
  assert(cml_train(&p, quadratic, 1, NULL, 0, NULL, &options, NULL) ==
         CML_SUCCESS);
  assert(saved == 6 && history.size == 3);
  assert(history.records[0].epoch == 2 && history.records[1].epoch == 4 &&
         history.records[2].epoch == 5);
  assert(fabs(value - 0.325) < 1e-14);
  cml_training_history_destroy(&history);
  assert(history.records == NULL && history.size == 0);
}

static double train_xor(CMLOptimizerMethod method) {
  CMLDenseConfig layers[] = {{4, CML_ACTIVATION_TANH},
                             {1, CML_ACTIVATION_SIGMOID}};
  CMLNetwork *network = cml_network_create(2, layers, 2, 42);
  assert(network != NULL);
  double samples[] = {0, 0, 0, 1, 1, 0, 1, 1};
  double targets[] = {0, 1, 1, 0};
  CMLMatrix x = {samples, 4, 2, 8}, y = {targets, 4, 1, 4};
  CMLTrainingOptions options = cml_training_options_default();
  options.epochs = 1200;
  options.optimizer.learning_rate = method == CML_OPTIMIZER_ADAM ? 0.03 : 0.3;
  options.method = method;
  options.batch_size = 2;
  options.seed = 83;
  options.shuffle = true;
  CMLTrainingResult result;
  assert(cml_network_train(network, &x, &y, NULL, NULL,
                           CML_NETWORK_BINARY_CROSS_ENTROPY, &options,
                           &result) == CML_SUCCESS);
  assert(result.optimization.final_loss < 0.04);
  cml_network_destroy(network);
  return result.optimization.final_loss;
}

int main(void) {
  update_formulas();
  double sgd = train_xor(CML_OPTIMIZER_SGD),
         adam = train_xor(CML_OPTIMIZER_ADAM);
  printf("Adaptive optimizer tests passed. XOR SGD %.6f, Adam %.6f\n", sgd,
         adam);
}
