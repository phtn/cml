#define _POSIX_C_SOURCE 200809L

#include "cml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
    return NAN;
  return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static bool compare(size_t rows) {
  CMLDataset *data = cml_dataset_create(rows, 1);
  if (data == NULL)
    return false;
  for (size_t i = 0; i < rows; ++i) {
    double x = ((double)(i % 149) - 74.0) / 74.0;
    data->X->data[i] = x;
    data->y->data[i] = 2.0 * x + 1.0;
  }
  const size_t batches[] = {0, 1, 8, 32, 64, 256};
  bool passed = true;
  for (size_t i = 0; i < sizeof(batches) / sizeof(*batches); ++i) {
    CMLLinearRegression *model = cml_linear_regression_create(1);
    CMLTrainingOptions options = cml_training_options_default();
    options.epochs = 3;
    options.optimizer.max_iterations = rows * options.epochs;
    options.batch_size = batches[i];
    CMLTrainingResult result;
    double start = now();
    CMLStatus status = cml_linear_regression_fit_iterative(
        model, data->X, data->y, NULL, NULL, &options, &result);
    double elapsed = now() - start;
    cml_linear_regression_destroy(model);
    if (status != CML_SUCCESS || !isfinite(elapsed)) {
      passed = false;
      break;
    }
    size_t batch = batches[i] == 0 || batches[i] > rows ? rows : batches[i];
    size_t index_bytes = batch == rows ? 0 : rows * sizeof(size_t);
    size_t dataset_bytes = rows * 2 * sizeof(double);
    printf("%8zu %7zu %9zu  %10.3f %11.3g %13zu %12zu\n", rows, batch,
           result.optimization.iterations, elapsed * 1e3,
           result.optimization.final_loss, dataset_bytes, index_bytes);
  }
  cml_dataset_destroy(data);
  return passed;
}

int main(int argc, char **argv) {
  bool stress = argc == 2 && strcmp(argv[1], "--stress") == 0;
  if (argc > 1 && !stress) {
    fprintf(stderr, "Usage: %s [--stress]\n", argv[0]);
    return EXIT_FAILURE;
  }
  puts(" Samples   Batch   Updates     Time ms   Final MSE  Dataset bytes  "
       "Index bytes");
  const size_t sizes[] = {10, 100, 1000, 10000, 100000, 1000000};
  size_t count = sizeof(sizes) / sizeof(*sizes) - (size_t)!stress;
  for (size_t i = 0; i < count; ++i)
    if (!compare(sizes[i]))
      return EXIT_FAILURE;
  return EXIT_SUCCESS;
}
