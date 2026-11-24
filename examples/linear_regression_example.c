#include "cml.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  const double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  const double y[] = {3.0, 5.0, 7.0, 9.0, 11.0};
  const size_t count = sizeof(x) / sizeof(x[0]);
  const double unseen = 10.0;
  CMLMatrix *features = cml_matrix_from_array(x, count, 1);
  CMLVector *targets = cml_vector_from_array(y, count);
  CMLVector *sample = cml_vector_from_array(&unseen, 1);
  CMLVector *weights = cml_vector_create(1);
  CMLLinearRegression *model = cml_linear_regression_create(1);

  double intercept, prediction;
  CMLStatus status = CML_ERROR_OUT_OF_MEMORY;
  if (features && targets && sample && weights && model) {
    status = cml_linear_regression_fit(model, features, targets);
    if (status == CML_SUCCESS) {
      status = cml_linear_regression_get_parameters(model, weights, &intercept);
    }
    if (status == CML_SUCCESS) {
      status = cml_linear_regression_predict(model, sample, &prediction);
    }
  }

  if (status == CML_SUCCESS) {
    printf("Fitted weight = %g, intercept = %g\n", weights->data[0], intercept);
    printf("Prediction for x = %g is %g\n", unseen, prediction);
  } else {
    fprintf(stderr, "Linear regression failed (%s)\n",
            cml_status_string(status));
  }

  cml_linear_regression_destroy(model);
  cml_vector_destroy(weights);
  cml_vector_destroy(sample);
  cml_vector_destroy(targets);
  cml_matrix_destroy(features);
  return status == CML_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
