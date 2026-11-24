#define _POSIX_C_SOURCE 200809L

#include "cml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
    return NAN;
  return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void reference(const CMLMatrix *a, const CMLMatrix *b, CMLMatrix *out) {
  size_t n = a->rows;
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < n; ++k)
        sum += a->data[i * n + k] * b->data[k * n + j];
      out->data[i * n + j] = sum;
    }
}

static bool benchmark(size_t n) {
  bool passed = false;
  CMLMatrix *a = cml_matrix_create(n, n), *b = cml_matrix_create(n, n);
  CMLMatrix *expected = cml_matrix_create(n, n), *out = cml_matrix_create(n, n);
  if (a == NULL || b == NULL || expected == NULL || out == NULL)
    goto done;
  for (size_t i = 0; i < a->capacity; ++i) {
    a->data[i] = ((double)(i % 123) - 61.0) / 61.0;
    b->data[i] = ((double)(i % 97) - 48.0) / 48.0;
  }
  double direct_time = INFINITY, best_time = INFINITY;
  for (size_t repeat = 0; repeat < 3; ++repeat) {
    double start = now();
    reference(a, b, expected);
    direct_time = fmin(direct_time, now() - start);
    start = now();
    if (cml_matrix_matmul(a, b, out) != CML_SUCCESS)
      goto done;
    best_time = fmin(best_time, now() - start);
  }
  double max_error = 0.0;
  for (size_t i = 0; i < out->capacity; ++i) {
    double error = fabs(out->data[i] - expected->data[i]);
    if (!isfinite(error) || error > 1e-10 * fmax(1.0, fabs(expected->data[i])))
      goto done;
    max_error = fmax(max_error, error);
  }
  if (!isfinite(direct_time) || !isfinite(best_time) || best_time <= 0.0)
    goto done;
  printf("%4zu  %10.3f  %10.3f  %8.2fx  %9.2g\n", n, direct_time * 1e3,
         best_time * 1e3, direct_time / best_time, max_error);
  passed = true;

done:
  cml_matrix_destroy(out);
  cml_matrix_destroy(expected);
  cml_matrix_destroy(b);
  cml_matrix_destroy(a);
  return passed;
}

int main(void) {
#if defined(CML_CBLAS_ACCELERATE)
  puts("Matrix backend  Apple Accelerate");
#elif defined(CML_CBLAS_OPENBLAS)
  puts("Matrix backend  OpenBLAS");
#else
  puts("Matrix backend  CML blocked C11");
#endif
  puts("   N   direct ms      CML ms     speedup  max error");
  const size_t sizes[] = {64, 128, 256, 512};
  for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); ++i) {
    if (!benchmark(sizes[i])) {
      fprintf(stderr, "Matrix benchmark failed for size %zu\n", sizes[i]);
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
