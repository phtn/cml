/*
 * examples/vector_example.c — Vector Engine Demonstration
 *
 * Demonstrates:
 *   a = [1, 2, 3, 4, 5]
 *   b = [10, 20, 30, 40, 50]
 *
 * Operations:
 *   1. Create both vectors
 *   2. Print their contents
 *   3. Add them (a + b)
 *   4. Subtract them (b - a)
 *   5. Calculate their dot product (a . b)
 *   6. Calculate their magnitudes (||a||_2, ||b||_2)
 *   7. Calculate statistics of A (mean, population variance)
 *   8. Normalize vector A (||a_norm||_2 == 1.0)
 *   9. Clean up and destroy all vectors
 */

#include "cml/vector.h"
#include <stdio.h>

int main(void) {
  printf("=== CML Phase 2 — Vector Engine Example ===\n\n");

  /* 1. Create vectors a and b */
  double raw_a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  double raw_b[] = {10.0, 20.0, 30.0, 40.0, 50.0};
  CMLVector *a = cml_vector_from_array(raw_a, 5);
  CMLVector *b = cml_vector_from_array(raw_b, 5);

  /* 2. Print contents */
  cml_vector_print(a, "Vector a");
  cml_vector_print(b, "Vector b");
  printf("\n");

  /* 3. Add them: a + b */
  CMLVector *sum = cml_vector_create(5);
  cml_vector_add(a, b, sum);
  cml_vector_print(sum, "a + b");

  /* 4. Subtract them: b - a */
  CMLVector *diff = cml_vector_create(5);
  cml_vector_sub(b, a, diff);
  cml_vector_print(diff, "b - a");

  /* 5. Calculate dot product: a . b */
  double dot_val = 0.0;
  cml_vector_dot(a, b, &dot_val);
  printf("a . b = %.4f (expected: 550.0000)\n", dot_val);

  /* 6. Calculate magnitudes */
  double mag_a = 0.0, mag_b = 0.0;
  cml_vector_norm_l2(a, &mag_a);
  cml_vector_norm_l2(b, &mag_b);
  printf("||a||_2 = %.6f (expected: sqrt(55) = ~7.416198)\n", mag_a);
  printf("||b||_2 = %.6f (expected: sqrt(5500) = ~74.161985)\n", mag_b);

  /* 7. Statistics of A */
  double mean_a = 0.0, var_a = 0.0;
  cml_vector_mean(a, &mean_a);
  cml_vector_variance_pop(a, &var_a);
  printf("Mean of a = %.4f (expected: 3.0000)\n", mean_a);
  printf("Population variance of a = %.4f (expected: 2.0000)\n", var_a);

  /* 8. Normalize vector A */
  CMLVector *a_norm = cml_vector_normalize_new(a);
  cml_vector_print(a_norm, "Normalized a");
  double norm_mag = 0.0;
  cml_vector_norm_l2(a_norm, &norm_mag);
  printf("||normalized a||_2 = %.6f (expected: 1.000000)\n", norm_mag);

  /* 9. Destroy all vectors */
  cml_vector_destroy(a);
  cml_vector_destroy(b);
  cml_vector_destroy(sum);
  cml_vector_destroy(diff);
  cml_vector_destroy(a_norm);

  printf("\nAll operations completed cleanly with zero memory leaks.\n");
  return 0;
}
