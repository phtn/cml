/*
 * examples/matrix_example.c — Matrix Engine Demonstration
 *
 * Demonstrates:
 *   A (2 x 3) = [[1, 2, 3],
 *                [4, 5, 6]]
 *
 *   B (3 x 2) = [[7,  8],
 *                [9,  10],
 *                [11, 12]]
 *
 * Operations:
 *   1. Create matrices A and B
 *   2. Print their contents
 *   3. Matrix multiplication: C = A * B (2 x 2)
 *   4. Matrix transpose: A^T (3 x 2)
 *   5. Matrix-vector product: y = A * x, with x = [2, 3, 4]^T
 *   6. Matrix statistics: trace, Frobenius norm, sum, mean
 *   7. In-place transpose of C
 *   8. Clean up and destroy all allocated structures
 */

#include "cml/matrix.h"
#include "cml/vector.h"

#include <stdio.h>

int main(void) {
  printf("=== CML Phase 3 — Matrix Engine Example ===\n\n");

  /* 1. Create matrix A (2 x 3) and B (3 x 2) */
  double raw_a[] = {1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0};
  double raw_b[] = {7.0,  8.0,
                    9.0,  10.0,
                    11.0, 12.0};

  CMLMatrix *a = cml_matrix_from_array(raw_a, 2, 3);
  CMLMatrix *b = cml_matrix_from_array(raw_b, 3, 2);

  /* 2. Print matrices */
  cml_matrix_print(a, "Matrix A");
  printf("\n");
  cml_matrix_print(b, "Matrix B");
  printf("\n");

  /* 3. Matrix Multiplication: C = A * B (2 x 2) */
  CMLMatrix *c = cml_matrix_matmul_new(a, b);
  cml_matrix_print(c, "C = A * B");
  printf("\n");

  /* 4. Transpose of A: A^T (3 x 2) */
  CMLMatrix *at = cml_matrix_transpose_new(a);
  cml_matrix_print(at, "A^T");
  printf("\n");

  /* 5. Matrix-Vector Multiplication: y = A * x */
  double raw_x[] = {2.0, 3.0, 4.0};
  CMLVector *x = cml_vector_from_array(raw_x, 3);
  cml_vector_print(x, "Vector x");

  CMLVector *y = cml_matrix_vec_mul_new(a, x);
  cml_vector_print(y, "y = A * x");
  printf("\n");

  /* 6. Matrix Statistics of C */
  double tr = 0.0, frob = 0.0, total_sum = 0.0, mean_val = 0.0;
  cml_matrix_trace(c, &tr);
  cml_matrix_norm_frobenius(c, &frob);
  cml_matrix_sum(c, &total_sum);
  cml_matrix_mean(c, &mean_val);

  printf("Properties of C:\n");
  printf("  Trace:          %.4f (expected: 212.0000)\n", tr);
  printf("  Frobenius norm: %.4f\n", frob);
  printf("  Sum:            %.4f (expected: 415.0000)\n", total_sum);
  printf("  Mean:           %.4f (expected: 103.7500)\n", mean_val);
  printf("\n");

  /* 7. In-place transpose of C */
  printf("Performing in-place transpose on C...\n");
  cml_matrix_transpose_inplace(c);
  cml_matrix_print(c, "C^T (in-place)");
  printf("\n");

  /* 8. Clean up */
  cml_matrix_destroy(a);
  cml_matrix_destroy(b);
  cml_matrix_destroy(c);
  cml_matrix_destroy(at);
  cml_vector_destroy(x);
  cml_vector_destroy(y);

  printf("Example completed successfully.\n");
  return 0;
}
