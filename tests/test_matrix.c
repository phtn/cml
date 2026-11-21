
/* ==============================================================================
 * tests/test_matrix.c — Test suite for cml/matrix.h
 *
 * Covers:
 *   - Lifecycle & Memory Safety (create, create_filled, zeros, ones, identity,
 *     from_array, clone, resize, clear, destroy, overflow checks)
 *   - Element access & bounds checking (get, set, fill, swap, getters)
 *   - Matrix arithmetic (add, sub, mul, div, scale, add_scalar, negate,
 * in-place, aliasing, allocating helpers, division by zero)
 *   - Matrix multiplication (cache locality, square, non-square, 1x1, identity,
 *     aliasing out == a, empty dimensions)
 *   - Transpose (out-of-place, in-place square & rectangular, double transpose)
 *   - Row & column operations (get/set row/col with CMLVector, swap rows/cols)
 *   - Matrix-vector & vector-matrix products (correctness, aliasing)
 *   - Diagonal operations (get_diag, set_diag, from_diag, trace)
 *   - Statistics (sum, mean, min, max, Frobenius norm, row/col sums & means)
 *   - Properties (is_square, is_symmetric, is_diagonal, is_identity, is_zero,
 *     is_one, upper/lower triangular, orthogonality, equality, max_abs_diff)
 *   - Advanced (submatrix, horizontal/vertical concatenation, broadcasting)
 *   - Edge cases & error handling (NULL pointers, dimension mismatch, overflow)
 * ------------------------------------------------------------------------------
 */

#include "cml/math.h"
#include "cml/matrix.h"
#include "cml/vector.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int s_pass = 0;
static int s_fail = 0;

#define TEST(description, condition)                                           \
  do {                                                                         \
    if (condition) {                                                           \
      printf("  PASS  %s\n", description);                                     \
      ++s_pass;                                                                \
    } else {                                                                   \
      printf("  FAIL  %s  (line %d)\n", description, __LINE__);                \
      ++s_fail;                                                                \
    }                                                                          \
  } while (0)

static void section(const char *name) { printf("\n[%s]\n", name); }

/* ==============================================================================
 * A. Lifecycle Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_lifecycle(void) {
  section("matrix lifecycle");

  /* 0 x 0 matrix */
  CMLMatrix *m0 = cml_matrix_create(0, 0);
  TEST("create 0x0 returns valid pointer", m0 != NULL);
  if (m0) {
    TEST("0x0 matrix has rows == 0", m0->rows == 0);
    TEST("0x0 matrix has cols == 0", m0->cols == 0);
    TEST("0x0 matrix has data == NULL", m0->data == NULL);
    cml_matrix_destroy(m0);
  }

  /* 0 x 5 and 5 x 0 matrices */
  CMLMatrix *m05 = cml_matrix_create(0, 5);
  TEST("create 0x5 returns valid empty matrix",
       m05 != NULL && m05->rows == 0 && m05->cols == 0);
  cml_matrix_destroy(m05);

  CMLMatrix *m50 = cml_matrix_create(5, 0);
  TEST("create 5x0 returns valid empty matrix",
       m50 != NULL && m50->rows == 0 && m50->cols == 0);
  cml_matrix_destroy(m50);

  /* 1 x 1 matrix */
  CMLMatrix *m1 = cml_matrix_create(1, 1);
  TEST("create 1x1 returns valid pointer", m1 != NULL);
  if (m1) {
    TEST("1x1 initialized to 0.0", m1->data[0] == 0.0);
    cml_matrix_destroy(m1);
  }

  /* 2 x 3 matrix */
  CMLMatrix *m23 = cml_matrix_create(2, 3);
  TEST("create 2x3 returns valid pointer", m23 != NULL);
  if (m23) {
    TEST("2x3 rows == 2", m23->rows == 2);
    TEST("2x3 cols == 3", m23->cols == 3);
    bool all_zero = true;
    for (size_t i = 0; i < 6; ++i) {
      if (m23->data[i] != 0.0)
        all_zero = false;
    }
    TEST("2x3 all elements zero", all_zero);
    cml_matrix_destroy(m23);
  }

  /* 100 x 100 matrix */
  CMLMatrix *m100 = cml_matrix_create(100, 100);
  TEST("create 100x100 returns valid pointer", m100 != NULL);
  if (m100) {
    TEST("100x100 rows == 100 and cols == 100",
         m100->rows == 100 && m100->cols == 100);
    cml_matrix_destroy(m100);
  }

  /* create_filled */
  CMLMatrix *mf = cml_matrix_create_filled(2, 2, 7.5);
  TEST("create_filled 2x2 returns valid pointer", mf != NULL);
  if (mf) {
    TEST("create_filled elements match value",
         mf->data[0] == 7.5 && mf->data[3] == 7.5);
    cml_matrix_destroy(mf);
  }

  /* zeros and ones */
  CMLMatrix *mz = cml_matrix_zeros(3, 3);
  TEST("zeros creates valid matrix",
       mz != NULL && cml_matrix_is_zero(mz, CML_EPSILON));
  cml_matrix_destroy(mz);

  CMLMatrix *mo = cml_matrix_ones(3, 3);
  TEST("ones creates valid matrix",
       mo != NULL && cml_matrix_is_one(mo, CML_EPSILON));
  cml_matrix_destroy(mo);

  /* identity */
  CMLMatrix *id3 = cml_matrix_identity(3);
  TEST("identity(3) creates valid matrix",
       id3 != NULL && cml_matrix_is_identity(id3, CML_EPSILON));
  cml_matrix_destroy(id3);

  CMLMatrix *id0 = cml_matrix_identity(0);
  TEST("identity(0) creates valid empty matrix", id0 != NULL && id0->rows == 0);
  cml_matrix_destroy(id0);

  /* from_array */
  double raw[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  CMLMatrix *m_arr = cml_matrix_from_array(raw, 2, 3);
  TEST("from_array returns valid pointer", m_arr != NULL);
  if (m_arr) {
    TEST("from_array data matches",
         m_arr->data[0] == 1.0 && m_arr->data[5] == 6.0);
    cml_matrix_destroy(m_arr);
  }

  /* clone independence */
  CMLMatrix *orig = cml_matrix_from_array(raw, 2, 3);
  CMLMatrix *cloned = cml_matrix_clone(orig);
  TEST("clone returns valid pointer", cloned != NULL);
  if (cloned && orig) {
    TEST("clone dimensions match",
         cloned->rows == orig->rows && cloned->cols == orig->cols);
    TEST("clone data matches original",
         cml_matrix_is_equal(orig, cloned, CML_EPSILON));
    cml_matrix_set(cloned, 0, 0, 999.0);
    double val_orig = 0.0;
    cml_matrix_get(orig, 0, 0, &val_orig);
    TEST("modifying clone does not modify original", val_orig == 1.0);
    cml_matrix_destroy(cloned);
  }
  cml_matrix_destroy(orig);

  /* Resize 2x3 to 3x3 (preserve 2D contents, zeroes new) */
  CMLMatrix *m_res = cml_matrix_from_array(raw, 2, 3);
  CMLStatus s_res = cml_matrix_resize(m_res, 3, 3);
  TEST("resize 2x3 to 3x3 succeeds", s_res == CML_SUCCESS);
  if (m_res) {
    TEST("resize updated dimensions", m_res->rows == 3 && m_res->cols == 3);
    /* row 0: 1, 2, 3; row 1: 4, 5, 6; row 2: 0, 0, 0 */
    double v00, v02, v10, v12, v20, v22;
    cml_matrix_get(m_res, 0, 0, &v00);
    cml_matrix_get(m_res, 0, 2, &v02);
    cml_matrix_get(m_res, 1, 0, &v10);
    cml_matrix_get(m_res, 1, 2, &v12);
    cml_matrix_get(m_res, 2, 0, &v20);
    cml_matrix_get(m_res, 2, 2, &v22);
    TEST("resize preserved row 0", v00 == 1.0 && v02 == 3.0);
    TEST("resize preserved row 1", v10 == 4.0 && v12 == 6.0);
    TEST("resize zeroed new row 2", v20 == 0.0 && v22 == 0.0);

    /* Resize 3x3 to 1x2 (shrink) */
    s_res = cml_matrix_resize(m_res, 1, 2);
    TEST("resize 3x3 to 1x2 succeeds", s_res == CML_SUCCESS);
    cml_matrix_get(m_res, 0, 0, &v00);
    cml_matrix_get(m_res, 0, 1, &v02);
    TEST("resize 1x2 preserved elements", v00 == 1.0 && v02 == 2.0);

    /* Resize to zero */
    s_res = cml_matrix_resize(m_res, 0, 0);
    TEST("resize to 0x0 succeeds", s_res == CML_SUCCESS && m_res->rows == 0);
    cml_matrix_destroy(m_res);
  }

  /* Clear */
  CMLMatrix *mc = cml_matrix_create(4, 4);
  cml_matrix_clear(mc);
  TEST("clear resets rows and cols to 0", mc->rows == 0 && mc->cols == 0);
  cml_matrix_destroy(mc);

  /* Dimension overflow */
  CMLMatrix *m_ovf = cml_matrix_create(SIZE_MAX / 2, SIZE_MAX / 2);
  TEST("create with dimension overflow returns NULL", m_ovf == NULL);

  /* Safe destroy NULL */
  cml_matrix_destroy(NULL);
  TEST("cml_matrix_destroy(NULL) is safe", 1);
}

/* ==============================================================================
 * B. Element Access Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_access(void) {
  section("element access & mutation");

  double data[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
  CMLMatrix *m = cml_matrix_from_array(data, 2, 3);
  TEST("matrix setup", m != NULL);

  double val = 0.0;
  TEST("get A[0][0] == 10",
       cml_matrix_get(m, 0, 0, &val) == CML_SUCCESS && val == 10.0);
  TEST("get A[0][2] == 30",
       cml_matrix_get(m, 0, 2, &val) == CML_SUCCESS && val == 30.0);
  TEST("get A[1][0] == 40",
       cml_matrix_get(m, 1, 0, &val) == CML_SUCCESS && val == 40.0);
  TEST("get A[1][2] == 60",
       cml_matrix_get(m, 1, 2, &val) == CML_SUCCESS && val == 60.0);

  /* Out of bounds access */
  TEST("get row out of bounds",
       cml_matrix_get(m, 2, 0, &val) == CML_ERROR_INDEX_OUT_OF_BOUNDS);
  TEST("get col out of bounds",
       cml_matrix_get(m, 0, 3, &val) == CML_ERROR_INDEX_OUT_OF_BOUNDS);

  /* Set element */
  TEST("set A[1][1] to 99", cml_matrix_set(m, 1, 1, 99.0) == CML_SUCCESS);
  cml_matrix_get(m, 1, 1, &val);
  TEST("verify A[1][1] is 99", val == 99.0);
  TEST("set out of bounds",
       cml_matrix_set(m, 5, 5, 1.0) == CML_ERROR_INDEX_OUT_OF_BOUNDS);

  /* Fill */
  cml_matrix_fill(m, 42.0);
  bool all_42 = true;
  for (size_t i = 0; i < 6; ++i) {
    if (m->data[i] != 42.0)
      all_42 = false;
  }
  TEST("fill entire matrix with 42", all_42);

  /* Swap elements */
  cml_matrix_set(m, 0, 0, 1.0);
  cml_matrix_set(m, 1, 2, 9.0);
  cml_matrix_swap_elements(m, 0, 0, 1, 2);
  double v0, v1;
  cml_matrix_get(m, 0, 0, &v0);
  cml_matrix_get(m, 1, 2, &v1);
  TEST("swap two elements", v0 == 9.0 && v1 == 1.0);

  /* Swap element with itself */
  TEST("swap element with itself",
       cml_matrix_swap_elements(m, 0, 0, 0, 0) == CML_SUCCESS);

  /* Accessors */
  TEST("cml_matrix_rows accessor", cml_matrix_rows(m) == 2);
  TEST("cml_matrix_cols accessor", cml_matrix_cols(m) == 3);
  TEST("cml_matrix_data accessor", cml_matrix_data(m) == m->data);

  cml_matrix_destroy(m);
}

/* ==============================================================================
 * C. Matrix Arithmetic Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_arithmetic(void) {
  section("matrix arithmetic");

  double a_raw[] = {1.0, 2.0, 3.0, 4.0};
  double b_raw[] = {5.0, 6.0, 7.0, 8.0};
  CMLMatrix *a = cml_matrix_from_array(a_raw, 2, 2);
  CMLMatrix *b = cml_matrix_from_array(b_raw, 2, 2);
  CMLMatrix *out = cml_matrix_create(2, 2);

  /* Add: A + B */
  cml_matrix_add(a, b, out);
  TEST("matrix add", out->data[0] == 6.0 && out->data[1] == 8.0 &&
                         out->data[2] == 10.0 && out->data[3] == 12.0);

  /* Sub: B - A */
  cml_matrix_sub(b, a, out);
  TEST("matrix sub", out->data[0] == 4.0 && out->data[1] == 4.0 &&
                         out->data[2] == 4.0 && out->data[3] == 4.0);

  /* Mul: element-wise (Hadamard) */
  cml_matrix_mul(a, b, out);
  TEST("matrix hadamard mul", out->data[0] == 5.0 && out->data[1] == 12.0 &&
                                  out->data[2] == 21.0 && out->data[3] == 32.0);

  /* Div: element-wise */
  double num_raw[] = {10.0, 20.0, 30.0, 40.0};
  double den_raw[] = {2.0, 4.0, 5.0, 8.0};
  CMLMatrix *num = cml_matrix_from_array(num_raw, 2, 2);
  CMLMatrix *den = cml_matrix_from_array(den_raw, 2, 2);
  cml_matrix_div(num, den, out);
  TEST("matrix element-wise div", out->data[0] == 5.0 && out->data[1] == 5.0 &&
                                      out->data[2] == 6.0 &&
                                      out->data[3] == 5.0);

  /* Division by zero */
  cml_matrix_set(den, 0, 1, 0.0);
  TEST("division by zero returns CML_ERROR_NUMERICAL",
       cml_matrix_div(num, den, out) == CML_ERROR_NUMERICAL);
  cml_matrix_destroy(num);
  cml_matrix_destroy(den);

  /* Scale: A * 3 */
  cml_matrix_scale(a, 3.0, out);
  TEST("matrix scalar scale", out->data[0] == 3.0 && out->data[1] == 6.0 &&
                                  out->data[2] == 9.0 && out->data[3] == 12.0);

  /* Add scalar: A + 10 */
  cml_matrix_add_scalar(a, 10.0, out);
  TEST("matrix add scalar", out->data[0] == 11.0 && out->data[1] == 12.0 &&
                                out->data[2] == 13.0 && out->data[3] == 14.0);

  /* Negate: -A */
  cml_matrix_negate(a, out);
  TEST("matrix negate", out->data[0] == -1.0 && out->data[3] == -4.0);

  /* In-place & Aliased Add */
  CMLMatrix *a_copy = cml_matrix_clone(a);
  cml_matrix_add_inplace(a_copy, b);
  TEST("matrix add_inplace", a_copy->data[0] == 6.0 && a_copy->data[3] == 12.0);
  cml_matrix_destroy(a_copy);

  /* Safe aliasing: cml_matrix_add(a, b, a) */
  a_copy = cml_matrix_clone(a);
  cml_matrix_add(a_copy, b, a_copy);
  TEST("aliased matrix add(a, b, a)",
       a_copy->data[0] == 6.0 && a_copy->data[3] == 12.0);
  cml_matrix_destroy(a_copy);

  /* Allocating helper */
  CMLMatrix *add_res = cml_matrix_add_new(a, b);
  TEST("cml_matrix_add_new", add_res != NULL && add_res->data[0] == 6.0);
  cml_matrix_destroy(add_res);

  cml_matrix_destroy(a);
  cml_matrix_destroy(b);
  cml_matrix_destroy(out);
}

/* ==============================================================================
 * D. Matrix Multiplication Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_multiplication(void) {
  section("matrix multiplication");

  /*
   * Example from PHASE_3.md Section 6:
   * A (2 x 3):
   * [ 1, 2, 3 ]
   * [ 4, 5, 6 ]
   *
   * B (3 x 2):
   * [  7,  8 ]
   * [  9, 10 ]
   * [ 11, 12 ]
   *
   * Expected AB (2 x 2):
   * [  58,  64 ]
   * [ 139, 154 ]
   */
  double a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double b_data[] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  CMLMatrix *a = cml_matrix_from_array(a_data, 2, 3);
  CMLMatrix *b = cml_matrix_from_array(b_data, 3, 2);
  CMLMatrix *c = cml_matrix_create(2, 2);

  CMLStatus s = cml_matrix_matmul(a, b, c);
  TEST("matmul 2x3 by 3x2 status CML_SUCCESS", s == CML_SUCCESS);
  TEST("matmul result C[0][0] == 58", c->data[0] == 58.0);
  TEST("matmul result C[0][1] == 64", c->data[1] == 64.0);
  TEST("matmul result C[1][0] == 139", c->data[2] == 139.0);
  TEST("matmul result C[1][1] == 154", c->data[3] == 154.0);

  /* 3x1 multiplied by 1x3 -> 3x3 outer product */
  double u_data[] = {1.0, 2.0, 3.0};
  double v_data[] = {4.0, 5.0, 6.0};
  CMLMatrix *u = cml_matrix_from_array(u_data, 3, 1);
  CMLMatrix *v = cml_matrix_from_array(v_data, 1, 3);
  CMLMatrix *uv = cml_matrix_matmul_new(u, v);
  TEST("matmul 3x1 by 1x3 shape is 3x3",
       uv != NULL && uv->rows == 3 && uv->cols == 3);
  TEST("matmul 3x1 by 1x3 values",
       uv->data[0] == 4.0 && uv->data[4] == 10.0 && uv->data[8] == 18.0);
  cml_matrix_destroy(u);
  cml_matrix_destroy(v);
  cml_matrix_destroy(uv);

  /* 1x1 multiplication */
  CMLMatrix *m1 = cml_matrix_create_filled(1, 1, 3.0);
  CMLMatrix *m2 = cml_matrix_create_filled(1, 1, 7.0);
  CMLMatrix *m_prod = cml_matrix_matmul_new(m1, m2);
  TEST("matmul 1x1 by 1x1 == 21", m_prod != NULL && m_prod->data[0] == 21.0);
  cml_matrix_destroy(m1);
  cml_matrix_destroy(m2);
  cml_matrix_destroy(m_prod);

  /* Multiply by identity matrix: A * I = A */
  CMLMatrix *sq = cml_matrix_from_array(a_data, 2, 2);
  CMLMatrix *id = cml_matrix_identity(2);
  CMLMatrix *sq_res = cml_matrix_matmul_new(sq, id);
  TEST("matmul A * I == A", cml_matrix_is_equal(sq, sq_res, CML_EPSILON));
  cml_matrix_destroy(id);

  /* Aliasing test: square matmul out == a */
  CMLMatrix *sq_copy = cml_matrix_clone(sq);
  CMLStatus s_alias = cml_matrix_matmul(sq_copy, sq, sq_copy);
  TEST("aliased matmul(a, b, a) succeeds", s_alias == CML_SUCCESS);
  /* sq is [[1,2],[3,4]], sq^2 is [[7, 10],[15, 22]] */
  TEST("aliased matmul result matches expected",
       sq_copy->data[0] == 7.0 && sq_copy->data[1] == 10.0 &&
           sq_copy->data[2] == 15.0 && sq_copy->data[3] == 22.0);
  cml_matrix_destroy(sq_copy);
  cml_matrix_destroy(sq);
  cml_matrix_destroy(sq_res);

  /* Dimension mismatch */
  TEST("matmul dimension mismatch returns CML_ERROR_INVALID_DIMENSION",
       cml_matrix_matmul(a, a, c) == CML_ERROR_INVALID_DIMENSION);

  cml_matrix_destroy(a);
  cml_matrix_destroy(b);
  cml_matrix_destroy(c);
}

/* ==============================================================================
 * E. Transpose Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_transpose(void) {
  section("matrix transpose");

  double a_raw[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  CMLMatrix *a = cml_matrix_from_array(a_raw, 2, 3);
  CMLMatrix *at = cml_matrix_transpose_new(a);

  TEST("transpose 2x3 dimensions become 3x2",
       at != NULL && at->rows == 3 && at->cols == 2);
  /* Expected: [[1,4],[2,5],[3,6]] */
  TEST("transpose 2x3 values", at->data[0] == 1.0 && at->data[1] == 4.0 &&
                                   at->data[2] == 2.0 && at->data[3] == 5.0 &&
                                   at->data[4] == 3.0 && at->data[5] == 6.0);

  /* Double transpose: (A^T)^T == A */
  CMLMatrix *att = cml_matrix_transpose_new(at);
  TEST("double transpose (A^T)^T == A",
       cml_matrix_is_equal(a, att, CML_EPSILON));
  cml_matrix_destroy(att);
  cml_matrix_destroy(at);

  /* In-place square transpose */
  double sq_raw[] = {1.0, 2.0, 3.0, 4.0};
  CMLMatrix *sq = cml_matrix_from_array(sq_raw, 2, 2);
  cml_matrix_transpose_inplace(sq);
  TEST("in-place square transpose", sq->data[1] == 3.0 && sq->data[2] == 2.0);

  /* In-place rectangular transpose */
  CMLMatrix *rec = cml_matrix_from_array(a_raw, 2, 3);
  cml_matrix_transpose_inplace(rec);
  TEST("in-place rectangular transpose dimensions",
       rec->rows == 3 && rec->cols == 2);
  TEST("in-place rectangular transpose data",
       rec->data[1] == 4.0 && rec->data[4] == 3.0);
  cml_matrix_destroy(rec);
  cml_matrix_destroy(sq);

  /* Zero-sized transpose */
  CMLMatrix *m0 = cml_matrix_create(0, 0);
  CMLMatrix *m0_t = cml_matrix_transpose_new(m0);
  TEST("transpose zero-sized matrix", m0_t != NULL && m0_t->rows == 0);
  cml_matrix_destroy(m0);
  cml_matrix_destroy(m0_t);

  cml_matrix_destroy(a);
}

/* ==============================================================================
 * F. Row and Column Operations Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_rows_columns(void) {
  section("matrix rows & columns");

  double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  CMLMatrix *m = cml_matrix_from_array(data, 3, 3);

  /* Extract row 1: [4, 5, 6] */
  CMLVector *r1 = cml_matrix_get_row_new(m, 1);
  TEST("extract row 1 size == 3", r1 != NULL && r1->size == 3);
  TEST("extract row 1 values",
       r1->data[0] == 4.0 && r1->data[1] == 5.0 && r1->data[2] == 6.0);
  cml_vector_destroy(r1);

  /* Extract column 2: [3, 6, 9] */
  CMLVector *c2 = cml_matrix_get_col_new(m, 2);
  TEST("extract col 2 size == 3", c2 != NULL && c2->size == 3);
  TEST("extract col 2 values",
       c2->data[0] == 3.0 && c2->data[1] == 6.0 && c2->data[2] == 9.0);

  /* Set row 0 from vector c2: [3, 6, 9] */
  cml_matrix_set_row(m, 0, c2);
  double v0, v1, v2;
  cml_matrix_get(m, 0, 0, &v0);
  cml_matrix_get(m, 0, 1, &v1);
  cml_matrix_get(m, 0, 2, &v2);
  TEST("set row 0 from vector", v0 == 3.0 && v1 == 6.0 && v2 == 9.0);

  /* Set col 1 from vector */
  cml_matrix_set_col(m, 1, c2);
  cml_matrix_get(m, 0, 1, &v0);
  cml_matrix_get(m, 1, 1, &v1);
  cml_matrix_get(m, 2, 1, &v2);
  TEST("set col 1 from vector", v0 == 3.0 && v1 == 6.0 && v2 == 9.0);
  cml_vector_destroy(c2);

  /* Reset matrix and swap row 0 and row 2 */
  cml_matrix_destroy(m);
  m = cml_matrix_from_array(data, 3, 3);
  cml_matrix_swap_rows(m, 0, 2);
  cml_matrix_get(m, 0, 0, &v0);
  cml_matrix_get(m, 2, 0, &v2);
  TEST("swap row 0 and row 2", v0 == 7.0 && v2 == 1.0);

  /* Swap cols 0 and 2 */
  cml_matrix_swap_cols(m, 0, 2);
  cml_matrix_get(m, 0, 0, &v0);
  cml_matrix_get(m, 0, 2, &v2);
  TEST("swap col 0 and col 2", v0 == 9.0 && v2 == 7.0);

  /* Error handling: invalid row/col */
  CMLVector *v_dummy = cml_vector_create(3);
  TEST("get row out of bounds",
       cml_matrix_get_row(m, 10, v_dummy) == CML_ERROR_INDEX_OUT_OF_BOUNDS);
  TEST("get col out of bounds",
       cml_matrix_get_col(m, 10, v_dummy) == CML_ERROR_INDEX_OUT_OF_BOUNDS);
  cml_vector_destroy(v_dummy);

  cml_matrix_destroy(m);
}

/* ==============================================================================
 * G. Matrix-Vector Multiplication Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_vector_multiplication(void) {
  section("matrix-vector multiplication");

  /*
   * Example from PHASE_3.md Section 9:
   * A (2 x 3):
   * [ 1, 2, 3 ]
   * [ 4, 5, 6 ]
   *
   * x (3):
   * [ 2, 3, 4 ]
   *
   * Expected Ax (2):
   * [ 20, 47 ]
   */
  double a_raw[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double x_raw[] = {2.0, 3.0, 4.0};
  CMLMatrix *a = cml_matrix_from_array(a_raw, 2, 3);
  CMLVector *x = cml_vector_from_array(x_raw, 3);

  CMLVector *y = cml_matrix_vec_mul_new(a, x);
  TEST("matrix-vector mul result valid", y != NULL && y->size == 2);
  TEST("matrix-vector mul y[0] == 20", y->data[0] == 20.0);
  TEST("matrix-vector mul y[1] == 47", y->data[1] == 47.0);

  /* Vector-matrix multiplication: x^T * A
   * u (2): [1, 2]
   * A (2 x 3)
   * u^T * A (3): [ 1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6 ] = [9, 12, 15]
   */
  double u_raw[] = {1.0, 2.0};
  CMLVector *u = cml_vector_from_array(u_raw, 2);
  CMLVector *v = cml_vector_mat_mul_new(u, a);
  TEST("vector-matrix mul result valid", v != NULL && v->size == 3);
  TEST("vector-matrix mul values",
       v->data[0] == 9.0 && v->data[1] == 12.0 && v->data[2] == 15.0);

  /* Aliasing: matrix_vec_mul out == x for square matrix */
  CMLMatrix *sq = cml_matrix_from_array(a_raw, 2, 2);
  CMLVector *x2 = cml_vector_from_array(u_raw, 2);
  /* sq: [[1,2],[3,4]], x: [1,2] -> [5, 11] */
  cml_matrix_vec_mul(sq, x2, x2);
  TEST("aliased matrix-vector mul", x2->data[0] == 5.0 && x2->data[1] == 11.0);

  cml_matrix_destroy(sq);
  cml_vector_destroy(x2);
  cml_matrix_destroy(a);
  cml_vector_destroy(x);
  cml_vector_destroy(y);
  cml_vector_destroy(u);
  cml_vector_destroy(v);
}

/* ==============================================================================
 * H. Diagonal Operations Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_diagonal(void) {
  section("matrix diagonal operations");

  double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  CMLMatrix *m = cml_matrix_from_array(data, 3, 3);

  /* Extract diagonal: [1, 5, 9] */
  CMLVector *diag = cml_matrix_get_diag_new(m);
  TEST("extract diagonal size == 3", diag != NULL && diag->size == 3);
  TEST("extract diagonal values",
       diag->data[0] == 1.0 && diag->data[1] == 5.0 && diag->data[2] == 9.0);

  /* Trace: 1 + 5 + 9 = 15 */
  double tr = 0.0;
  cml_matrix_trace(m, &tr);
  TEST("trace == 15", tr == 15.0);

  /* Set diagonal: [10, 20, 30] */
  double new_d[] = {10.0, 20.0, 30.0};
  CMLVector *v_new = cml_vector_from_array(new_d, 3);
  cml_matrix_set_diag(m, v_new);
  cml_matrix_trace(m, &tr);
  TEST("set diagonal updates trace to 60", tr == 60.0);

  /* Create matrix from diagonal */
  CMLMatrix *m_from_d = cml_matrix_from_diag(v_new);
  TEST("from_diag creates diagonal matrix",
       cml_matrix_is_diagonal(m_from_d, CML_EPSILON));
  TEST("from_diag diagonal values", m_from_d->data[0] == 10.0 &&
                                        m_from_d->data[4] == 20.0 &&
                                        m_from_d->data[8] == 30.0);

  /* Rectangular diagonal */
  CMLMatrix *rec = cml_matrix_create(2, 4);
  cml_matrix_set(rec, 0, 0, 100.0);
  cml_matrix_set(rec, 1, 1, 200.0);
  CMLVector *rec_diag = cml_matrix_get_diag_new(rec);
  TEST("rectangular diagonal length is min(2, 4) == 2",
       rec_diag != NULL && rec_diag->size == 2);
  TEST("rectangular diagonal values",
       rec_diag->data[0] == 100.0 && rec_diag->data[1] == 200.0);

  cml_vector_destroy(rec_diag);
  cml_matrix_destroy(rec);
  cml_vector_destroy(diag);
  cml_vector_destroy(v_new);
  cml_matrix_destroy(m_from_d);
  cml_matrix_destroy(m);
}

/* ==============================================================================
 * I. Matrix Statistics Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_statistics(void) {
  section("matrix statistics");

  double data[] = {1.0, 2.0, 3.0, 4.0};
  CMLMatrix *m = cml_matrix_from_array(data, 2, 2);

  double s = 0.0, mean = 0.0, min_v = 0.0, max_v = 0.0, frob = 0.0;
  cml_matrix_sum(m, &s);
  cml_matrix_mean(m, &mean);
  cml_matrix_min(m, &min_v);
  cml_matrix_max(m, &max_v);
  cml_matrix_norm_frobenius(m, &frob);

  TEST("sum == 10", s == 10.0);
  TEST("mean == 2.5", mean == 2.5);
  TEST("min == 1", min_v == 1.0);
  TEST("max == 4", max_v == 4.0);
  TEST("Frobenius norm == sqrt(30)",
       cml_is_close(frob, sqrt(30.0), CML_EPSILON));

  /* Row sums and col sums */
  CMLVector *r_sums = cml_vector_create(2);
  CMLVector *c_sums = cml_vector_create(2);
  cml_matrix_row_sums(m, r_sums);
  cml_matrix_col_sums(m, c_sums);
  TEST("row sums: [3, 7]", r_sums->data[0] == 3.0 && r_sums->data[1] == 7.0);
  TEST("col sums: [4, 6]", c_sums->data[0] == 4.0 && c_sums->data[1] == 6.0);

  /* Row means and col means */
  CMLVector *r_means = cml_vector_create(2);
  CMLVector *c_means = cml_vector_create(2);
  cml_matrix_row_means(m, r_means);
  cml_matrix_col_means(m, c_means);
  TEST("row means: [1.5, 3.5]",
       r_means->data[0] == 1.5 && r_means->data[1] == 3.5);
  TEST("col means: [2.0, 3.0]",
       c_means->data[0] == 2.0 && c_means->data[1] == 3.0);

  cml_vector_destroy(r_sums);
  cml_vector_destroy(c_sums);
  cml_vector_destroy(r_means);
  cml_vector_destroy(c_means);
  cml_matrix_destroy(m);
}

/* ==============================================================================
 * J. Matrix Properties Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_properties(void) {
  section("matrix properties");

  CMLMatrix *id = cml_matrix_identity(3);
  TEST("identity is square", cml_matrix_is_square(id));
  TEST("identity is identity", cml_matrix_is_identity(id, CML_EPSILON));
  TEST("identity is symmetric", cml_matrix_is_symmetric(id, CML_EPSILON));
  TEST("identity is diagonal", cml_matrix_is_diagonal(id, CML_EPSILON));
  TEST("identity is orthogonal", cml_matrix_is_orthogonal(id, CML_EPSILON));
  TEST("identity is upper triangular",
       cml_matrix_is_upper_triangular(id, CML_EPSILON));
  TEST("identity is lower triangular",
       cml_matrix_is_lower_triangular(id, CML_EPSILON));
  cml_matrix_destroy(id);

  /* Symmetric matrix */
  double sym_data[] = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
  CMLMatrix *sym = cml_matrix_from_array(sym_data, 3, 3);
  TEST("symmetric matrix detected", cml_matrix_is_symmetric(sym, CML_EPSILON));
  cml_matrix_destroy(sym);

  /* Zero and One matrix */
  CMLMatrix *z = cml_matrix_zeros(2, 3);
  TEST("zero matrix detected", cml_matrix_is_zero(z, CML_EPSILON));
  cml_matrix_destroy(z);

  CMLMatrix *o = cml_matrix_ones(2, 3);
  TEST("one matrix detected", cml_matrix_is_one(o, CML_EPSILON));
  cml_matrix_destroy(o);

  /* Equality and Max Abs Diff */
  double d1[] = {1.0, 2.0, 3.0, 4.0};
  double d2[] = {1.0, 2.0, 3.0, 4.05};
  CMLMatrix *m1 = cml_matrix_from_array(d1, 2, 2);
  CMLMatrix *m2 = cml_matrix_from_array(d2, 2, 2);
  TEST("is_equal with tight tol is false", !cml_matrix_is_equal(m1, m2, 0.01));
  TEST("is_equal with loose tol is true", cml_matrix_is_equal(m1, m2, 0.1));

  double max_diff = 0.0;
  cml_matrix_max_abs_diff(m1, m2, &max_diff);
  TEST("max_abs_diff == 0.05", cml_is_close(max_diff, 0.05, CML_EPSILON));

  cml_matrix_destroy(m1);
  cml_matrix_destroy(m2);
}

/* ==============================================================================
 * K. Advanced Features Tests (Submatrix, Concatenation, Broadcasting)
 * ------------------------------------------------------------------------------
 */

static void test_matrix_advanced(void) {
  section("advanced matrix operations");

  /*
   * Submatrix example from PHASE_3.md Section 13:
   * 4x4 matrix, extract rows 1-2, cols 1-2 -> 2x2:
   * [  6,  7 ]
   * [ 10, 11 ]
   */
  double grid4x4[16];
  for (size_t i = 0; i < 16; ++i)
    grid4x4[i] = (double)(i + 1);
  CMLMatrix *big = cml_matrix_from_array(grid4x4, 4, 4);

  CMLMatrix *sub = cml_matrix_submatrix_new(big, 1, 2, 1, 2);
  TEST("submatrix extraction size is 2x2",
       sub != NULL && sub->rows == 2 && sub->cols == 2);
  TEST("submatrix values", sub->data[0] == 6.0 && sub->data[1] == 7.0 &&
                               sub->data[2] == 10.0 && sub->data[3] == 11.0);
  cml_matrix_destroy(sub);

  /* Horizontal Concatenation: [A | B] */
  double left_data[] = {1.0, 2.0, 3.0, 4.0}; /* 2x2 */
  double right_data[] = {5.0, 6.0};          /* 2x1 */
  CMLMatrix *left = cml_matrix_from_array(left_data, 2, 2);
  CMLMatrix *right = cml_matrix_from_array(right_data, 2, 1);
  CMLMatrix *h_cat = cml_matrix_concat_h_new(left, right);
  TEST("horizontal concat size is 2x3",
       h_cat != NULL && h_cat->rows == 2 && h_cat->cols == 3);
  TEST("horizontal concat values",
       h_cat->data[0] == 1.0 && h_cat->data[1] == 2.0 &&
           h_cat->data[2] == 5.0 && h_cat->data[3] == 3.0 &&
           h_cat->data[4] == 4.0 && h_cat->data[5] == 6.0);
  cml_matrix_destroy(h_cat);
  cml_matrix_destroy(right);

  /* Vertical Concatenation: [A / B] */
  double bot_data[] = {5.0, 6.0}; /* 1x2 */
  CMLMatrix *bot = cml_matrix_from_array(bot_data, 1, 2);
  CMLMatrix *v_cat = cml_matrix_concat_v_new(left, bot);
  TEST("vertical concat size is 3x2",
       v_cat != NULL && v_cat->rows == 3 && v_cat->cols == 2);
  TEST("vertical concat values",
       v_cat->data[4] == 5.0 && v_cat->data[5] == 6.0);
  cml_matrix_destroy(v_cat);
  cml_matrix_destroy(bot);
  cml_matrix_destroy(left);

  /* Broadcasting */
  CMLMatrix *mat2x2 = cml_matrix_create_filled(2, 2, 10.0);
  double v_row_d[] = {1.0, 2.0};
  CMLVector *v_row = cml_vector_from_array(v_row_d, 2);
  CMLMatrix *b_out = cml_matrix_create(2, 2);

  cml_matrix_add_row_vec(mat2x2, v_row, b_out);
  TEST("broadcast row addition",
       b_out->data[0] == 11.0 && b_out->data[1] == 12.0 &&
           b_out->data[2] == 11.0 && b_out->data[3] == 12.0);

  cml_matrix_add_col_vec(mat2x2, v_row, b_out);
  TEST("broadcast col addition",
       b_out->data[0] == 11.0 && b_out->data[1] == 11.0 &&
           b_out->data[2] == 12.0 && b_out->data[3] == 12.0);

  cml_vector_destroy(v_row);
  cml_matrix_destroy(b_out);
  cml_matrix_destroy(mat2x2);
  cml_matrix_destroy(big);
}

/* ==============================================================================
 * L. Edge Cases & Error Handling Tests
 * ------------------------------------------------------------------------------
 */

static void test_matrix_edge_cases(void) {
  section("matrix edge cases & error handling");

  double val = 0.0;
  TEST("get NULL mat",
       cml_matrix_get(NULL, 0, 0, &val) == CML_ERROR_INVALID_ARGUMENT);
  TEST("get NULL out", cml_matrix_get((CMLMatrix *)1, 0, 0, NULL) ==
                           CML_ERROR_INVALID_ARGUMENT);
  TEST("set NULL mat",
       cml_matrix_set(NULL, 0, 0, 1.0) == CML_ERROR_INVALID_ARGUMENT);
  TEST("fill NULL mat",
       cml_matrix_fill(NULL, 1.0) == CML_ERROR_INVALID_ARGUMENT);
  TEST("resize NULL mat",
       cml_matrix_resize(NULL, 2, 2) == CML_ERROR_INVALID_ARGUMENT);
  TEST("add NULL mats",
       cml_matrix_add(NULL, NULL, NULL) == CML_ERROR_INVALID_ARGUMENT);
  TEST("matmul NULL mats",
       cml_matrix_matmul(NULL, NULL, NULL) == CML_ERROR_INVALID_ARGUMENT);
  TEST("transpose NULL mat",
       cml_matrix_transpose(NULL, NULL) == CML_ERROR_INVALID_ARGUMENT);

  /* Mismatched dimensions */
  CMLMatrix *m22 = cml_matrix_create(2, 2);
  CMLMatrix *m33 = cml_matrix_create(3, 3);
  TEST("add dimension mismatch",
       cml_matrix_add(m22, m33, m22) == CML_ERROR_INVALID_DIMENSION);
  TEST("sub dimension mismatch",
       cml_matrix_sub(m22, m33, m22) == CML_ERROR_INVALID_DIMENSION);
  TEST("mul dimension mismatch",
       cml_matrix_mul(m22, m33, m22) == CML_ERROR_INVALID_DIMENSION);
  TEST("div dimension mismatch",
       cml_matrix_div(m22, m33, m22) == CML_ERROR_INVALID_DIMENSION);
  TEST("scale dimension mismatch",
       cml_matrix_scale(m22, 2.0, m33) == CML_ERROR_INVALID_DIMENSION);

  /* Statistics on empty matrix */
  CMLMatrix *m00 = cml_matrix_create(0, 0);
  TEST("mean of empty matrix returns error",
       cml_matrix_mean(m00, &val) == CML_ERROR_INVALID_DIMENSION);
  TEST("min of empty matrix returns error",
       cml_matrix_min(m00, &val) == CML_ERROR_INVALID_DIMENSION);
  TEST("max of empty matrix returns error",
       cml_matrix_max(m00, &val) == CML_ERROR_INVALID_DIMENSION);
  TEST("sum of empty matrix is 0.0",
       cml_matrix_sum(m00, &val) == CML_SUCCESS && val == 0.0);
  TEST("Frobenius norm of empty matrix is 0.0",
       cml_matrix_norm_frobenius(m00, &val) == CML_SUCCESS && val == 0.0);

  cml_matrix_destroy(m00);
  cml_matrix_destroy(m22);
  cml_matrix_destroy(m33);
}

/* ==============================================================================
 * main
 * ------------------------------------------------------------------------------
 */

int main(void) {
  printf("=== CML Matrix Engine Test Suite ===\n");

  test_matrix_lifecycle();
  test_matrix_access();
  test_matrix_arithmetic();
  test_matrix_multiplication();
  test_matrix_transpose();
  test_matrix_rows_columns();
  test_matrix_vector_multiplication();
  test_matrix_diagonal();
  test_matrix_statistics();
  test_matrix_properties();
  test_matrix_advanced();
  test_matrix_edge_cases();

  printf("\n--- Results: %d passed, %d failed ---\n", s_pass, s_fail);
  return (s_fail == 0) ? 0 : 1;
}
