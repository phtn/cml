/* ==============================================================================
 * tests/test_vector.c — Test suite for cml/vector.h
 *
 * Covers:
 *   - Lifecycle (create, create_filled, from_array, clone, resize, clear,
 * destroy)
 *   - Element access & bounds checking (get, set, fill, swap)
 *   - Vector arithmetic (add, sub, mul, div, scale, add_scalar, negate)
 *   - In-place operations & safe aliasing (out == a, out == b)
 *   - Advanced operations (dot product, L1/L2/inf norms, normalize)
 *   - Statistics (sum, mean, min, max, pop/sample variance, pop/sample stddev)
 *   - Edge cases & error handling (NULL, dimensions, zero-size,
 * division-by-zero)
 * ------------------------------------------------------------------------------
 */

#include "cml/math.h"
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

static void test_lifecycle(void) {
  section("vector lifecycle");

  /* Create 0-element vector */
  CMLVector *v0 = cml_vector_create(0);
  TEST("create 0 elements returns valid pointer", v0 != NULL);
  if (v0) {
    TEST("0-element vector has size 0", v0->size == 0);
    cml_vector_destroy(v0);
  }

  /* Create 1-element vector */
  CMLVector *v1 = cml_vector_create(1);
  TEST("create 1 element returns valid pointer", v1 != NULL);
  if (v1) {
    TEST("1-element vector has size 1", v1->size == 1);
    TEST("1-element vector initialized to 0", v1->data[0] == 0.0);
    cml_vector_destroy(v1);
  }

  /* Create 100-element vector */
  CMLVector *v100 = cml_vector_create(100);
  TEST("create 100 elements returns valid pointer", v100 != NULL);
  if (v100) {
    TEST("100-element vector has size 100", v100->size == 100);
    bool all_zero = true;
    for (size_t i = 0; i < 100; ++i) {
      if (v100->data[i] != 0.0) {
        all_zero = false;
        break;
      }
    }
    TEST("100-element vector all zeros", all_zero);
    cml_vector_destroy(v100);
  }

  /* Create filled vector */
  CMLVector *vf = cml_vector_create_filled(5, 7.5);
  TEST("create_filled returns valid pointer", vf != NULL);
  if (vf) {
    TEST("create_filled size is 5", vf->size == 5);
    bool all_7_5 = true;
    for (size_t i = 0; i < 5; ++i) {
      if (!cml_is_close(vf->data[i], 7.5, 1e-9)) {
        all_7_5 = false;
        break;
      }
    }
    TEST("create_filled elements match value", all_7_5);
    cml_vector_destroy(vf);
  }

  /* Create from raw array */
  double raw[] = {1.0, 2.0, 3.0, 4.0};
  CMLVector *va = cml_vector_from_array(raw, 4);
  TEST("from_array returns valid pointer", va != NULL);
  if (va) {
    TEST("from_array matches raw data",
         va->data[0] == 1.0 && va->data[3] == 4.0);
    cml_vector_destroy(va);
  }

  /* Clone vector & independence */
  CMLVector *orig = cml_vector_create_filled(4, 10.0);
  CMLVector *clone = cml_vector_clone(orig);
  TEST("clone returns valid pointer", clone != NULL);
  if (clone && orig) {
    TEST("clone matches original size", clone->size == orig->size);
    TEST("clone matches original contents", clone->data[0] == 10.0);
    /* Modify clone, verify original unchanged */
    clone->data[0] = 999.0;
    TEST("modifying clone does not modify original", orig->data[0] == 10.0);
    cml_vector_destroy(clone);
  }
  cml_vector_destroy(orig);

  /* Resize larger */
  CMLVector *vr = cml_vector_create(4);
  if (vr) {
    for (size_t i = 0; i < 4; ++i)
      vr->data[i] = (double)(i + 1);
    CMLStatus st = cml_vector_resize(vr, 8);
    TEST("resize larger succeeds", st == CML_SUCCESS);
    TEST("resize larger updates size", vr->size == 8);
    TEST("resize larger preserves existing data", vr->data[3] == 4.0);
    TEST("resize larger zeroes new elements",
         vr->data[4] == 0.0 && vr->data[7] == 0.0);

    /* Resize smaller */
    st = cml_vector_resize(vr, 2);
    TEST("resize smaller succeeds", st == CML_SUCCESS);
    TEST("resize smaller updates size", vr->size == 2);
    TEST("resize smaller preserves initial elements",
         vr->data[0] == 1.0 && vr->data[1] == 2.0);

    /* Resize to zero */
    st = cml_vector_resize(vr, 0);
    TEST("resize to zero succeeds", st == CML_SUCCESS);
    TEST("resize to zero sets size to 0", vr->size == 0);

    cml_vector_destroy(vr);
  }

  /* Clear */
  CMLVector *vc = cml_vector_create(10);
  if (vc) {
    CMLStatus st = cml_vector_clear(vc);
    TEST("clear succeeds", st == CML_SUCCESS);
    TEST("clear resets size to 0", vc->size == 0);
    cml_vector_destroy(vc);
  }

  /* Destroy NULL is safe */
  cml_vector_destroy(NULL);
  TEST("destroy NULL is safe", true);
}

/* ==============================================================================
 * B. Element Access & Mutation Tests
 * ------------------------------------------------------------------------------
 */

static void test_element_access(void) {
  section("element access & mutation");

  double init_data[] = {10.0, 20.0, 30.0, 40.0, 50.0};
  CMLVector *v = cml_vector_from_array(init_data, 5);
  TEST("setup vector [10, 20, 30, 40, 50]", v != NULL);

  if (v) {
    double val = 0.0;
    TEST("get element 0",
         cml_vector_get(v, 0, &val) == CML_SUCCESS && val == 10.0);
    TEST("get element 2",
         cml_vector_get(v, 2, &val) == CML_SUCCESS && val == 30.0);
    TEST("get element 4",
         cml_vector_get(v, 4, &val) == CML_SUCCESS && val == 50.0);
    TEST("get element 5 out of bounds",
         cml_vector_get(v, 5, &val) == CML_ERROR_INDEX_OUT_OF_BOUNDS);

    /* Set element */
    TEST("set element 2 to 99", cml_vector_set(v, 2, 99.0) == CML_SUCCESS);
    cml_vector_get(v, 2, &val);
    TEST("element 2 is now 99", val == 99.0);
    TEST("set out of bounds",
         cml_vector_set(v, 99, 1.0) == CML_ERROR_INDEX_OUT_OF_BOUNDS);

    /* Fill */
    TEST("fill vector with 5.0", cml_vector_fill(v, 5.0) == CML_SUCCESS);
    bool all_5 = true;
    for (size_t i = 0; i < 5; ++i) {
      if (v->data[i] != 5.0) {
        all_5 = false;
        break;
      }
    }
    TEST("all elements are 5.0", all_5);

    /* Swap */
    v->data[0] = 10.0;
    v->data[4] = 50.0;
    TEST("swap index 0 and 4", cml_vector_swap(v, 0, 4) == CML_SUCCESS);
    TEST("swapped correctly", v->data[0] == 50.0 && v->data[4] == 10.0);
    TEST("swap index with itself", cml_vector_swap(v, 2, 2) == CML_SUCCESS);
    TEST("swap out of bounds",
         cml_vector_swap(v, 0, 10) == CML_ERROR_INDEX_OUT_OF_BOUNDS);

    cml_vector_destroy(v);
  }
}

/* ==============================================================================
 * C. Arithmetic Operations Tests
 * ------------------------------------------------------------------------------
 */

static void test_arithmetic(void) {
  section("vector arithmetic");

  double raw_a[] = {1.0, 2.0, 3.0};
  double raw_b[] = {4.0, 5.0, 6.0};
  CMLVector *a = cml_vector_from_array(raw_a, 3);
  CMLVector *b = cml_vector_from_array(raw_b, 3);
  CMLVector *out = cml_vector_create(3);

  /* Addition: [1,2,3] + [4,5,6] = [5,7,9] */
  CMLStatus st = cml_vector_add(a, b, out);
  TEST("add status CML_SUCCESS", st == CML_SUCCESS);
  TEST("add result [5, 7, 9]",
       out->data[0] == 5.0 && out->data[1] == 7.0 && out->data[2] == 9.0);

  /* Subtraction: [5,7,9] - [1,2,3] = [4,5,6] */
  st = cml_vector_sub(out, a, b);
  TEST("sub result [4, 5, 6]",
       b->data[0] == 4.0 && b->data[1] == 5.0 && b->data[2] == 6.0);

  /* Element-wise multiplication: [1,2,3] * [4,5,6] = [4,10,18] */
  st = cml_vector_mul(a, b, out);
  TEST("mul result [4, 10, 18]",
       out->data[0] == 4.0 && out->data[1] == 10.0 && out->data[2] == 18.0);

  /* Element-wise division: [10,20,30] / [2,4,5] = [5,5,6] */
  double div_num[] = {10.0, 20.0, 30.0};
  double div_den[] = {2.0, 4.0, 5.0};
  CMLVector *num = cml_vector_from_array(div_num, 3);
  CMLVector *den = cml_vector_from_array(div_den, 3);
  st = cml_vector_div(num, den, out);
  TEST("div result [5, 5, 6]", st == CML_SUCCESS && out->data[0] == 5.0 &&
                                   out->data[1] == 5.0 && out->data[2] == 6.0);

  /* Division by zero */
  den->data[1] = 0.0;
  st = cml_vector_div(num, den, out);
  TEST("division by zero returns CML_ERROR_NUMERICAL",
       st == CML_ERROR_NUMERICAL);

  /* Scalar multiplication: 3 * [1,2,3] = [3,6,9] */
  st = cml_vector_scale(a, 3.0, out);
  TEST("scale result [3, 6, 9]", st == CML_SUCCESS && out->data[0] == 3.0 &&
                                     out->data[1] == 6.0 &&
                                     out->data[2] == 9.0);

  /* Scalar addition: [1,2,3] + 5 = [6,7,8] */
  st = cml_vector_add_scalar(a, 5.0, out);
  TEST("add_scalar result [6, 7, 8]",
       st == CML_SUCCESS && out->data[0] == 6.0 && out->data[1] == 7.0 &&
           out->data[2] == 8.0);

  /* Negation: -[1,2,3] = [-1,-2,-3] */
  st = cml_vector_negate(a, out);
  TEST("negate result [-1, -2, -3]",
       st == CML_SUCCESS && out->data[0] == -1.0 && out->data[1] == -2.0 &&
           out->data[2] == -3.0);

  /* In-place operations & aliasing */
  st = cml_vector_add_inplace(a, b);
  TEST("add_inplace works", st == CML_SUCCESS && a->data[0] == 5.0);

  /* Aliasing: vector_add(a, b, a) */
  st = cml_vector_add(a, b, a);
  TEST("aliased add vector_add(a, b, a) works", st == CML_SUCCESS);

  /* Allocating helpers */
  CMLVector *anew = cml_vector_add_new(b, b);
  TEST("add_new returns valid vector", anew != NULL && anew->data[0] == 8.0);
  cml_vector_destroy(anew);

  cml_vector_destroy(a);
  cml_vector_destroy(b);
  cml_vector_destroy(out);
  cml_vector_destroy(num);
  cml_vector_destroy(den);
}

/* ==============================================================================
 * D. Advanced Operations & Norms Tests
 * ------------------------------------------------------------------------------
 */

static void test_advanced(void) {
  section("dot product, norms & normalization");

  /* Dot product: [1,2,3] . [4,5,6] = 4 + 10 + 18 = 32 */
  double da[] = {1.0, 2.0, 3.0};
  double db[] = {4.0, 5.0, 6.0};
  CMLVector *a = cml_vector_from_array(da, 3);
  CMLVector *b = cml_vector_from_array(db, 3);
  double dot_val = 0.0;
  CMLStatus st = cml_vector_dot(a, b, &dot_val);
  TEST("dot product [1,2,3].[4,5,6] == 32",
       st == CML_SUCCESS && cml_is_close(dot_val, 32.0, 1e-9));

  /* Dot product with negative values */
  double neg_b[] = {-4.0, 5.0, -6.0};
  CMLVector *nb = cml_vector_from_array(neg_b, 3);
  st = cml_vector_dot(a, nb, &dot_val);
  TEST("dot product with negatives == -12",
       st == CML_SUCCESS && cml_is_close(dot_val, -12.0, 1e-9));
  cml_vector_destroy(nb);

  /* Norms: v = [-3, 4, -2] */
  double dv[] = {-3.0, 4.0, -2.0};
  CMLVector *v = cml_vector_from_array(dv, 3);
  double norm_val = 0.0;

  /* L1 norm = |-3| + |4| + |-2| = 9 */
  cml_vector_norm_l1(v, &norm_val);
  TEST("L1 norm == 9", cml_is_close(norm_val, 9.0, 1e-9));

  /* Infinity norm = max(3, 4, 2) = 4 */
  cml_vector_norm_inf(v, &norm_val);
  TEST("Infinity norm == 4", cml_is_close(norm_val, 4.0, 1e-9));
  cml_vector_destroy(v);

  /* L2 norm (magnitude): [3, 4] -> 5 */
  double d34[] = {3.0, 4.0};
  CMLVector *v34 = cml_vector_from_array(d34, 2);
  cml_vector_norm_l2(v34, &norm_val);
  TEST("L2 norm of [3, 4] == 5", cml_is_close(norm_val, 5.0, 1e-9));

  /* Normalization: [3, 4] / 5 -> [0.6, 0.8] */
  CMLVector *unit = cml_vector_create(2);
  st = cml_vector_normalize(v34, unit);
  TEST("normalize succeeds", st == CML_SUCCESS);
  TEST("unit vector is [0.6, 0.8]", cml_is_close(unit->data[0], 0.6, 1e-9) &&
                                        cml_is_close(unit->data[1], 0.8, 1e-9));

  /* Unit vector magnitude is 1.0 */
  double unit_mag = 0.0;
  cml_vector_norm_l2(unit, &unit_mag);
  TEST("unit vector magnitude is 1.0", cml_is_close(unit_mag, 1.0, 1e-9));

  /* Zero vector normalization returns CML_ERROR_NUMERICAL */
  CMLVector *zero_vec = cml_vector_create(3);
  st = cml_vector_normalize(zero_vec, zero_vec);
  TEST("zero-vector normalization returns CML_ERROR_NUMERICAL",
       st == CML_ERROR_NUMERICAL);

  cml_vector_destroy(zero_vec);
  cml_vector_destroy(unit);
  cml_vector_destroy(v34);
  cml_vector_destroy(a);
  cml_vector_destroy(b);
}

/* ==============================================================================
 * E. Statistics Tests
 * ------------------------------------------------------------------------------
 */

static void test_statistics(void) {
  section("vector statistics");

  /* Test vector from docs: [2, 4, 6, 8, 10] */
  double raw[] = {2.0, 4.0, 6.0, 8.0, 10.0};
  CMLVector *v = cml_vector_from_array(raw, 5);
  TEST("setup vector [2, 4, 6, 8, 10]", v != NULL);

  if (v) {
    double val = 0.0;

    /* Sum: 30 */
    cml_vector_sum(v, &val);
    TEST("sum == 30", cml_is_close(val, 30.0, 1e-9));

    /* Mean: 6 */
    cml_vector_mean(v, &val);
    TEST("mean == 6", cml_is_close(val, 6.0, 1e-9));

    /* Min: 2 */
    cml_vector_min(v, &val);
    TEST("min == 2", cml_is_close(val, 2.0, 1e-9));

    /* Max: 10 */
    cml_vector_max(v, &val);
    TEST("max == 10", cml_is_close(val, 10.0, 1e-9));

    /* Population variance: 8 */
    cml_vector_variance_pop(v, &val);
    TEST("population variance == 8", cml_is_close(val, 8.0, 1e-9));

    /* Population standard deviation: sqrt(8) */
    cml_vector_stddev_pop(v, &val);
    TEST("population stddev == sqrt(8)", cml_is_close(val, sqrt(8.0), 1e-9));

    /* Sample variance: 10 */
    cml_vector_variance_sample(v, &val);
    TEST("sample variance == 10", cml_is_close(val, 10.0, 1e-9));

    /* Sample standard deviation: sqrt(10) */
    cml_vector_stddev_sample(v, &val);
    TEST("sample stddev == sqrt(10)", cml_is_close(val, sqrt(10.0), 1e-9));

    cml_vector_destroy(v);
  }

  /* Single element sample variance returns error (df = 0) */
  CMLVector *single = cml_vector_create_filled(1, 42.0);
  double dummy = 0.0;
  CMLStatus st = cml_vector_variance_sample(single, &dummy);
  TEST("sample variance with 1 element returns CML_ERROR_INVALID_DIMENSION",
       st == CML_ERROR_INVALID_DIMENSION);
  cml_vector_destroy(single);
}

/* ==============================================================================
 * F. Edge Cases & Error Handling Tests
 * ------------------------------------------------------------------------------
 */

static void test_edge_cases(void) {
  section("edge cases & error handling");

  /* NULL arguments */
  double dummy = 0.0;
  TEST("dot with NULL a",
       cml_vector_dot(NULL, NULL, &dummy) == CML_ERROR_INVALID_ARGUMENT);
  TEST("get with NULL vec",
       cml_vector_get(NULL, 0, &dummy) == CML_ERROR_INVALID_ARGUMENT);
  TEST("get with NULL out",
       cml_vector_get(NULL, 0, NULL) == CML_ERROR_INVALID_ARGUMENT);
  TEST("set with NULL vec",
       cml_vector_set(NULL, 0, 1.0) == CML_ERROR_INVALID_ARGUMENT);
  TEST("resize with NULL vec",
       cml_vector_resize(NULL, 5) == CML_ERROR_INVALID_ARGUMENT);

  /* Dimension mismatches */
  CMLVector *v2 = cml_vector_create(2);
  CMLVector *v3 = cml_vector_create(3);
  CMLVector *out2 = cml_vector_create(2);

  TEST("add mismatched dimensions returns CML_ERROR_INVALID_DIMENSION",
       cml_vector_add(v2, v3, out2) == CML_ERROR_INVALID_DIMENSION);
  TEST("dot mismatched dimensions returns CML_ERROR_INVALID_DIMENSION",
       cml_vector_dot(v2, v3, &dummy) == CML_ERROR_INVALID_DIMENSION);

  /* Empty vector handling */
  CMLVector *empty = cml_vector_create(0);
  TEST("mean of empty vector returns CML_ERROR_INVALID_DIMENSION",
       cml_vector_mean(empty, &dummy) == CML_ERROR_INVALID_DIMENSION);
  TEST("min of empty vector returns CML_ERROR_INVALID_DIMENSION",
       cml_vector_min(empty, &dummy) == CML_ERROR_INVALID_DIMENSION);
  TEST("sum of empty vector is 0.0",
       cml_vector_sum(empty, &dummy) == CML_SUCCESS && dummy == 0.0);
  TEST("dot of empty vectors is 0.0",
       cml_vector_dot(empty, empty, &dummy) == CML_SUCCESS && dummy == 0.0);

  cml_vector_destroy(v2);
  cml_vector_destroy(v3);
  cml_vector_destroy(out2);
  cml_vector_destroy(empty);
}

/* ==============================================================================
 * main
 * ------------------------------------------------------------------------------
 */

int main(void) {
  printf("=== CML Vector Engine Test Suite ===\n");

  test_lifecycle();
  test_element_access();
  test_arithmetic();
  test_advanced();
  test_statistics();
  test_edge_cases();

  printf("\n--- Results: %d passed, %d failed ---\n", s_pass, s_fail);
  return (s_fail == 0) ? 0 : 1;
}
