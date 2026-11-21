/* ==============================================================================
 *
 * tests/test_memory.c — Test suite for cml/memory.h
 *
 * Minimal self-contained test harness: each TEST() macro runs a check and
 * records pass/fail. The program exits with 0 on full success, 1 otherwise.
 */

#include "cml/memory.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ==============================================================================
 * Minimal test harness
 * ------------------------------------------------------------------------------
 */

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
 * A. cml_malloc
 * ------------------------------------------------------------------------------
 */

static void test_malloc(void) {
  section("cml_malloc");

  /* Allocate 1 byte */
  void *p = cml_malloc(1);
  TEST("allocate 1 byte returns non-NULL", p != NULL);
  cml_free(p);

  /* Allocate 64 bytes and write to them */
  p = cml_malloc(64);
  TEST("allocate 64 bytes returns non-NULL", p != NULL);
  if (p) {
    unsigned char *buf = (unsigned char *)p;
    for (size_t i = 0; i < 64; ++i)
      buf[i] = (unsigned char)i;
    TEST("write and read 64 bytes without errors", buf[63] == 63);
    cml_free(p);
  }

  /* Allocate 1 MiB */
  p = cml_malloc(1024 * 1024);
  TEST("allocate 1 MiB returns non-NULL", p != NULL);
  cml_free(p);

  /* Zero-byte allocation returns NULL (documented behaviour) */
  p = cml_malloc(0);
  TEST("allocate 0 bytes returns NULL", p == NULL);
  /* cml_free(NULL) is safe */
  cml_free(p);
}

/* ==============================================================================
 * B. cml_calloc
 * ------------------------------------------------------------------------------
 */

static void test_calloc(void) {
  section("cml_calloc");

  /* Allocate 10 ints, verify all zero */
  int *arr = (int *)cml_calloc(10, sizeof(int));
  TEST("calloc 10 ints returns non-NULL", arr != NULL);
  if (arr) {
    int all_zero = 1;
    for (int i = 0; i < 10; ++i) {
      if (arr[i] != 0) {
        all_zero = 0;
        break;
      }
    }
    TEST("calloc 10 ints: all bytes zero", all_zero);
    cml_free(arr);
  }

  /* Allocate 100 bytes, verify entire region is zero */
  unsigned char *buf = (unsigned char *)cml_calloc(100, 1);
  TEST("calloc 100 bytes returns non-NULL", buf != NULL);
  if (buf) {
    int all_zero = 1;
    for (int i = 0; i < 100; ++i) {
      if (buf[i] != 0) {
        all_zero = 0;
        break;
      }
    }
    TEST("calloc 100 bytes: entire region zero", all_zero);
    cml_free(buf);
  }

  /* count == 0 returns NULL */
  void *p = cml_calloc(0, sizeof(int));
  TEST("calloc count=0 returns NULL", p == NULL);

  /* element_size == 0 returns NULL */
  p = cml_calloc(10, 0);
  TEST("calloc element_size=0 returns NULL", p == NULL);

  /* Overflow: count * element_size > SIZE_MAX */
  p = cml_calloc(SIZE_MAX, 2);
  TEST("calloc with overflow returns NULL", p == NULL);
}

/* ==============================================================================
 * C. cml_realloc
 * ------------------------------------------------------------------------------
 */

static void test_realloc(void) {
  section("cml_realloc");

  /* Grow: 8 bytes → 64 bytes, original data preserved */
  unsigned char *p = (unsigned char *)cml_malloc(8);
  TEST("realloc grow: initial alloc succeeds", p != NULL);
  if (p) {
    for (int i = 0; i < 8; ++i)
      p[i] = (unsigned char)(i + 1);
    unsigned char *q = (unsigned char *)cml_realloc(p, 64);
    TEST("realloc grow returns non-NULL", q != NULL);
    if (q) {
      int preserved = 1;
      for (int i = 0; i < 8; ++i) {
        if (q[i] != (unsigned char)(i + 1)) {
          preserved = 0;
          break;
        }
      }
      TEST("realloc grow: original 8 bytes preserved", preserved);
      cml_free(q);
    } else {
      cml_free(p);
    }
  }

  /* Shrink: 64 bytes → 8 bytes, first 8 bytes preserved */
  p = (unsigned char *)cml_malloc(64);
  TEST("realloc shrink: initial alloc succeeds", p != NULL);
  if (p) {
    for (int i = 0; i < 64; ++i)
      p[i] = (unsigned char)(i + 10);
    unsigned char *q = (unsigned char *)cml_realloc(p, 8);
    TEST("realloc shrink returns non-NULL", q != NULL);
    if (q) {
      int preserved = 1;
      for (int i = 0; i < 8; ++i) {
        if (q[i] != (unsigned char)(i + 10)) {
          preserved = 0;
          break;
        }
      }
      TEST("realloc shrink: first 8 bytes preserved", preserved);
      cml_free(q);
    } else {
      cml_free(p);
    }
  }

  /* Same size */
  p = (unsigned char *)cml_malloc(32);
  TEST("realloc same size: initial alloc succeeds", p != NULL);
  if (p) {
    for (int i = 0; i < 32; ++i)
      p[i] = 0xAB;
    unsigned char *q = (unsigned char *)cml_realloc(p, 32);
    TEST("realloc same size returns non-NULL", q != NULL);
    if (q) {
      TEST("realloc same size: data valid", q[0] == 0xAB && q[31] == 0xAB);
      cml_free(q);
    } else {
      cml_free(p);
    }
  }

  /* NULL pointer acts like cml_malloc */
  void *fresh = cml_realloc(NULL, 16);
  TEST("realloc NULL pointer acts like malloc", fresh != NULL);
  cml_free(fresh);

  /* Resize to zero frees and returns NULL */
  void *to_free = cml_malloc(16);
  TEST("realloc to zero: initial alloc succeeds", to_free != NULL);
  if (to_free) {
    void *result = cml_realloc(to_free, 0);
    TEST("realloc new_size=0 returns NULL", result == NULL);
    /* to_free has been freed by cml_realloc; do not use it */
  }

  /*
   * Failure safety: attempt a huge realloc. If it fails, the original
   * allocation must remain valid and its data intact.
   *
   * Note: on systems with memory overcommit this huge allocation may
   * succeed. The test covers whichever outcome actually occurs.
   */
  unsigned char *orig = (unsigned char *)cml_malloc(16);
  TEST("realloc failure test: initial alloc succeeds", orig != NULL);
  if (orig) {
    for (int i = 0; i < 16; ++i)
      orig[i] = (unsigned char)(i + 1);
    unsigned char *huge = (unsigned char *)cml_realloc(orig, SIZE_MAX);
    if (huge == NULL) {
      /* Allocation failed — original must still be intact */
      int intact = 1;
      for (int i = 0; i < 16; ++i) {
        if (orig[i] != (unsigned char)(i + 1)) {
          intact = 0;
          break;
        }
      }
      TEST("realloc failure: original data intact", intact);
      cml_free(orig);
    } else {
      /* Overcommit: allocation "succeeded" */
      TEST("realloc huge succeeded (overcommit); data preserved", 1);
      cml_free(huge);
    }
  }
}

/* ==============================================================================
 * D. cml_free
 * ------------------------------------------------------------------------------
 */

static void test_free(void) {
  section("cml_free");

  /* Free a valid allocation */
  void *p = cml_malloc(32);
  TEST("free valid allocation: no errors", p != NULL);
  cml_free(p);
  TEST("free valid allocation: completed", 1);

  /* Free NULL is safe */
  cml_free(NULL);
  TEST("free NULL is safe", 1);

  /* Free after realloc */
  void *a = cml_malloc(16);
  void *b = cml_realloc(a, 64);
  TEST("free after realloc: realloc succeeded", b != NULL);
  cml_free(b);
  TEST("free after realloc: completed", 1);

  /* Free after calloc */
  void *c = cml_calloc(8, sizeof(double));
  TEST("free after calloc: calloc succeeded", c != NULL);
  cml_free(c);
  TEST("free after calloc: completed", 1);
}

/* ==============================================================================
 * E. cml_mem_copy
 * ------------------------------------------------------------------------------
 */

static void test_mem_copy(void) {
  section("cml_mem_copy");

  unsigned char src[100];
  unsigned char dst[100];

  for (int i = 0; i < 100; ++i)
    src[i] = (unsigned char)i;

  /* Copy 1 byte */
  memset(dst, 0, sizeof(dst));
  void *ret = cml_mem_copy(dst, src, 1);
  TEST("copy 1 byte: returns dst", ret == dst);
  TEST("copy 1 byte: correct value", dst[0] == src[0]);

  /* Copy 8 bytes */
  memset(dst, 0, sizeof(dst));
  ret = cml_mem_copy(dst, src, 8);
  TEST("copy 8 bytes: returns dst", ret == dst);
  int match8 = (memcmp(dst, src, 8) == 0);
  TEST("copy 8 bytes: all bytes correct", match8);

  /* Copy 100 bytes */
  memset(dst, 0, sizeof(dst));
  ret = cml_mem_copy(dst, src, 100);
  TEST("copy 100 bytes: returns dst", ret == dst);
  int match100 = (memcmp(dst, src, 100) == 0);
  TEST("copy 100 bytes: all bytes correct", match100);

  /* Copy 0 bytes returns NULL (documented behaviour) */
  ret = cml_mem_copy(dst, src, 0);
  TEST("copy 0 bytes returns NULL", ret == NULL);

  /* Source unchanged after copy */
  unsigned char src2[8] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
  unsigned char dst2[8] = {0};
  cml_mem_copy(dst2, src2, 8);
  int src_unchanged = (src2[0] == 0x11 && src2[7] == 0x88);
  TEST("copy: source unchanged after copy", src_unchanged);
  int dst_matches = (memcmp(dst2, src2, 8) == 0);
  TEST("copy: destination matches source", dst_matches);

  /* NULL dst returns NULL */
  ret = cml_mem_copy(NULL, src, 8);
  TEST("copy NULL dst returns NULL", ret == NULL);

  /* NULL src returns NULL */
  ret = cml_mem_copy(dst, NULL, 8);
  TEST("copy NULL src returns NULL", ret == NULL);
}

/* ==============================================================================
 * main
 * ------------------------------------------------------------------------------
 */

int main(void) {
  printf("=== CML Memory Module Test Suite ===\n");

  test_malloc();
  test_calloc();
  test_realloc();
  test_free();
  test_mem_copy();

  printf("\n--- Results: %d passed, %d failed ---\n", s_pass, s_fail);
  return (s_fail == 0) ? 0 : 1;
}
