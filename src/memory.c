/*
 * src/memory.c — CML Memory Management Implementation
 *
 * Implements the five public functions declared in cml/memory.h.
 * No global mutable state; no error printing; caller controls error handling.
 */

#include "cml/memory.h"

#include <stdint.h> /* SIZE_MAX */
#include <stdlib.h> /* malloc, calloc, realloc, free */
#include <string.h> /* memcpy */

/* ==============================================================================
 * cml_malloc
 * ------------------------------------------------------------------------------
 */
void *cml_malloc(size_t size) {
  if (size == 0) {
    return NULL;
  }
  return malloc(size);
}

/* ==============================================================================
 * cml_calloc
 *
 * Overflow check: if count * element_size > SIZE_MAX the multiplication
 * wraps, producing a dangerously small allocation. We detect this before
 * calling the system calloc.
 * ------------------------------------------------------------------------------
 */
void *cml_calloc(size_t count, size_t element_size) {
  if (count == 0 || element_size == 0) {
    return NULL;
  }

  /* Overflow guard: a * b > SIZE_MAX  <=>  a > SIZE_MAX / b (b != 0) */
  if (count > SIZE_MAX / element_size) {
    return NULL; /* would overflow */
  }

  return calloc(count, element_size);
}

/* ==============================================================================
 * cml_realloc
 *
 * Critical pattern: we store the return value in a temporary pointer so that
 * on failure we do not overwrite `ptr` and lose the original allocation.
 *
 * new_size == 0: frees `ptr` and returns NULL (documented behaviour).
 * ------------------------------------------------------------------------------
 */
void *cml_realloc(void *ptr, size_t new_size) {
  if (new_size == 0) {
    free(ptr);
    return NULL;
  }

  void *tmp = realloc(ptr, new_size);
  if (tmp == NULL) {
    /* Original `ptr` is still valid; return NULL to signal failure. */
    return NULL;
  }

  return tmp;
}

/* ==============================================================================
 * cml_free
 *
 * free(NULL) is defined by the C standard to be a safe no-op, so we simply
 * delegate. The wrapper exists for naming consistency and to allow future
 * debug instrumentation (e.g., Bonus D).
 * ------------------------------------------------------------------------------
 */
void cml_free(void *ptr) { free(ptr); }

/* ==============================================================================
 * cml_mem_copy
 *
 * Uses memcpy, which is undefined for overlapping regions. See header docs.
 * ------------------------------------------------------------------------------
 */
void *cml_mem_copy(void *dst, const void *src, size_t n) {
  if (dst == NULL || src == NULL || n == 0) {
    return NULL;
  }
  return memcpy(dst, src, n);
}
