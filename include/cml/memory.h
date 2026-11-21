
#ifndef CML_MEMORY_H
#define CML_MEMORY_H

/*
 * cml/memory.h — CML Memory Management Module
 *
 * Public API for safe allocation, reallocation, freeing, and copying of
 * memory within the C Machine Learning Library (cml).
 *
 * Design conventions:
 *   - All sizes are expressed as `size_t`.
 *   - On failure, functions return NULL; they never abort or print messages.
 *   - Zero-sized requests consistently return NULL (documented per function).
 *   - The caller owns every pointer returned and must free it with cml_free.
 */

#include <stddef.h>
/* ==============================================================================
 * cml_malloc
 *
 * Allocate `size` bytes of uninitialised memory.
 *
 * Parameters:
 *   size  — number of bytes to allocate; must be > 0.
 *
 * Returns:
 *   Pointer to allocated memory on success.
 *   NULL if size == 0 or allocation fails.
 *
 * Ownership:
 *   The caller is responsible for freeing the returned pointer with cml_free.
 * ------------------------------------------------------------------------------
 */
void *cml_malloc(size_t size);

/* ==============================================================================
 * cml_calloc
 *
 * Allocate `count` elements of `element_size` bytes each, all zero-initialised.
 * Detects multiplication overflow before computing the total byte count.
 *
 * Parameters:
 *   count        — number of elements; must be > 0.
 *   element_size — size of each element in bytes; must be > 0.
 *
 * Returns:
 *   Pointer to zero-initialised memory on success.
 *   NULL if count == 0, element_size == 0, overflow is detected, or
 *   allocation fails.
 *
 * Ownership:
 *   The caller is responsible for freeing the returned pointer with cml_free.
 * ------------------------------------------------------------------------------
 */
void *cml_calloc(size_t count, size_t element_size);

/* ==============================================================================
 * cml_realloc
 *
 * Resize a previously allocated block to `new_size` bytes.
 *
 * On failure the original block is left untouched and remains valid; the
 * caller must NOT replace `ptr` with the return value until it has checked
 * for NULL.
 *
 * Parameters:
 *   ptr      — pointer previously returned by
 * cml_malloc/cml_calloc/cml_realloc, or NULL (treated as a fresh cml_malloc
 * call). new_size — desired size in bytes.
 *
 * Returns:
 *   Pointer to resized memory on success (may differ from `ptr`).
 *   NULL if new_size == 0 (original is freed; equivalent to cml_free) or if
 *   allocation fails (original block is preserved).
 *
 * Ownership:
 *   On success, `ptr` is no longer valid; use the returned pointer.
 *   On failure (non-NULL return expected but NULL returned), `ptr` is still
 *   valid and the caller must free it separately.
 * ------------------------------------------------------------------------------
 */
void *cml_realloc(void *ptr, size_t new_size);

/* ==============================================================================
 * cml_free
 *
 * Release a block of memory previously allocated by this module.
 *
 * Parameters:
 *   ptr — pointer to free, or NULL (safe no-op).
 *
 * Returns:
 *   Nothing. Passing NULL is safe and has no effect.
 *
 * Ownership:
 *   After this call `ptr` is invalid and must not be used.
 *   Double-free is undefined behaviour; do not call cml_free twice on the
 *   same pointer.
 * ------------------------------------------------------------------------------
 */
void cml_free(void *ptr);

/* ==============================================================================
 * cml_mem_copy
 *
 * Copy `n` bytes from `src` to `dst`.
 *
 * The source and destination regions MUST NOT overlap. For overlapping
 * regions use memmove directly; this function uses memcpy for performance.
 *
 * Parameters:
 *   dst — destination buffer; must point to at least `n` writable bytes.
 *   src — source buffer; must point to at least `n` readable bytes.
 *   n   — number of bytes to copy.
 *
 * Returns:
 *   `dst` on success.
 *   NULL if dst == NULL, src == NULL, or n == 0.
 *
 * Ownership:
 *   Both buffers remain owned by the caller.
 * ------------------------------------------------------------------------------
 */
void *cml_mem_copy(void *dst, const void *src, size_t n);

#endif /* CML_MEMORY_H */
