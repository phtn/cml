
#ifndef CML_ERROR_H
#define CML_ERROR_H

/*
 * cml/error.h — CML Error Handling & Logging Module
 *
 * Provides:
 *   - CMLStatus: standardized error codes for the whole library.
 *   - cml_status_string(): human-readable status descriptions.
 *   - CMLLogLevel: severity levels for log messages.
 *   - cml_log(): printf-style logger with timestamp and level prefix.
 *   - cml_log_set_level(): runtime log-level filtering.
 *   - cml_log_set_file(): redirect (and mirror) log output to a file.
 *   - cml_log_close_file(): flush and close the log file.
 *
 * Design conventions:
 *   - Functions never abort or crash on bad input.
 *   - The logger does NOT use cml_malloc; it uses only stack/static storage
 *     so it remains usable even when the memory allocator itself fails.
 *   - Global logger state is intentional and documented (see Task F notes).
 *   - Thread safety: NOT guaranteed in this phase.
 */

#include <stddef.h>
#include <stdio.h>

/* ===========================================================================
 * CMLStatus — standardized error codes
 * ========================================================================= */

typedef enum CMLStatus {
    CML_SUCCESS                   = 0,
    CML_ERROR_INVALID_ARGUMENT    = 1,
    CML_ERROR_OUT_OF_MEMORY       = 2,
    CML_ERROR_OVERFLOW            = 3,
    CML_ERROR_INVALID_DIMENSION   = 4,
    CML_ERROR_INDEX_OUT_OF_BOUNDS = 5,
    CML_ERROR_IO                  = 6,
    CML_ERROR_NUMERICAL           = 7,
    CML_ERROR_NOT_IMPLEMENTED     = 8,
    CML_ERROR_UNKNOWN             = 9,
} CMLStatus;

/* ---------------------------------------------------------------------------
 * cml_status_string
 *
 * Convert a CMLStatus code into a human-readable, statically allocated string.
 *
 * Parameters:
 *   status — any CMLStatus value.
 *
 * Returns:
 *   A non-NULL pointer to a constant string. The pointer is always valid and
 *   never NULL; unknown codes return "Unknown error". The caller must NOT free
 *   or modify the returned string.
 *
 * Memory: allocates nothing.
 * --------------------------------------------------------------------------- */
const char *cml_status_string(CMLStatus status);

/* ===========================================================================
 * CMLLogLevel — severity levels
 * ========================================================================= */

typedef enum CMLLogLevel {
    CML_LOG_DEBUG = 0,
    CML_LOG_INFO  = 1,
    CML_LOG_WARN  = 2,
    CML_LOG_ERROR = 3,
    CML_LOG_FATAL = 4,
} CMLLogLevel;

/* ---------------------------------------------------------------------------
 * cml_log
 *
 * Emit a formatted log message at the given severity level.
 *
 * Output format:
 *   [YYYY-MM-DD HH:MM:SS] [LEVEL] message\n
 *
 * Messages below the current minimum level are silently suppressed.
 * Output goes to stderr and, if a log file is open, to the file as well.
 *
 * Parameters:
 *   level  — severity level; invalid values are silently ignored.
 *   fmt    — printf-compatible format string; NULL is treated as "".
 *   ...    — format arguments.
 *
 * Returns: nothing. Logging never crashes the application.
 * --------------------------------------------------------------------------- */
void cml_log(CMLLogLevel level, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

/* ---------------------------------------------------------------------------
 * cml_log_set_level
 *
 * Set the minimum severity level for log output.
 * Default: CML_LOG_DEBUG (all messages visible).
 *
 * Parameters:
 *   level — new minimum level; invalid values are silently ignored.
 * --------------------------------------------------------------------------- */
void cml_log_set_level(CMLLogLevel level);

/* ---------------------------------------------------------------------------
 * cml_log_set_file
 *
 * Open `path` in append mode and direct all future log output to it
 * (in addition to stderr). Any previously open log file is closed first.
 *
 * Parameters:
 *   path — file path string; NULL or empty string returns
 *           CML_ERROR_INVALID_ARGUMENT without opening anything.
 *
 * Returns:
 *   CML_SUCCESS              — file opened successfully.
 *   CML_ERROR_INVALID_ARGUMENT — path is NULL or empty.
 *   CML_ERROR_IO             — fopen failed (file not opened).
 * --------------------------------------------------------------------------- */
CMLStatus cml_log_set_file(const char *path);

/* ---------------------------------------------------------------------------
 * cml_log_close_file
 *
 * Flush and close the current log file, if any.
 * Safe to call when no file is open (no-op).
 * --------------------------------------------------------------------------- */
void cml_log_close_file(void);

#endif /* CML_ERROR_H */
