/*
 * src/error.c — CML Error Handling & Logging Implementation
 */

#include "cml/error.h"

#include <stdarg.h> /* va_list, va_start, va_copy, va_end */
#include <stdio.h>  /* fprintf, vsnprintf, fopen, fclose, fflush */
#include <string.h> /* strlen */
#include <time.h>   /* time, localtime, strftime */

/* ==============================================================================
 * Internal logger state
 *
 * Global mutable state is intentional here. A logger is inherently a process-
 * wide singleton. The tradeoff is that it is not thread-safe (documented in
 * the header). A future phase can add a mutex.
 * ------------------------------------------------------------------------------
 */

static CMLLogLevel s_min_level = CML_LOG_DEBUG; /* default: show everything */
static FILE *s_log_file = NULL;

/* Maximum formatted message length. Messages longer than this are truncated
 * with a visible marker so the caller is aware. No heap allocation is used. */
#define LOG_BUF_SIZE 2048
#define TRUNC_MARKER " ...[truncated]"

/* ==============================================================================
 * cml_status_string
 * ------------------------------------------------------------------------------
 */

const char *cml_status_string(CMLStatus status) {
  switch (status) {
  case CML_SUCCESS:
    return "Success";
  case CML_ERROR_INVALID_ARGUMENT:
    return "Invalid argument";
  case CML_ERROR_OUT_OF_MEMORY:
    return "Out of memory";
  case CML_ERROR_OVERFLOW:
    return "Arithmetic overflow";
  case CML_ERROR_INVALID_DIMENSION:
    return "Invalid dimension";
  case CML_ERROR_INDEX_OUT_OF_BOUNDS:
    return "Index out of bounds";
  case CML_ERROR_IO:
    return "I/O error";
  case CML_ERROR_NUMERICAL:
    return "Numerical error";
  case CML_ERROR_NOT_IMPLEMENTED:
    return "Not implemented";
  case CML_ERROR_UNKNOWN:
    return "Unknown error";
  default:
    return "Unknown error";
  }
}

/* ==============================================================================
 * Internal helpers
 * ------------------------------------------------------------------------------
 */

/* Return the string label for a log level. */
static const char *level_label(CMLLogLevel level) {
  switch (level) {
  case CML_LOG_DEBUG:
    return "DEBUG";
  case CML_LOG_INFO:
    return "INFO ";
  case CML_LOG_WARN:
    return "WARN ";
  case CML_LOG_ERROR:
    return "ERROR";
  case CML_LOG_FATAL:
    return "FATAL";
  default:
    return "?????";
  }
}

/* ==============================================================================
 * Write the timestamp portion "[YYYY-MM-DD HH:MM:SS]" into `buf` (size `n`).
 * Falls back to "[----]" if the system time is unavailable.
 * ------------------------------------------------------------------------------
 */
static void format_timestamp(char *buf, size_t n) {
  time_t now = time(NULL);
  if (now == (time_t)-1) {
    (void)snprintf(buf, n, "[----]");
    return;
  }
  struct tm *tm_info = localtime(&now);
  if (tm_info == NULL) {
    (void)snprintf(buf, n, "[----]");
    return;
  }
  (void)strftime(buf, n, "[%Y-%m-%d %H:%M:%S]", tm_info);
}

/* Emit one complete log line to the given FILE stream. */
static void emit_line(FILE *stream, const char *timestamp, const char *label,
                      const char *msg) {
  (void)fprintf(stream, "%s [%s] %s\n", timestamp, label, msg);
  (void)fflush(stream);
}

/* ==============================================================================
 * cml_log
 * ------------------------------------------------------------------------------
 */

void cml_log(CMLLogLevel level, const char *fmt, ...) {
  /* Validate level */
  if (level < CML_LOG_DEBUG || level > CML_LOG_FATAL) {
    return;
  }
  /* Filter */
  if (level < s_min_level) {
    return;
  }

  /* Format the user message into a stack buffer */
  char msg[LOG_BUF_SIZE];
  va_list args;
  va_start(args, fmt);

  if (fmt == NULL) {
    msg[0] = '\0';
  } else {
    int written = vsnprintf(msg, sizeof(msg), fmt, args);
    /* vsnprintf returns the number of chars that *would* have been written.
     * If it exceeds our buffer we append a truncation marker. */
    if (written >= (int)sizeof(msg)) {
      /* Overwrite the last bytes with the truncation marker */
      const char *marker = TRUNC_MARKER;
      size_t marker_len = strlen(marker);
      size_t buf_last = sizeof(msg) - 1 - marker_len;
      (void)memcpy(msg + buf_last, marker, marker_len + 1);
    }
  }

  va_end(args);

  /* Build timestamp */
  char ts[32];
  format_timestamp(ts, sizeof(ts));

  const char *label = level_label(level);

  /* Emit to stderr */
  emit_line(stderr, ts, label, msg);

  /* Mirror to log file if open */
  if (s_log_file != NULL) {
    emit_line(s_log_file, ts, label, msg);
  }
}

/* ==============================================================================
 * cml_log_set_level
 * ------------------------------------------------------------------------------
 */

void cml_log_set_level(CMLLogLevel level) {
  if (level < CML_LOG_DEBUG || level > CML_LOG_FATAL) {
    return; /* invalid — ignore silently */
  }
  s_min_level = level;
}

/* ==============================================================================
 * cml_log_set_file
 * ------------------------------------------------------------------------------
 */

CMLStatus cml_log_set_file(const char *path) {
  if (path == NULL || path[0] == '\0') {
    return CML_ERROR_INVALID_ARGUMENT;
  }

  /* Close any previously open log file */
  cml_log_close_file();

  s_log_file = fopen(path, "a");
  if (s_log_file == NULL) {
    return CML_ERROR_IO;
  }

  return CML_SUCCESS;
}

/* ==============================================================================
 * cml_log_close_file
 * ------------------------------------------------------------------------------
 */

void cml_log_close_file(void) {
  if (s_log_file != NULL) {
    (void)fflush(s_log_file);
    (void)fclose(s_log_file);
    s_log_file = NULL;
  }
}
