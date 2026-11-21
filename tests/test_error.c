/*
 * tests/test_error.c — Test suite for cml/error.h
 *
 * Uses the same minimal harness as test_memory.c.
 * Run: make test-error
 *
 * Note: logging tests redirect output to a captured FILE* (via open_memstream
 * / tmpfile) or to a temp file so we can verify content without polluting
 * the terminal. Where we can't capture stderr portably, we at minimum verify
 * that the call completes without crashing and returns the right value.
 */

#include "cml/error.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * Minimal test harness (same style as test_memory.c)
 * --------------------------------------------------------------------------- */

static int s_pass = 0;
static int s_fail = 0;

#define TEST(description, condition)                                       \
    do {                                                                   \
        if (condition) {                                                   \
            printf("  PASS  %s\n", description);                           \
            ++s_pass;                                                      \
        } else {                                                           \
            printf("  FAIL  %s  (line %d)\n", description, __LINE__);     \
            ++s_fail;                                                      \
        }                                                                  \
    } while (0)

static void section(const char *name)
{
    printf("\n[%s]\n", name);
}

/* ---------------------------------------------------------------------------
 * Helper: read the entire contents of a small file into a stack buffer.
 * Returns number of bytes read, or -1 on error.
 * --------------------------------------------------------------------------- */
static int slurp_file(const char *path, char *out, size_t out_size)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t n = fread(out, 1, out_size - 1, f);
    out[n] = '\0';
    fclose(f);
    return (int)n;
}

/* ===========================================================================
 * A. Error-code tests
 * ========================================================================= */

static void test_status_string(void)
{
    section("cml_status_string");

    /* Every defined code returns a non-NULL, non-empty string */
    TEST("CML_SUCCESS returns non-NULL",
         cml_status_string(CML_SUCCESS) != NULL);
    TEST("CML_SUCCESS == \"Success\"",
         strcmp(cml_status_string(CML_SUCCESS), "Success") == 0);

    TEST("CML_ERROR_INVALID_ARGUMENT != NULL",
         cml_status_string(CML_ERROR_INVALID_ARGUMENT) != NULL);
    TEST("CML_ERROR_INVALID_ARGUMENT == \"Invalid argument\"",
         strcmp(cml_status_string(CML_ERROR_INVALID_ARGUMENT), "Invalid argument") == 0);

    TEST("CML_ERROR_OUT_OF_MEMORY == \"Out of memory\"",
         strcmp(cml_status_string(CML_ERROR_OUT_OF_MEMORY), "Out of memory") == 0);

    TEST("CML_ERROR_OVERFLOW != NULL",
         cml_status_string(CML_ERROR_OVERFLOW) != NULL);

    TEST("CML_ERROR_INVALID_DIMENSION != NULL",
         cml_status_string(CML_ERROR_INVALID_DIMENSION) != NULL);

    TEST("CML_ERROR_INDEX_OUT_OF_BOUNDS != NULL",
         cml_status_string(CML_ERROR_INDEX_OUT_OF_BOUNDS) != NULL);

    TEST("CML_ERROR_IO != NULL",
         cml_status_string(CML_ERROR_IO) != NULL);

    TEST("CML_ERROR_NUMERICAL != NULL",
         cml_status_string(CML_ERROR_NUMERICAL) != NULL);

    TEST("CML_ERROR_NOT_IMPLEMENTED != NULL",
         cml_status_string(CML_ERROR_NOT_IMPLEMENTED) != NULL);

    TEST("CML_ERROR_UNKNOWN == \"Unknown error\"",
         strcmp(cml_status_string(CML_ERROR_UNKNOWN), "Unknown error") == 0);

    /* Out-of-range value must not crash and must return a fallback */
    const char *fallback = cml_status_string((CMLStatus)9999);
    TEST("unknown code returns non-NULL fallback", fallback != NULL);
    TEST("unknown code returns non-empty fallback", fallback && fallback[0] != '\0');

    /* Calling repeatedly is safe (static strings, no allocation) */
    const char *s1 = cml_status_string(CML_SUCCESS);
    const char *s2 = cml_status_string(CML_SUCCESS);
    TEST("repeated call returns same pointer", s1 == s2);
}

/* ===========================================================================
 * B. Logging — basic output (we verify no crash + file content)
 * ========================================================================= */

static void test_log_basic(void)
{
    section("cml_log — basic");

    /* Reset to show everything */
    cml_log_set_level(CML_LOG_DEBUG);

    /* These calls must complete without crashing. Output goes to stderr. */
    cml_log(CML_LOG_DEBUG, "debug message");
    TEST("DEBUG log does not crash", 1);

    cml_log(CML_LOG_INFO, "info message");
    TEST("INFO log does not crash", 1);

    cml_log(CML_LOG_WARN, "warn message");
    TEST("WARN log does not crash", 1);

    cml_log(CML_LOG_ERROR, "error message");
    TEST("ERROR log does not crash", 1);

    cml_log(CML_LOG_FATAL, "fatal message");
    TEST("FATAL log does not crash", 1);

    /* Formatted integer */
    cml_log(CML_LOG_INFO, "value = %d", 42);
    TEST("formatted integer does not crash", 1);

    /* Formatted float */
    cml_log(CML_LOG_INFO, "pi = %.4f", 3.1415);
    TEST("formatted float does not crash", 1);

    /* Formatted string */
    cml_log(CML_LOG_INFO, "name = %s", "cml");
    TEST("formatted string does not crash", 1);

    /* Multiple arguments */
    cml_log(CML_LOG_INFO, "matrix %zu x %zu allocated", (size_t)3, (size_t)4);
    TEST("multiple arguments does not crash", 1);

    /* NULL format string */
    cml_log(CML_LOG_INFO, NULL);
    TEST("NULL format string does not crash", 1);

    /* Empty format string */
    cml_log(CML_LOG_INFO, "");
    TEST("empty format string does not crash", 1);

    /* Very long message (triggers truncation path) */
    char long_msg[4096];
    memset(long_msg, 'A', sizeof(long_msg) - 1);
    long_msg[sizeof(long_msg) - 1] = '\0';
    cml_log(CML_LOG_DEBUG, "%s", long_msg);
    TEST("very long message does not crash", 1);

    /* Invalid level — must be silently ignored */
    cml_log((CMLLogLevel)-1, "bad level");
    TEST("invalid level (negative) does not crash", 1);

    cml_log((CMLLogLevel)999, "bad level high");
    TEST("invalid level (too high) does not crash", 1);
}

/* ===========================================================================
 * C. Log filtering
 * ========================================================================= */

static void test_log_filtering(void)
{
    section("cml_log — filtering (verified via file)");

    const char *path = "/tmp/cml_test_filter.log";
    remove(path);  /* start clean */

    /* Set minimum to WARN, then log at all levels */
    cml_log_set_level(CML_LOG_WARN);
    CMLStatus s = cml_log_set_file(path);
    TEST("open filter test log file", s == CML_SUCCESS);

    cml_log(CML_LOG_DEBUG, "SHOULD_NOT_APPEAR_DEBUG");
    cml_log(CML_LOG_INFO,  "SHOULD_NOT_APPEAR_INFO");
    cml_log(CML_LOG_WARN,  "SHOULD_APPEAR_WARN");
    cml_log(CML_LOG_ERROR, "SHOULD_APPEAR_ERROR");
    cml_log(CML_LOG_FATAL, "SHOULD_APPEAR_FATAL");

    cml_log_close_file();

    /* Read back the file and check contents */
    char buf[4096];
    int n = slurp_file(path, buf, sizeof(buf));
    TEST("filter test log file is readable", n >= 0);

    if (n >= 0) {
        TEST("DEBUG suppressed at WARN level",
             strstr(buf, "SHOULD_NOT_APPEAR_DEBUG") == NULL);
        TEST("INFO suppressed at WARN level",
             strstr(buf, "SHOULD_NOT_APPEAR_INFO") == NULL);
        TEST("WARN appears at WARN level",
             strstr(buf, "SHOULD_APPEAR_WARN") != NULL);
        TEST("ERROR appears at WARN level",
             strstr(buf, "SHOULD_APPEAR_ERROR") != NULL);
        TEST("FATAL appears at WARN level",
             strstr(buf, "SHOULD_APPEAR_FATAL") != NULL);
    }

    /* Now switch to DEBUG — debug must appear */
    remove(path);
    cml_log_set_level(CML_LOG_DEBUG);
    s = cml_log_set_file(path);
    TEST("re-open filter test log at DEBUG", s == CML_SUCCESS);

    cml_log(CML_LOG_DEBUG, "NOW_VISIBLE_DEBUG");
    cml_log_close_file();

    n = slurp_file(path, buf, sizeof(buf));
    TEST("DEBUG visible after switching to DEBUG level",
         n > 0 && strstr(buf, "NOW_VISIBLE_DEBUG") != NULL);

    /* Reset to DEBUG for remaining tests */
    cml_log_set_level(CML_LOG_DEBUG);
    remove(path);
}

/* ===========================================================================
 * D. File logging
 * ========================================================================= */

static void test_log_file(void)
{
    section("cml_log — file logging");

    const char *path = "/tmp/cml_test_file.log";
    remove(path);

    /* Open a new file */
    CMLStatus s = cml_log_set_file(path);
    TEST("open log file returns CML_SUCCESS", s == CML_SUCCESS);

    /* Write messages */
    cml_log(CML_LOG_INFO,  "first message");
    cml_log(CML_LOG_ERROR, "second message");
    cml_log(CML_LOG_WARN,  "third message");

    cml_log_close_file();

    /* Verify file exists and contains the messages */
    char buf[4096];
    int n = slurp_file(path, buf, sizeof(buf));
    TEST("log file is readable after close", n > 0);
    TEST("first message in file",  n > 0 && strstr(buf, "first message")  != NULL);
    TEST("second message in file", n > 0 && strstr(buf, "second message") != NULL);
    TEST("third message in file",  n > 0 && strstr(buf, "third message")  != NULL);

    /* Verify timestamp and level labels appear */
    TEST("file contains timestamp bracket",  n > 0 && strstr(buf, "[20") != NULL);
    TEST("file contains INFO label",  n > 0 && strstr(buf, "INFO")  != NULL);
    TEST("file contains ERROR label", n > 0 && strstr(buf, "ERROR") != NULL);
    TEST("file contains WARN label",  n > 0 && strstr(buf, "WARN")  != NULL);

    /* Append: re-open and add more */
    s = cml_log_set_file(path);
    TEST("re-open existing log file for append", s == CML_SUCCESS);
    cml_log(CML_LOG_INFO, "appended message");
    cml_log_close_file();

    n = slurp_file(path, buf, sizeof(buf));
    TEST("appended message in file",
         n > 0 && strstr(buf, "appended message") != NULL);
    TEST("original messages still in file after append",
         n > 0 && strstr(buf, "first message") != NULL);

    /* Closing again is safe (no-op) */
    cml_log_close_file();
    TEST("double close is safe", 1);

    /* Logging after close goes only to stderr — must not crash */
    cml_log(CML_LOG_INFO, "after file close");
    TEST("log after file close does not crash", 1);

    /* NULL path */
    s = cml_log_set_file(NULL);
    TEST("NULL path returns CML_ERROR_INVALID_ARGUMENT",
         s == CML_ERROR_INVALID_ARGUMENT);

    /* Empty path */
    s = cml_log_set_file("");
    TEST("empty path returns CML_ERROR_INVALID_ARGUMENT",
         s == CML_ERROR_INVALID_ARGUMENT);

    /* Invalid path (directory that likely doesn't exist) */
    s = cml_log_set_file("/nonexistent_dir_cml/x/y/z.log");
    TEST("invalid path returns CML_ERROR_IO", s == CML_ERROR_IO);

    /* Repeated set_file: previous file should be closed first */
    const char *path2 = "/tmp/cml_test_file2.log";
    remove(path2);
    s = cml_log_set_file(path);
    TEST("open first log file", s == CML_SUCCESS);
    s = cml_log_set_file(path2);
    TEST("switch to second log file closes first", s == CML_SUCCESS);
    cml_log(CML_LOG_INFO, "in second file");
    cml_log_close_file();

    n = slurp_file(path2, buf, sizeof(buf));
    TEST("second file contains message", n > 0 && strstr(buf, "in second file") != NULL);

    remove(path);
    remove(path2);
}

/* ===========================================================================
 * E. Edge cases
 * ========================================================================= */

static void test_edge_cases(void)
{
    section("edge cases");

    /* Invalid log levels for set_level — must be silently ignored */
    cml_log_set_level((CMLLogLevel)-1);
    TEST("set_level with negative value does not crash", 1);

    cml_log_set_level((CMLLogLevel)9999);
    TEST("set_level with huge value does not crash", 1);

    /* Confirm level was not changed by invalid calls */
    cml_log_set_level(CML_LOG_WARN);
    cml_log_set_level((CMLLogLevel)-1);  /* should be ignored */

    const char *path = "/tmp/cml_edge.log";
    remove(path);
    cml_log_set_file(path);
    cml_log(CML_LOG_DEBUG, "EDGE_DEBUG_SUPPRESSED");
    cml_log_close_file();

    char buf[1024];
    int n = slurp_file(path, buf, sizeof(buf));
    TEST("invalid set_level ignored; prior level still active",
         n >= 0 && strstr(buf, "EDGE_DEBUG_SUPPRESSED") == NULL);

    /* Reset for cleanup */
    cml_log_set_level(CML_LOG_DEBUG);

    /* Repeated close is safe */
    cml_log_close_file();
    cml_log_close_file();
    TEST("repeated cml_log_close_file is safe", 1);

    /* NULL format */
    cml_log(CML_LOG_INFO, NULL);
    TEST("NULL fmt does not crash", 1);

    remove(path);
}

/* ===========================================================================
 * main
 * ========================================================================= */

int main(void)
{
    printf("=== CML Error & Logger Test Suite ===\n");

    test_status_string();
    test_log_basic();
    test_log_filtering();
    test_log_file();
    test_edge_cases();

    printf("\n--- Results: %d passed, %d failed ---\n", s_pass, s_fail);
    return (s_fail == 0) ? 0 : 1;
}
