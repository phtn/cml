# **PHASE 1**

Let's continue building your C machine learning library, **CML (C Machine Learning)**. Since you're progressing through Phase 0, we'll make the next assignment build directly on your memory module.

The next phase is **Phase 0 — Assignment 02: Error Handling & Logging System**.

This is important because a machine learning library needs to handle invalid inputs, allocation failures, numerical errors, invalid dimensions, and unexpected states without crashing or silently producing incorrect results.

# **Assignment 02 — Error Handling & Logging**

**Difficulty:** Beginner → Intermediate

**Estimated time:** 3–6 hours

**Dependencies:** Assignment 01 — Memory Module

## **1. Objective**

Build a reusable error-handling and logging system that other modules in CML can use.

By the end, you should be able to:

- Represent errors using standardized error codes.
- Convert error codes into readable messages.
- Log messages with different severity levels.
- Enable or disable log levels.
- Write logs to a file.
- Handle errors without crashing.
- Propagate errors between functions.

## **2. Project structure**

Extend your existing project:

```
cml/
├── include/
│   └── cml/
│       ├── memory.h
│       └── error.h
├── src/
│   ├── memory.c
│   └── error.c
├── tests/
│   ├── test_memory.c
│   └── test_error.c
├── examples/
│   └── error_example.c
├── Makefile
└── README.md
```

You will create three new files:

- `include/cml/error.h`
- `src/error.c`
- `tests/test_error.c`

# **3. Requirements**

## **Task A — Define standardized error codes**

Create an error type called `CMLStatus`.

Your library should support at least these error categories:

| Error code                      | Meaning                               |
| ------------------------------- | ------------------------------------- |
| `CML_SUCCESS`                   | Operation completed successfully      |
| `CML_ERROR_INVALID_ARGUMENT`    | Invalid argument passed to a function |
| `CML_ERROR_OUT_OF_MEMORY`       | Memory allocation failed              |
| `CML_ERROR_OVERFLOW`            | Arithmetic overflow                   |
| `CML_ERROR_INVALID_DIMENSION`   | Invalid dimensions                    |
| `CML_ERROR_INDEX_OUT_OF_BOUNDS` | Index exceeds valid range             |
| `CML_ERROR_IO`                  | Input/output error                    |
| `CML_ERROR_NUMERICAL`           | Numerical computation failed          |
| `CML_ERROR_NOT_IMPLEMENTED`     | Feature is not implemented            |
| `CML_ERROR_UNKNOWN`             | Unknown error                         |

### **Requirements**

- Use an `enum`.
- All error codes must have unique values.
- Success must be distinguishable from failure.
- Every error code must have a readable description.
- Do not use arbitrary integer error values throughout your library.

**Your task:** Design the enum and decide whether the numbering should be explicit or implicit.

## **Task B — Error-to-string conversion**

Implement a function that converts an error code into a human-readable string.

Conceptually:

```
const char *cml_status_string(CMLStatus status);
```

Expected behavior:

| Input                        | Expected output      |
| ---------------------------- | -------------------- |
| `CML_SUCCESS`                | `"Success"`          |
| `CML_ERROR_OUT_OF_MEMORY`    | `"Out of memory"`    |
| `CML_ERROR_INVALID_ARGUMENT` | `"Invalid argument"` |
| Unknown value                | `"Unknown error"`    |

### **Constraints**

- The function must not allocate memory.
- The returned string must remain valid after the function returns.
- It must be safe to call repeatedly.
- It must not crash when given an invalid error code.

## **Task C — Logging severity levels**

Design a logging system supporting the following levels:

| Level   | Purpose                              |
| ------- | ------------------------------------ |
| `DEBUG` | Detailed development information     |
| `INFO`  | Normal operational information       |
| `WARN`  | Something unexpected but recoverable |
| `ERROR` | An operation failed                  |
| `FATAL` | A critical failure                   |

You should design your own enum for the log levels.

Your logging output should contain:

- Timestamp
- Log level
- Message

Example output:

```
[2026-11-21 10:15:32] [INFO] CML initialized
[2026-11-21 10:15:32] [WARN] Matrix dimension is unusually large
[2026-11-21 10:15:32] [ERROR] Memory allocation failed
```

You don't have to match the exact formatting. Focus on readability and consistency.

## **Task D — Implement the logger**

Design a logging function that supports formatted output, similar to `printf`.

For example, your future modules should be able to log:

```
cml_log(CML_LOG_INFO, "Matrix created: %zu x %zu", rows, cols);
```

The logger must support:

- Strings
- Integers
- Floating-point values
- Multiple formatting arguments
- Newline handling
- Empty messages

### **Constraints**

- Use variadic arguments.
- Do not manually concatenate strings.
- Do not create a fixed-size message buffer that silently truncates messages without handling it.
- Logging should not crash the application if the message is malformed or resources are unavailable.

Hint: Look into the C standard library's `stdarg.h`.

## **Task E — Log filtering**

Add configurable log filtering.

For example, if the minimum level is `WARN`, then:

```
DEBUG → ignored
INFO  → ignored
WARN  → displayed
ERROR → displayed
FATAL → displayed
```

Requirements:

- Users can configure the minimum log level.
- The default level should be documented.
- Changing the log level should take effect immediately.
- Suppressed messages should not be printed.
- Invalid log levels should be handled safely.

## **Task F — File logging**

Add the ability to write log output to a file.

Design a function that lets the user configure a log file.

Requirements:

- Open the file safely.
- Handle failure to open the file.
- Support appending to an existing log file.
- Don't lose messages because the file was not flushed appropriately.
- Provide a way to close the file.
- Prevent invalid file operations.

Example usage:

```
cml_log_set_file("cml.log");

cml_log(CML_LOG_INFO, "Training started");
cml_log(CML_LOG_ERROR, "Invalid dataset");

cml_log_close_file();
```

You can decide whether logs go to the terminal, a file, or both.

# **4. API design challenge**

Design your public header before implementing the functions.

Your header should contain declarations for functions conceptually similar to:

```
CMLStatus cml_status_string(...);

void cml_log(...);

void cml_log_set_level(...);

CMLStatus cml_log_set_file(...);

void cml_log_close_file(...);
```

These are conceptual signatures, not necessarily the exact signatures you should use.

**Your job is to determine:**

1. What should each function return?
2. How should invalid arguments be handled?
3. Should logging itself return errors?
4. What happens when logging to a file fails?
5. Should logging be thread-safe? (For this assignment, thread safety can be optional.)
6. Should logging have global state? If so, what are the tradeoffs?

# **5. Test cases**

Write your own test suite.

## **Error-code tests**

- Every valid error code returns a valid string.
- Success returns the correct message.
- Unknown error codes return a fallback message.
- Calling the conversion function repeatedly works.

## **Logging tests**

- `DEBUG` logs correctly.
- `INFO` logs correctly.
- `WARN` logs correctly.
- `ERROR` logs correctly.
- `FATAL` logs correctly.
- Formatted integer values print correctly.
- Formatted floating-point values print correctly.
- Strings print correctly.
- Empty messages do not crash.
- Multiple arguments work.

## **Filtering tests**

Set minimum level to `WARN`.

Verify:

- Debug is suppressed.
- Info is suppressed.
- Warn is printed.
- Error is printed.
- Fatal is printed.

Then change the level to `DEBUG` and verify that debug messages are enabled.

## **File tests**

- Opening a new file works.
- Messages are written to the file.
- Multiple messages are appended during the same session.
- Closing the file works.
- Attempting to open an invalid path is handled safely.
- Logging after closing the file doesn't crash.

## **Edge cases**

- Invalid severity values.
- Null format string.
- Null file path.
- Very long log message.
- Repeated initialization.
- Repeated file closing.

# **6. Hints (without solutions)**

### **Hint 1 — Variadic functions**

Study:

```
#include <stdarg.h>
```

Look up:

- `va_list`
- `va_start`
- `va_end`
- `va_copy`
- `vsnprintf`

These are fundamental to implementing printf-style logging.

### **Hint 2 — Timestamps**

Study:

```
#include <time.h>
```

Useful functions include:

- `time`
- `localtime`
- `strftime`

Think about what happens if a time conversion fails.

### **Hint 3 — File handling**

Study:

```
#include <stdio.h>
```

Useful functions include:

- `fopen`
- `fprintf`
- `vfprintf`
- `fflush`
- `fclose`

### **Hint 4 — Error propagation**

Imagine a future matrix function:

```
matrix_create()
    ├── validate dimensions
    ├── calculate allocation size
    ├── allocate memory
    └── initialize matrix
```

Each operation might fail.

Think about how an error from memory allocation should propagate back to `matrix_create()` without hiding its original cause.

### **Hint 5 — Avoid unnecessary allocations**

Logging should ideally not require your custom memory module to allocate memory for every log message.

This helps prevent a situation where your memory allocator fails and your logging system also fails because it depends on that allocator.

# **7. Bonus challenges**

If you finish the main requirements, try these:

- Add custom log prefixes.
- Add ANSI terminal colors.
- Allow users to disable timestamps.
- Add a custom log callback.
- Add log rotation (e.g. when file exceeds 5 MB).
- Include source file and line number in logs.
- Add macros such as `CML_LOG_ERROR(...)` that automatically capture source location.
- Make the logger thread-safe using C11 synchronization or platform mechanisms (later phase).

Example of what a more advanced log might look like:

```
[2026-11-21 10:15:32] [ERROR] [matrix.c:142] Invalid matrix dimensions: 0 x 5
```

# **8. Scoring rubric**

| Category                          | Points  |
| --------------------------------- | ------- |
| Error codes and status strings    | 15      |
| Logging functionality             | 20      |
| Formatting and variadic arguments | 15      |
| Log filtering                     | 10      |
| File logging                      | 15      |
| Edge cases and robustness         | 15      |
| Code quality and documentation    | 10      |
| **Total**                         | **100** |

# **9. Completion criteria**

You're finished when:

- Your library compiles without warnings.
- All your tests pass.
- No memory leaks are detected.
- Invalid inputs don't cause unexpected crashes.
- The logger works without depending on your logger to report its own failures.
- You can use the error module from another C file.

Build using your `cc` compiler:

```
cc -Wall -Wextra -Wpedantic -std=c11 -I./include \
    src/error.c tests/test_error.c \
    -o test_error
```

Run:

```
./test_error
```

If you have Apple Clang, you can additionally use Address Sanitizer:

```
cc -Wall -Wextra -Wpedantic -std=c11 \
    -fsanitize=address -g -I./include \
    src/error.c tests/test_error.c \
    -o test_error
```

## **How this fits into your ML library**

Eventually your library should be able to do this:

```
CML
├── Memory       → allocates memory safely
├── Error        → standardizes failures
├── Logger       → helps debug failures
├── Math         → numerical operations
├── Vector       → vector operations
├── Matrix       → matrix operations
├── Statistics   → data analysis
├── Optimizer    → gradient descent
└── Neural Net   → model training
```

The goal isn't just to make a logger. It's to build infrastructure that every future module can depend on.

**Don't rush this assignment.** Design your API, write the header first, implement one function at a time, then test.

When you've finished, send me your `error.h`, `error.c`, `test_error.c`, and Makefile. I'll review your implementation like a code reviewer, evaluate it against the rubric, identify bugs and design weaknesses, and give you the next assignment based on what you've learned.

Your next assignment: build it yourself — no solution code yet. 🚀
