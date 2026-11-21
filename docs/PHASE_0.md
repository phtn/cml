# **Phase 0 · Assignment 01**

## **Build your C Core Utilities — Memory Module**

We're starting your ML library from the ground up. Your first assignment is to build a small **memory management module** that safely allocates, resizes, and frees memory.

This is intentionally not an ML algorithm yet. Almost everything you'll eventually build—vectors, matrices, tensors, datasets, neural-network weights—will rely on allocating memory correctly.

**Your goal:** write the implementation yourself. I'll review your code, test your logic, identify bugs, and explain improvements without taking away the learning process.

## **1. Project setup**

Create a project named `cml` (C Machine Learning Library).

Your initial structure:

```
cml/
├── include/
│   └── cml/
│       └── memory.h
├── src/
│   └── memory.c
├── tests/
│   └── test_memory.c
├── examples/
├── Makefile
└── README.md
```

You may use a different structure if you have a good reason, but keep header files separate from implementation files.

## **2. Assignment requirements**

Implement the following five functions.

### **Function 1 —&#x20;**`cml_malloc`

Allocate memory for a requested number of bytes.

**Requirements:**

- Accept a byte count.
- Return a pointer to the allocated memory.
- Handle allocation failure.
- Handle zero-byte allocation consistently.
- Do not assume allocation succeeds.

Think about what your library should do if the user requests 1 GB of memory and the allocation fails.

### **Function 2 —&#x20;**`cml_calloc`

Allocate memory for an array of elements and initialize all allocated bytes to zero.

Signature concept:

```
pointer = cml_calloc(count, element_size);
```

**Requirements:**

- Accept an element count.
- Accept the size of each element.
- Ensure all returned bytes are zero-initialized.
- Detect multiplication overflow before calculating `count * element_size`.
- Handle zero count or zero element size consistently.
- Handle allocation failure.

Important: this is your first encounter with integer overflow in memory allocation. If the count and element size are huge, their multiplication may overflow and produce a much smaller allocation than expected.

### **Function 3 —&#x20;**`cml_realloc`

Resize an existing allocation.

**Requirements:**

- Accept a previously allocated pointer.
- Accept the requested new size.
- Support growing and shrinking memory.
- Handle allocation failure safely.
- Preserve the original data when resizing succeeds.
- Never lose the original pointer if resizing fails.
- Decide how resizing to zero behaves and document it.

This function is especially important because it introduces one of the most common C memory-management mistakes.

### **Function 4 —&#x20;**`cml_free`

Release allocated memory.

**Requirements:**

- Accept a pointer.
- Safely handle `NULL`.
- Never attempt to free memory that your function didn't receive as a valid allocated pointer.
- Decide whether your wrapper should simply call `free` or provide additional behavior.

### **Function 5 —&#x20;**`cml_mem_copy`

Copy a specified number of bytes from one memory location to another.

**Requirements:**

- Accept destination, source, and byte count.
- Return the destination pointer or an appropriate error.
- Handle zero bytes.
- Document overlapping memory behavior.
- Do not copy more bytes than the destination can hold (consider what your API can and cannot know).

Hint: the C standard library already has a function for copying memory, but you should build a wrapper around it with your own API conventions.

# **3. API design constraints**

Your library should use consistent naming:

```
cml_<module>_<operation>
```

For this assignment:

```
cml_malloc
cml_calloc
cml_realloc
cml_free
cml_mem_copy
```

Choose your return types and parameters carefully.

### **Things you must decide**

1. Should your allocation functions return `void *` or a typed pointer?
2. Should failure return `NULL`, an error code, or both?
3. Should your functions print error messages, or should the caller decide what to do?
4. Should you use `size_t` for sizes? Why?
5. Should your library allow zero-byte allocations?
6. Should you expose the standard library's behavior directly, or normalize behavior across platforms?

**Recommended design principle:** your library should not unexpectedly print errors or terminate the program. Let the caller control how errors are handled.

# **4. Constraints**

These rules apply to this assignment.

- Language: C (C11 or newer)
- No C++
- No external libraries
- You may use the standard C library
- No global mutable state
- No memory leaks
- No undefined behavior
- No unnecessary type casts
- No hardcoded allocation sizes
- Use `size_t` for byte counts
- All public functions must be declared in `memory.h`
- All implementation details must live in `memory.c`
- Every function must have a documented contract
- You must compile with warnings enabled

Recommended compiler flags:

```
-Wall -Wextra -Wpedantic -Wconversion -Wshadow
```

During development, also use sanitizers when available:

```
-fsanitize=address,undefined
```

Don't worry if some warnings arise from your compiler or platform. Your objective is to understand and resolve warnings rather than blindly suppress them.

# **5. Test cases**

Create a test suite in `tests/test_memory.c`.

You should write these tests yourself.

## **A.&#x20;**`cml_malloc`

| Test                      | Expected behavior                      |
| ------------------------- | -------------------------------------- |
| Allocate 1 byte           | Valid allocation or documented failure |
| Allocate 64 bytes         | Valid allocation                       |
| Allocate 1 MB             | Valid allocation or safe failure       |
| Allocate 0 bytes          | Consistent documented behavior         |
| Write to allocated memory | Memory can be accessed within bounds   |
| Free allocated memory     | No sanitizer errors                    |

## **B.&#x20;**`cml_calloc`

| Test                      | Expected behavior             |
| ------------------------- | ----------------------------- |
| Allocate 10 integers      | All bytes initialized to zero |
| Allocate 100 bytes        | All bytes initialized to zero |
| Count = 0                 | Documented behavior           |
| Element size = 0          | Documented behavior           |
| Huge count × element size | Overflow safely detected      |
| Allocation failure        | Safely handled                |

For the zero initialization test, don't just check one element. Verify the entire allocated region.

## **C.&#x20;**`cml_realloc`

| Test                       | Expected behavior                  |
| -------------------------- | ---------------------------------- |
| Grow 8 bytes to 64 bytes   | Original 8 bytes preserved         |
| Shrink 64 bytes to 8 bytes | First 8 bytes preserved            |
| Resize to same size        | Data remains valid                 |
| Resize with `NULL` pointer | Behaves consistently with your API |
| Resize to zero             | Documented behavior                |
| Allocation failure         | Original allocation remains usable |

### **Important failure test**

You need to test the behavior of a failed resize.

This is the scenario:

1. Allocate 16 bytes.
2. Fill them with known data.
3. Attempt to resize to an extremely large amount.
4. If resizing fails, verify the original allocation still exists.
5. Verify the original data is still intact.
6. Free the original allocation.

Don't assume a huge allocation must fail on every operating system. Memory overcommit can make allocation behavior surprising. Design your API so it is safe regardless.

## **D.&#x20;**`cml_free`

| Test                    | Expected behavior |
| ----------------------- | ----------------- |
| Free a valid allocation | No errors         |
| Free `NULL`             | Safe              |
| Free after realloc      | Safe              |
| Free after calloc       | Safe              |

Do not test double-free by actually triggering undefined behavior in your normal tests. Instead, your code should clearly document the ownership rules.

## **E.&#x20;**`cml_mem_copy`

| Test                     | Expected behavior          |
| ------------------------ | -------------------------- |
| Copy 1 byte              | Correct                    |
| Copy 8 bytes             | Correct                    |
| Copy 100 bytes           | Correct                    |
| Copy 0 bytes             | Safe                       |
| Copy into another buffer | Destination matches source |
| Copy from constant data  | Source remains unchanged   |

For overlap behavior, decide whether your API supports overlapping memory regions. If it does, choose the correct underlying standard-library function.

# **6. Hints (without giving you the solution)**

These are hints for where to focus your thinking.

**Hint 1 — Allocation functions**

Look at the standard library's `stdlib.h`. Understand what `malloc`, `calloc`, `realloc`, and `free` actually return and how they behave on failure.

**Hint 2 — Overflow**

Before multiplying two unsigned sizes, think about whether the result could exceed the maximum value representable by `size_t`.

There is a limit defined in `stdint.h` that will be useful.

**Hint 3 — Reallocation**

`realloc` has a special property: on failure, the original allocation remains valid. Your implementation must not accidentally destroy that guarantee.

**Hint 4 — Zero bytes**

Standard C allocation behavior around zero-sized allocations has subtleties. Decide what your library promises instead of leaving behavior undocumented.

**Hint 5 — Copying memory**

Understand the difference between copying bytes and copying strings. Your memory-copy function should not assume null terminators.

**Hint 6 — Tests**

A test should verify the result, not merely that the program didn't crash.

For example, after copying, compare every byte—not just the first byte.

# **7. Documentation requirement**

In your `README.md`, document each function with:

- What it does
- Parameters
- Return value
- Failure behavior
- Edge-case behavior
- Ownership rules
- Example usage

Your documentation should answer this question:

> If another developer uses my library, can they use my memory functions without reading my implementation?

# **8. Bonus challenges (optional)**

Don't start these until the main assignment is completed.

### **Bonus A — Allocation statistics**

Track:

- Total allocations
- Total frees
- Currently active allocations
- Total bytes allocated
- Peak memory usage

### **Bonus B — Memory duplication**

Implement a function conceptually similar to:

```
cml_mem_duplicate(source, size);
```

It allocates new memory and copies the contents.

### **Bonus C — Memory comparison**

Implement a wrapper that compares two memory regions byte by byte.

### **Bonus D — Debug mode**

Create an optional debug configuration that tracks allocations and reports allocations that were never freed when the program exits.

This will eventually be extremely useful when your ML library begins allocating hundreds of matrices, gradients, and model parameters.

# **9. How I'll review your submission**

I'll review your code like a code review for a real C library.

| Category                     | Points  |
| ---------------------------- | ------- |
| Correctness                  | 25      |
| Memory safety                | 25      |
| Edge-case handling           | 15      |
| API design                   | 15      |
| Test quality                 | 10      |
| Documentation & code quality | 10      |
| **Total**                    | **100** |

I'll look for things like:

- Memory leaks
- Use-after-free
- Integer overflow
- Incorrect `realloc` usage
- Undefined behavior
- Missing edge cases
- API inconsistency
- Unnecessary complexity
- Poor naming
- Compiler warnings

I won't just fix your mistakes. I'll explain **why they are mistakes, what underlying C concept they reveal, and how you can fix them yourself**.

# **10. Your submission**

When you finish, send me these files:

- `include/cml/memory.h`
- `src/memory.c`
- `tests/test_memory.c`
- `Makefile`
- `README.md`

You can paste the files directly here, or upload your project as a ZIP.

## **Assignment 01**

**Build a safe memory utility module.**

Don't rush. Try implementing each function yourself, compile frequently, and write tests as you go.

Your first goal isn't to write perfect code. It's to understand every line you write.

Once you submit your implementation, I'll review it, score it out of 100, give you hints for corrections, and then we'll move to Assignment 02: **Error Handling & Logging**.
