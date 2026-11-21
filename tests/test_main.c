#include <stdio.h>
#include <assert.h>

void test_example(void) {
    assert(1 + 1 == 2);
    printf("  PASS: test_example\n");
}

int main(void) {
    printf("Running tests...\n");
    test_example();
    printf("All tests passed!\n");
    return 0;
}
