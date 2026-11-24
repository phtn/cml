CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -std=c11
CPPFLAGS += -Iinclude
LDLIBS += -lm

BUILD_DIR ?= build
BLAS ?= none
ifeq ($(BLAS),accelerate)
CPPFLAGS += -DCML_CBLAS_ACCELERATE
LDLIBS += -framework Accelerate
else ifeq ($(BLAS),openblas)
ifeq ($(shell pkg-config --exists openblas && echo yes),)
$(error OpenBLAS was not found by pkg-config)
endif
CPPFLAGS += -DCML_CBLAS_OPENBLAS $(shell pkg-config --cflags openblas)
LDLIBS += $(shell pkg-config --libs openblas)
else ifneq ($(BLAS),none)
$(error BLAS must be none, accelerate, or openblas)
endif

LIBRARY := $(BUILD_DIR)/libcml.a
SOURCES := $(filter-out src/main.c,$(wildcard src/*.c))
OBJECTS := $(patsubst src/%.c,$(BUILD_DIR)/obj/%.o,$(SOURCES))
TESTS := memory error vector matrix statistics preprocessing dataset loss optimizer \
         linear_regression classification activation logistic network backpropagation \
         advanced_optimizers models neighbors
TEST_TARGETS := $(foreach name,$(TESTS),test-$(subst _,-,$(name)))

.PHONY: all clean test example sanitize benchmark $(TEST_TARGETS) \
        example-vector example-matrix example-analysis example-linear-regression \
        example-xor example-model-workflow
all: $(LIBRARY) $(BUILD_DIR)/cml

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/obj: | $(BUILD_DIR)
	mkdir -p $@

$(BUILD_DIR)/obj/%.o: src/%.c | $(BUILD_DIR)/obj
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c -o $@ $<

$(LIBRARY): $(OBJECTS)
	$(AR) rcs $@ $^

$(BUILD_DIR)/cml: $(BUILD_DIR)/obj/main.o $(LIBRARY)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test: $(TEST_TARGETS)

define test_target
test-$(subst _,-,$1): $(BUILD_DIR)/test_$1
	$$<
endef
$(foreach name,$(TESTS),$(eval $(call test_target,$(name))))

$(BUILD_DIR)/test_%: tests/test_%.c $(LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) -UNDEBUG $(LDFLAGS) -MMD -MP -MF $@.d -o $@ $< $(LIBRARY) $(LDLIBS)

example: example-vector example-matrix example-analysis example-linear-regression \
         example-xor example-model-workflow

example-vector: $(BUILD_DIR)/vector_example
example-matrix: $(BUILD_DIR)/matrix_example
example-analysis: $(BUILD_DIR)/data_analysis_example
example-linear-regression: $(BUILD_DIR)/linear_regression_example
example-xor: $(BUILD_DIR)/xor_example
example-model-workflow: $(BUILD_DIR)/model_workflow

example-vector example-matrix example-analysis example-linear-regression \
example-xor example-model-workflow:
	$<

$(BUILD_DIR)/%_example: examples/%_example.c $(LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -MMD -MP -MF $@.d -o $@ $< $(LIBRARY) $(LDLIBS)

$(BUILD_DIR)/model_workflow: examples/model_workflow.c $(LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -MMD -MP -MF $@.d -o $@ $< $(LIBRARY) $(LDLIBS)

$(BUILD_DIR)/benchmark_%: benchmarks/benchmark_%.c $(LIBRARY)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -MMD -MP -MF $@.d -o $@ $< $(LIBRARY) $(LDLIBS)

benchmark: $(BUILD_DIR)/benchmark_matrix $(BUILD_DIR)/benchmark_training
	$(BUILD_DIR)/benchmark_matrix
	$(BUILD_DIR)/benchmark_training

sanitize:
	ASAN_OPTIONS="allocator_may_return_null=1$${ASAN_OPTIONS:+:$${ASAN_OPTIONS}}" \
	$(MAKE) BUILD_DIR="$(BUILD_DIR)/sanitize" \
	        CFLAGS="$(CFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" \
	        LDFLAGS="$(LDFLAGS) -fsanitize=address,undefined" test

clean:
	rm -rf $(BUILD_DIR)

-include $(wildcard $(BUILD_DIR)/*.d $(BUILD_DIR)/obj/*.d)
