# cml — C Machine Learning Library

A from-scratch machine learning library written in C11.

```
CML
├── Memory        → allocates memory safely (Phase 0)
├── Error         → standardizes failures & logging (Phase 1)
├── Math          → numerical foundation & tolerances (Phase 2)
├── Vector        → vector engine & linear algebra (Phase 2)
├── Matrix        → 2D numerical data structures (Phase 3)
├── Statistics    → descriptive & inferential statistics (Phase 4)
├── Preprocessing → scaling, imputation, outlier detection (Phase 4)
├── Dataset       → datasets, shuffling, train/test splitting (Phase 4)
├── Regression    → closed-form simple linear regression (Phase 5)
├── Loss          → regression cost functions (Phase 6)
├── Classification → binary label and dataset validation (Phase 7)
├── Activation    → scalar and vector sigmoid (Phase 7)
└── ...
```

---

## Build & Test

```sh
# Build default target
make

# Run all test suites
make test

# Run specific test suites
make test-memory
make test-error
make test-vector
make test-matrix
make test-statistics
make test-preprocessing
make test-dataset
make test-linear-regression
make test-loss
make test-optimizer
make test-classification
make test-activation
make test-logistic
make test-network
make test-backpropagation

# Run example programs
make example
make example-vector
make example-matrix
make example-analysis
make example-linear-regression
make example-xor

# Run with AddressSanitizer & UndefinedBehaviorSanitizer
make sanitize
```

---

## Phase 9 · Backpropagation

Train dense networks with mean squared error, sigmoid binary cross-entropy,
or softmax categorical cross-entropy. The central-difference checker verifies
the analytical derivatives. Run `make example-xor` to see a two-layer network
learn the four XOR cases. [Phase 9 guide](docs/PHASE_9.md)

---

## Phase 8 · Dense Networks

Compose dense layers with identity, ReLU, sigmoid, tanh, or softmax. Network
weights are initialized from a seed. Single-sample and batch predictions use
the same forward pass. [Phase 8 guide](docs/PHASE_8.md)

---

## Phase 7 · Classification Foundation and Sigmoid

Binary targets live in `CMLDataset.y` as exact `0.0` and `1.0` values. The
classification validators borrow the data and can report both class counts.
Imbalanced and single-class datasets are valid at this stage. An empty or
inconsistent shape returns `CML_ERROR_INVALID_DIMENSION`. Missing storage or
a finite label outside the encoding returns `CML_ERROR_INVALID_ARGUMENT`.
NaN and infinity in labels or features return `CML_ERROR_NUMERICAL`.
Unaddressable dimensions return `CML_ERROR_OVERFLOW`.
[Classification API](include/cml/classification.h)

The sigmoid computes $1/(1+e^{-z})$ for nonnegative logits and
$e^z/(1+e^z)$ for negative logits, so the exponential always receives a
nonpositive argument for real inputs. The scalar and vector forms propagate
NaN and map negative and positive infinity to `0.0` and `1.0`. The vector
form can write to an equal-size output or update the input in place.
[Activation API](include/cml/activation.h)

### Logistic regression and evaluation

The binary classifier owns its weights and bias, stored in a single parameter
vector. Creation initializes zero parameters and a threshold of `0.5`. A
successful fit or parameter installation makes it ready to predict. The model
supports threshold changes, cloning, batch prediction, class weighting, and L1
and L2 penalties. Binary metrics provide a confusion matrix and six rates. ROC
evaluation groups tied scores and reports AUC and the best F1 threshold.

Multiclass classification supports one-vs-rest and joint softmax training with
integer class indices. [Phase 7 guide](docs/PHASE_7.md)

---

## Phase 6 · Loss Functions and Gradients

For $m$ samples, define the residual $r_i = x_i^T w + b - y_i$.
The mean squared error is $J = \frac{1}{m}\sum_i r_i^2$.
Its weight and intercept derivatives are
$\nabla_w J = \frac{2}{m}X^T r$ and
$\frac{\partial J}{\partial b} = \frac{2}{m}\sum_i r_i$.

`cml_linear_regression_mse_gradient` takes a parameter vector ordered as
weights followed by intercept and writes the gradient in the same order. It
accepts any positive number of feature columns. Use `cml_vector_norm_l2` to
measure the gradient magnitude. The loss API provides SSE, MSE, RMSE, and MAE
for paired vectors. [Loss API](include/cml/loss.h)

Loss and gradient calls preserve their outputs on failure. They return
`CML_ERROR_INVALID_ARGUMENT` for null pointers, `CML_ERROR_INVALID_DIMENSION`
for empty or incompatible shapes, and `CML_ERROR_NUMERICAL` for non-finite
inputs or overflow in the calculation. Gradient dimensions that exceed the
addressable size return `CML_ERROR_OVERFLOW`. Scratch allocation failures
return `CML_ERROR_OUT_OF_MEMORY`.

The optimizer supports generic objectives, reproducible full-batch, stochastic,
and mini-batch training, momentum, learning-rate schedules, gradient clipping,
and early stopping. A training observer receives epoch metrics and candidate
parameters. The central-difference checker verifies analytical gradients.
[Optimizer types](include/cml/optimizer.h)

---

## Phase 5 · Simple Linear Regression

Fit a model with `cml_linear_regression_fit` using an `N × D` feature
matrix and a target vector of length `N`. Multiple features use Householder
QR and require a full-rank design. The model owns its
coefficients and borrows the training data for the duration of the fit.
Use `cml_linear_regression_predict` with a one-element feature vector after a
successful fit. A constant feature yields a zero slope and the target mean.
A failed refit preserves the previous coefficients.

`make example-linear-regression` fits `y = 2x + 1` and predicts `21` for the
unseen input `10`. [Public API](include/cml/linear_regression.h)

---

## Phase 4 · Statistics & Data Processing

Comprises three integrated modules:
1. **Statistics** ([`include/cml/statistics.h`](include/cml/statistics.h), [`src/statistics.c`](src/statistics.c))
2. **Preprocessing** ([`include/cml/preprocessing.h`](include/cml/preprocessing.h), [`src/preprocessing.c`](src/preprocessing.c))
3. **Dataset** ([`include/cml/dataset.h`](include/cml/dataset.h), [`src/dataset.c`](src/dataset.c))

### Key Capabilities

1. **Descriptive Statistics:**
   - Vector metrics: `cml_stat_sum`, `cml_stat_mean`, `cml_stat_median`, `cml_stat_min`, `cml_stat_max`, `cml_stat_range`.
   - Numerically stable variance and standard deviation using Welford's algorithm (`cml_stat_variance`, `cml_stat_stddev`).
   - Order statistics: `cml_stat_percentile`, `cml_stat_quantile`, `cml_stat_quartiles` ($Q_1, Q_2, Q_3$), `cml_stat_iqr`.
   - Column-wise matrix statistics: `cml_stat_matrix_col_means`, `cml_stat_matrix_col_medians`, `cml_stat_matrix_col_stddev`, `cml_stat_matrix_col_iqr`.

2. **Covariance & Pearson Correlation:**
   - Sample and population covariance between vectors (`cml_stat_covariance`).
   - Pearson correlation coefficient $r \in [-1, 1]$ (`cml_stat_correlation`).
   - Pairwise covariance and correlation matrices for datasets (`cml_stat_covariance_matrix`, `cml_stat_correlation_matrix`).

3. **Data Quality & Missing Value Imputation:**
   - Detection: `cml_prep_count_nan`, `cml_prep_count_inf`, `cml_prep_count_nonfinite`, `cml_prep_is_finite`.
   - Imputation strategies: constant replacement, mean imputation, median imputation, and removal of invalid rows.

4. **Outlier Detection & Handling:**
   - Interquartile Range (IQR) method: $[Q_1 - k \cdot IQR, Q_3 + k \cdot IQR]$.
   - Z-score method: $|z| > \text{threshold}$.
   - Handling: Winsorization (`cml_prep_winsorize`), median replacement (`cml_prep_outliers_replace_median`).

5. **Feature Scalers (Data Leakage Prevention):**
   - Implements strict `fit` and `transform` interfaces.
   - `CMLMinMaxScaler`: maps features to $[a, b]$, handles constant features.
   - `CMLStandardScaler`: transforms to zero mean and unit variance ($z = (x - \mu) / \sigma$).
   - `CMLRobustScaler`: centers by median and scales by IQR.
   - Inverse transformation support for all scalers.
   - `CMLPipeline`: sequential chaining of imputation, clipping, and scaling.

6. **Dataset Architecture & Splitting:**
   - `CMLDataset`: encapsulates feature matrix $X$ and target vector $y$ with metadata.
   - Reproducible Fisher-Yates shuffling (`cml_dataset_shuffle`) with explicit 64-bit seed.
   - Disjoint train/validation/test splitting (`cml_dataset_split`) preserving feature-target alignment.
   - CSV dataset loader (`cml_dataset_from_csv`) and exporter (`cml_dataset_to_csv`).

---

## Phase 3 · Matrix Engine

Declared in [`include/cml/matrix.h`](include/cml/matrix.h), implemented in [`src/matrix.c`](src/matrix.c).

### `CMLMatrix` Structure

```c
typedef struct CMLMatrix {
    double *data;     /* contiguous dynamically allocated row-major array */
    size_t rows;      /* number of rows */
    size_t cols;      /* number of columns */
    size_t capacity;  /* allocated buffer capacity in elements */
} CMLMatrix;
```

**Memory Layout:**
Contiguous row-major storage. Element at row $r$ and column $c$ is indexed linearly via:
$$\text{index} = r \times \text{cols} + c$$

---

## Phase 2 · Vector Engine

Declared in [`include/cml/vector.h`](include/cml/vector.h) and [`include/cml/math.h`](include/cml/math.h), implemented in [`src/vector.c`](src/vector.c) and [`src/math.c`](src/math.c).

### `CMLVector` Structure

```c
typedef struct CMLVector {
    double *data;     /* contiguous dynamically allocated array */
    size_t size;      /* number of active elements */
    size_t capacity;  /* allocated buffer capacity */
} CMLVector;
```

---

## Phase 1 · Error & Logging Module

Declared in [`include/cml/error.h`](include/cml/error.h), implemented in [`src/error.c`](src/error.c).
- Standardized `CMLStatus` error codes.
- `cml_status_string()` for human-readable error descriptions.
- `cml_log()` printf-style variadic logger with severity levels (`DEBUG`, `INFO`, `WARN`, `ERROR`, `FATAL`).
- Runtime log level filtering (`cml_log_set_level()`).
- File logging & mirroring (`cml_log_set_file()`, `cml_log_close_file()`).

---

## Phase 0 · Memory Module

Declared in [`include/cml/memory.h`](include/cml/memory.h), implemented in [`src/memory.c`](src/memory.c).
- `cml_malloc`, `cml_calloc`, `cml_realloc`, `cml_free`, `cml_mem_copy`.
- Safe failure handling and multiplication overflow protection.
