# **Phase 6 — Gradient Descent & Optimization**

In Phase 5, your linear regression model calculated the optimal parameters mathematically using least squares. In Phase 6, you'll teach CML to find optimal parameters through repeated updates, gradually minimizing a cost function.

This is a foundational phase. Everything you implement here will later be reused for logistic regression, neural networks, backpropagation, and advanced optimizers.

## **Phase 6 objectives**

By the end of this phase, your CML library should support:

- Cost/loss functions
- Analytical gradients
- Batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-batch Gradient Descent
- Learning rate management
- Convergence detection
- Training history
- Gradient checking
- Optimization diagnostics
- Configurable training
- Reproducible training

## **The core idea**

$ \theta\_{t+1}=\theta_t-\alpha\nabla J(\theta_t) $

Parameters are updated in the direction that reduces the loss.

Where:

$ \\\theta\ $ = model parameters (weights and bias)
$ \\\alpha\ $ = learning rate
$ \\J(\theta)\ $ = cost function
$ \\\nabla J(\theta)\ $ = gradient
$ \\t\ $ = current iteration

# **Assignment 01 — Optimization Foundation**

## **Objective**

Build a reusable optimization foundation that doesn't depend exclusively on linear regression.

Your optimizer should eventually be capable of optimizing arbitrary parameter vectors using supplied objective and gradient functions.

### **Requirements**

Design an optimizer API that supports:

- Optimization configuration
- Learning rate
- Maximum iterations
- Convergence tolerance
- Gradient tolerance
- Momentum configuration (later milestone)
- Optimizer state
- Training result information
- Error handling
- Cancellation or early stopping (bonus)

### **Your design task**

Create a proposed API and structures for:

- `CMLOptimizerConfig`
- `CMLOptimizer`
- `CMLOptimizationResult`
- `CMLObjectiveFunction`
- `CMLGradientFunction`

Decide whether the optimizer should be:

1. A generic optimization engine accepting callbacks.
2. A specialized linear regression optimizer.
3. Both, with a generic engine and model-specific wrappers.

**Recommended architecture: Option 3.**

This will make the optimizer reusable for future phases.

# **Assignment 02 — Cost Functions**

Before updating parameters, we need to know how bad the current predictions are.

## **2.1 Mean Squared Error**

$ J(\theta)=\frac{1}{m}\sum\_{i=1}^{m}(\hat y_i-y_i)^2 $

Implement:

- MSE
- SSE
- RMSE
- MAE

For linear regression:

$ \hat y_i=w x_i+b $

### **Test dataset**

Actual values:

$ y=[3,5,7,9] $

Predicted values:

$ \hat y=[2,6,8,10] $

Expected:

| Metric | Expected |
| ------ | -------- |
| SSE    | 4        |
| MSE    | 1        |
| RMSE   | 1        |
| MAE    | 1        |

Test additional cases:

- Perfect predictions
- Empty input
- Single sample
- Different vector sizes
- NaN and infinity
- Very large values
- Very small values

# **Assignment 03 — Derivatives & Gradients**

This is one of the most important assignments in CML.

You need to calculate how each parameter affects the loss.

For simple linear regression:

$ J(w,b)=\frac{1}{m}\sum\_{i=1}^{m}(wx_i+b-y_i)^2 $

The partial derivatives are:

$ \frac{\partial J}{\partial w} = \frac{2}{m}\sum\_{i=1}^{m}x_i(wx_i+b-y_i) $

$ \frac{\partial J}{\partial b} = \frac{2}{m}\sum\_{i=1}^{m}(wx_i+b-y_i) $

These gradients tell us how to change the weight and bias.

### **Requirements**

Implement:

- Gradient for weight
- Gradient for bias
- Combined gradient
- Gradient for multiple regression
- Gradient vector
- Gradient norm

### **Test**

Dataset:

$ X=[1,2,3] $

$ Y=[3,5,7] $

Initial parameters:

$ w=0,\quad b=0 $

Predictions:

$ \hat Y=[0,0,0] $

Calculate the gradients manually first, then compare your implementation against the expected values.

Your gradient implementation must be independent of the optimizer.

# **Assignment 04 — Batch Gradient Descent**

Implement your first complete optimizer.

## **Update rule**

Every iteration updates the parameters using the gradient calculated over the entire training dataset.

$ w\_{t+1}=w_t-\alpha\frac{\partial J}{\partial w} $

$ b\_{t+1}=b_t-\alpha\frac{\partial J}{\partial b} $

### **Requirements**

Implement training with:

- Initial weight and bias
- Learning rate
- Maximum iterations
- Cost evaluation
- Gradient computation
- Parameter updates
- Convergence detection
- Training status
- Final parameters

### **Experiment**

Train on:

$ X=[1,2,3,4,5] $

$ Y=[3,5,7,9,11] $

True parameters:

$ w=2,\quad b=1 $

Try these learning rates:

| Learning rate | Expected behavior                         |
| ------------- | ----------------------------------------- |
| 0.0001        | Very slow convergence                     |
| 0.001         | Slow but stable                           |
| 0.01          | Usually reasonable                        |
| 0.1           | May converge quickly depending on scaling |
| 1.0           | Likely unstable/divergent                 |

Observe how learning rate affects training.

### **Important**

Don't hardcode a learning rate that works for only one dataset.

Your optimizer must expose the learning rate as a configurable parameter.

# **Assignment 05 — Learning Rate & Stability**

Build a learning-rate management system.

### **Features**

- Constant learning rate
- Learning rate decay
- Step decay
- Exponential decay
- Optional adaptive learning rates (later optimizers)

Example decay:

$ \alpha_t=\frac{\alpha_0}{1+kt} $

Exponential decay:

$ \alpha_t=\alpha_0 e^{-kt} $

### **Investigate**

What happens when:

- Learning rate is too small?
- Learning rate is too large?
- Loss oscillates?
- Loss increases every iteration?
- Loss becomes infinity?
- Gradients become NaN?

Implement a strategy for detecting unstable training.

Possible policies:

- Return an error
- Stop training
- Roll back the last update
- Reduce learning rate
- Notify through a callback

Document which behavior your implementation supports.

# **Assignment 06 — Convergence Detection**

An optimizer shouldn't necessarily run every iteration.

Implement several convergence criteria.

### **1. Gradient tolerance**

$ \\|\nabla J(\theta)\\|<\epsilon $

### **2. Loss improvement tolerance**

$ |J_t-J\_{t-1}|<\epsilon $

### **3. Parameter change tolerance**

$ \\|\theta_t-\theta\_{t-1}\\|<\epsilon $

### **4. Patience**

Stop if improvement fails to occur for a configurable number of consecutive iterations.

### **Requirements**

Allow the user to configure:

- Maximum iterations
- Tolerance
- Patience
- Minimum iterations
- Convergence criteria
- Early stopping

You should distinguish between:

- `CONVERGED`
- `MAX_ITERATIONS`
- `DIVERGED`
- `ERROR`
- `CANCELLED`

Do not simply return `true` or `false` for all training outcomes.

# **Assignment 07 — Stochastic Gradient Descent (SGD)**

Batch Gradient Descent calculates gradients from all training samples.

SGD calculates gradients from a single sample.

$ \theta\_{t+1}=\theta_t-\alpha\nabla J_i(\theta_t) $

## **Requirements**

Implement:

- Random sample selection
- Sample-by-sample updates
- Configurable random seed
- Reproducible runs
- Epoch tracking
- Loss measurement per epoch
- Shuffling

### **Experiment**

Use the same dataset as Assignment 04.

Compare:

- Batch GD
- SGD

Record:

- Number of updates
- Number of epochs
- Final loss
- Final weight
- Final bias
- Training time

Understand why SGD has noisier updates but can be more computationally efficient for large datasets.

# **Assignment 08 — Mini-Batch Gradient Descent**

Mini-batch GD combines the advantages of batch GD and SGD.

Instead of using one sample or all samples, use a subset.

$ \nabla J_B(\theta) = \frac{1}{|B|}\sum\_{i\in B}\nabla J_i(\theta) $

### **Requirements**

Support configurable batch sizes:

- 1 → SGD
- 2
- 4
- 8
- 16
- 32
- 64
- Full dataset → Batch GD

Implement:

- Dataset shuffling
- Batch selection
- Last partial batch
- Gradient accumulation
- Parameter updates
- Epoch-level loss

### **Important edge cases**

Suppose the dataset has 100 samples and batch size is 32.

Your batches must contain:

- Batch 1: 32 samples
- Batch 2: 32 samples
- Batch 3: 32 samples
- Batch 4: 4 samples

You must not discard the last four samples.

# **Assignment 09 — Momentum**

Momentum helps reduce oscillation and accelerate convergence.

Instead of directly using the gradient:

$ v_t=\beta v\_{t-1}+(1-\beta)\nabla J(\theta_t) $

$ \theta\_{t+1}=\theta_t-\alpha v_t $

Alternatively, implement classical momentum:

$ v_t=\beta v\_{t-1}+\nabla J(\theta_t) $

$ \theta\_{t+1}=\theta_t-\alpha v_t $

Choose one formulation initially and document it.

### **Requirements**

- Momentum coefficient
- Velocity initialization
- Weight velocity
- Bias velocity
- Reset optimizer state
- Configuration validation

Experiment with:

$ \beta=0.0,\ 0.5,\ 0.9,\ 0.99 $

Compare the convergence speed and stability.

# **Assignment 10 — Training History**

Your optimizer should provide useful information about training.

Create a training history structure capable of recording:

- Epoch
- Iteration
- Training loss
- Validation loss
- Learning rate
- Gradient norm
- Parameter norm
- Weight values
- Bias
- Duration
- Convergence information

Example conceptual output:

```
Epoch:  1 | Loss: 18.24 | Grad: 8.21
Epoch: 10 | Loss:  2.41 | Grad: 2.31
Epoch: 50 | Loss:  0.08 | Grad: 0.32
Epoch:100 | Loss:  0.001 | Grad: 0.04
```

### **Requirements**

- Optional history collection
- Memory-efficient recording
- Configurable recording frequency
- History retrieval
- History cleanup
- No memory leaks

Bonus: produce a CSV training log.

# **Assignment 11 — Gradient Checking**

This is a critical assignment for validating your mathematical implementation.

A gradient implementation can compile, run, and produce believable numbers while being completely wrong.

You need an independent way to verify your analytical gradients.

## **Numerical approximation**

For a parameter \\(\theta\\):

$ \frac{\partial J}{\partial\theta} \approx \frac{J(\theta+h)-J(\theta-h)}{2h} $

Where \\(h\\) is a small value, such as \\(10^{-5}\\).

Compare:

- Analytical gradient
- Numerical gradient
- Absolute error
- Relative error

$ E\_{\text{abs}}=|g_a-g_n| $

$ E\_{\text{rel}} = \frac{|g_a-g_n|} {\max(|g_a|,|g_n|,\epsilon)} $

### **Requirements**

Create a gradient checker that supports:

- Vector parameters
- Multiple parameters
- Configurable epsilon
- Absolute tolerance
- Relative tolerance
- Per-parameter error reporting
- Overall pass/fail

### **Test**

Run gradient checking on:

- Linear regression weight
- Linear regression bias
- Multiple regression weights
- Random parameter vectors
- Different datasets

This assignment should catch bugs before you proceed to neural networks.

# **Assignment 12 — Multiple Linear Regression Optimization**

Extend your optimizer from simple linear regression to multiple features.

The model:

$ \hat y=Xw+b $

The gradients:

$ \frac{\partial J}{\partial w} = \frac{2}{m}X^T(Xw+b-y) $

$ \frac{\partial J}{\partial b} = \frac{2}{m}\sum\_{i=1}^{m}(\hat y_i-y_i) $

### **Requirements**

- Arbitrary feature count
- Weight vector
- Bias
- Batch GD
- SGD
- Mini-batch GD
- Momentum
- Gradient checking
- Training history

### **Test dataset**

Use:

$ y=5x_1+3x_2-2x_3+10 $

Generate enough data with different feature values to learn the coefficients.

Expected:

| Parameter | Expected |
| --------- | -------- |
| \\(w_1\\) | 5        |
| \\(w_2\\) | 3        |
| \\(w_3\\) | -2       |
| Bias      | 10       |

The trained model should approach these values within your chosen tolerance.

# **Assignment 13 — Feature Scaling & Optimization Behavior**

Investigate why gradient descent can behave poorly when features have drastically different scales.

Example:

| Feature    | Values   |
| ---------- | -------- |
| House size | 500–5000 |
| Bedrooms   | 1–6      |
| Age        | 0–100    |

These features operate on very different scales.

The gradient may become dominated by the large-scale feature, making optimization inefficient.

### **Requirements**

Integrate with your Phase 4 preprocessing module.

Test training with:

1. Raw features
2. Min-max scaled features
3. Standardized features

Compare:

- Number of iterations
- Convergence speed
- Loss
- Final parameters
- Stability

Important: fit your scaler on training data only.

Do not introduce data leakage by scaling the full dataset before splitting.

# **Assignment 14 — Validation & Early Stopping**

Implement validation-based training.

Split your dataset into:

- Training data
- Validation data
- Testing data

The validation set is used to determine when training stops and which model parameters are best.

### **Early stopping**

If validation loss stops improving for `patience` epochs:

1. Stop training.
2. Restore the best model parameters.
3. Report the best epoch.
4. Report final training and validation metrics.

Example:

```
Epoch  10: train=0.25 validation=0.31
Epoch  20: train=0.10 validation=0.16
Epoch  30: train=0.04 validation=0.11
Epoch  40: train=0.01 validation=0.13
Epoch  50: train=0.002 validation=0.18

Early stopping triggered.

Best epoch: 30
Best validation loss: 0.11
```

### **Requirements**

- Best parameter snapshot
- Patience counter
- Validation loss evaluation
- Restore best parameters
- Configurable minimum improvement
- Prevent unnecessary allocation every epoch

# **Assignment 15 — Optimization Diagnostics**

Create tools to understand optimizer failures.

Your optimizer should help identify:

| Problem                | Symptoms                                        |
| ---------------------- | ----------------------------------------------- |
| Learning rate too high | Loss explodes                                   |
| Learning rate too low  | Progress extremely slow                         |
| Bad scaling            | Uneven convergence                              |
| Vanishing gradients    | Very small gradient                             |
| Exploding gradients    | Extremely large gradient                        |
| Poor initialization    | Slow convergence                                |
| Noisy SGD              | Loss fluctuates                                 |
| Overfitting            | Train loss decreases, validation loss increases |

### **Implement diagnostics**

- Gradient norm
- Parameter norm
- Loss change
- Learning rate
- Divergence detection
- NaN/Inf detection
- Gradient clipping (bonus)

### **Gradient clipping**

Implement norm-based clipping:

$ g\_{\text{clipped}}= \begin{cases} g & \\|g\\|\leq c\\\ \frac{c}{\\|g\\|}g & \\|g\\|>c \end{cases} $

Where \\(c\\) is the maximum gradient norm.

This becomes especially useful in neural networks.

# **Assignment 16 — Optimizer API & Library Architecture**

Keep optimization separate from model implementations.

Suggested new project structure:

```
cml/
├── include/cml/
│   ├── optimizer.h
│   ├── gradient_descent.h
│   ├── loss.h
│   ├── training.h
│   └── linear_regression.h
├── src/
│   ├── optimizer.c
│   ├── gradient_descent.c
│   ├── loss.c
│   ├── training.c
│   └── linear_regression.c
├── tests/
│   ├── test_optimizer.c
│   ├── test_gradient_descent.c
│   ├── test_loss.c
│   ├── test_gradient_check.c
│   └── test_training.c
├── examples/
│   ├── batch_gradient_descent.c
│   ├── stochastic_gradient_descent.c
│   └── mini_batch.c
└── benchmarks/
    └── benchmark_optimizers.c
```

## **Architectural responsibilities**

| Component   | Responsibility                   |
| ----------- | -------------------------------- |
| Loss        | Calculates model error           |
| Gradient    | Calculates parameter derivatives |
| Optimizer   | Updates parameters               |
| Training    | Coordinates training iterations  |
| History     | Stores training metrics          |
| Diagnostics | Detects optimization issues      |
| Model       | Defines prediction behavior      |

Avoid tightly coupling gradient descent to your linear regression model. The optimizer should eventually be reusable by logistic regression and neural networks.

# **Assignment 17 — Benchmarking & Performance**

Test your optimizer with datasets of varying sizes.

| Dataset size | Purpose                 |
| ------------ | ----------------------- |
| 10           | Basic correctness       |
| 100          | Small dataset           |
| 1,000        | Typical                 |
| 10,000       | Medium                  |
| 100,000      | Large                   |
| 1,000,000    | Performance stress test |

Compare:

- Batch GD
- SGD
- Mini-batch GD
- Mini-batch size 8
- Mini-batch size 32
- Mini-batch size 64
- Mini-batch size 256

Measure:

- Training time
- Memory usage
- Number of iterations
- Final loss
- Convergence speed

For fair comparisons, use the same dataset, initialization, and convergence criteria where applicable.

# **Assignment 18 — Capstone**

## **Build a complete training workflow**

Use a real or synthetic dataset.

Example: house price prediction.

Features:

- House size
- Bedrooms
- Age
- Distance to city center

Target:

\- House price

Your application should:

1. Load the dataset.
2. Validate the data.
3. Handle missing values.
4. Shuffle reproducibly.
5. Split training, validation, and testing sets.
6. Fit preprocessing on training data.
7. Transform validation/test data.
8. Initialize model parameters.
9. Configure the optimizer.
10. Train the model.
11. Record training history.
12. Evaluate validation performance.
13. Apply early stopping.
14. Restore the best model.
15. Evaluate on test data.
16. Compare with Phase 5 closed-form regression.
17. Print model coefficients.
18. Print metrics.
19. Save the model.
20. Load the saved model and verify predictions.

### **Example final report**

```
CML — Linear Regression Training Report
======================================

Optimizer: Mini-Batch Gradient Descent
Learning Rate: 0.01
Batch Size: 32
Epochs: 500
Best Epoch: 342

Training MSE:   0.00214
Validation MSE: 0.00251
Testing MSE:    0.00267

R²: 0.9821

Convergence: CONVERGED

Weights:
  Size:       0.00421
  Bedrooms:   0.08731
  Age:       -0.02152
  Distance:  -0.13042

Bias: 124.482
```

# **Phase 6 grading rubric**

| Category                     | Points  |
| ---------------------------- | ------- |
| Cost functions               | 10      |
| Analytical gradients         | 15      |
| Batch GD                     | 15      |
| SGD                          | 10      |
| Mini-batch GD                | 10      |
| Momentum                     | 10      |
| Convergence & early stopping | 10      |
| Gradient checking            | 10      |
| Numerical stability          | 5       |
| Tests, documentation & API   | 5       |
| **Total**                    | **100** |

Bonus points:

- Gradient clipping
- Learning-rate schedules
- CSV training history
- Generic optimizer callbacks
- Validation-based early stopping
- Model checkpointing
- Performance benchmarks
- Adam optimizer (optional advanced extension)

# **Recommended execution order**

Don't attempt to implement all 18 assignments simultaneously. We'll work through them in manageable milestones and review your implementation before advancing.

## **Phase 6 milestones**

1. **M1 — Loss functions & gradient mathematics** (Assignments 1–3)
2. **M2 — Batch gradient descent** (Assignment 4)
3. **M3 — Learning rate & convergence** (Assignments 5–6)
4. **M4 — SGD & mini-batch** (Assignments 7–8)
5. **M5 — Momentum** (Assignment 9)
6. **M6 — Training history** (Assignment 10)
7. **M7 — Gradient checking** (Assignment 11)
8. **M8 — Multiple regression** (Assignment 12)
9. **M9 — Scaling & validation** (Assignments 13–14)
10. **M10 — Diagnostics** (Assignment 15)
11. **M11 — Architecture & benchmarks** (Assignments 16–17)
12. **M12 — Capstone** (Assignment 18)

## **Your first task: M1**

Start by designing the API. Submit:

1. `loss.h`
2. `optimizer.h`
3. Your proposed optimizer and result structures
4. The mathematical formulas you plan to implement for MSE and gradients
5. Your error-handling strategy
6. Your tests for loss functions and gradient calculations


## Running the implementation

Run the optimizer, gradient checking, and closed-form comparison tests with
`make test-optimizer`. The generic optimizer accepts separate objective and
gradient callbacks. For data-based training, `cml_train` supplies complete or
partial batches of indices to a callback, with deterministic shuffling.

An epoch processes each sample once unless the update limit is reached first.
The training observer receives losses, learning rate, norms, and borrowed
parameters. It can collect history, write a CSV, checkpoint a best candidate,
or cancel. Validation keeps the lowest-loss parameters, including epoch zero.
Patience counts epochs without an improvement above the configured minimum.

`cml_linear_regression_fit_iterative` uses mean squared error with parameters
in weight-then-intercept order. Closed-form multiple regression uses scaled
Householder QR and reports a numerical error for a rank-deficient design.
Both fits preserve existing parameters on failure. Gradient checking uses
central differences with a step proportional to each parameter's magnitude.
