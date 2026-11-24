#include "cml/optimizer.h"
#include "cml/memory.h"
#include "internal.h"

#include <float.h>
#include <string.h>

struct CMLOptimizer {
  CMLOptimizerConfig config;
};

static bool valid_config(const CMLOptimizerConfig *c) {
  return c != NULL && isfinite(c->learning_rate) && c->learning_rate > 0.0 &&
         c->max_iterations > 0 && isfinite(c->loss_tolerance) &&
         c->loss_tolerance >= 0.0 && isfinite(c->gradient_tolerance) &&
         c->gradient_tolerance >= 0.0;
}

CMLOptimizer *cml_optimizer_create(const CMLOptimizerConfig *config) {
  if (!valid_config(config))
    return NULL;
  CMLOptimizer *optimizer = cml_malloc(sizeof(*optimizer));
  if (optimizer != NULL)
    optimizer->config = *config;
  return optimizer;
}

void cml_optimizer_destroy(CMLOptimizer *optimizer) { cml_free(optimizer); }

CMLTrainingOptions cml_training_options_default(void) {
  return (CMLTrainingOptions){
      .optimizer = {.learning_rate = 0.1, .max_iterations = 10000},
      .epochs = 1000,
      .record_every = 1,
      .beta1 = 0.9,
      .beta2 = 0.999,
      .epsilon = 1e-8};
}

static bool valid_options(const CMLTrainingOptions *o) {
  if (o == NULL || !valid_config(&o->optimizer) || o->epochs == 0 ||
      o->min_epochs > o->epochs || !isfinite(o->momentum) ||
      o->momentum < 0.0 || o->momentum >= 1.0 || !isfinite(o->clip_norm) ||
      o->clip_norm < 0.0 || !isfinite(o->decay) || o->decay < 0.0 ||
      !isfinite(o->min_improvement) || o->min_improvement < 0.0 ||
      !isfinite(o->parameter_tolerance) || o->parameter_tolerance < 0.0)
    return false;
  if (o->method < CML_OPTIMIZER_SGD || o->method > CML_OPTIMIZER_ADAM)
    return false;
  if (o->method != CML_OPTIMIZER_SGD &&
      (o->momentum != 0.0 || !isfinite(o->epsilon) || o->epsilon <= 0.0))
    return false;
  if ((o->method == CML_OPTIMIZER_RMSPROP || o->method == CML_OPTIMIZER_ADAM) &&
      (!isfinite(o->beta2) || o->beta2 < 0.0 || o->beta2 >= 1.0))
    return false;
  if (o->method == CML_OPTIMIZER_ADAM &&
      (!isfinite(o->beta1) || o->beta1 < 0.0 || o->beta1 >= 1.0))
    return false;
  if (o->history != NULL &&
      (o->history->size > o->history->capacity ||
       (o->history->capacity != 0 && o->history->records == NULL)))
    return false;
  switch (o->schedule) {
  case CML_RATE_CONSTANT:
  case CML_RATE_INVERSE_TIME:
  case CML_RATE_EXPONENTIAL:
    return true;
  case CML_RATE_STEP:
    return o->decay > 0.0 && o->decay <= 1.0 && o->decay_step > 0;
  }
  return false;
}

static double learning_rate(const CMLTrainingOptions *options, size_t epoch) {
  double rate = options->optimizer.learning_rate;
  switch (options->schedule) {
  case CML_RATE_CONSTANT:
    return rate;
  case CML_RATE_INVERSE_TIME:
    return rate / (1.0 + options->decay * (double)epoch);
  case CML_RATE_STEP:
    return rate * pow(options->decay, (double)(epoch / options->decay_step));
  case CML_RATE_EXPONENTIAL:
    return rate * exp(-options->decay * (double)epoch);
  }
  return NAN;
}

static CMLStatus evaluate(CMLBatchFunction batch, const CMLVector *candidate,
                          size_t count, void *context, double *loss,
                          CMLVector *gradient, double *norm) {
  CMLStatus status = batch(candidate, NULL, count, context, loss, gradient);
  if (status != CML_SUCCESS)
    return status;
  if (!isfinite(*loss))
    return CML_ERROR_NUMERICAL;
  return gradient == NULL ? CML_SUCCESS : cml_vector_norm_l2(gradient, norm);
}

void cml_training_history_destroy(CMLTrainingHistory *history) {
  if (history != NULL) {
    cml_free(history->records);
    *history = (CMLTrainingHistory){0};
  }
}

static CMLStatus append_record(CMLTrainingHistory *history,
                               const CMLTrainingRecord *record) {
  if (history->size == history->capacity) {
    const size_t max = SIZE_MAX / sizeof(*history->records);
    if (history->capacity > max / 2)
      return CML_ERROR_OVERFLOW;
    size_t capacity = history->capacity == 0 ? 16 : history->capacity * 2;
    void *records =
        cml_realloc(history->records, capacity * sizeof(*history->records));
    if (records == NULL)
      return CML_ERROR_OUT_OF_MEMORY;
    history->records = records;
    history->capacity = capacity;
  }
  history->records[history->size++] = *record;
  return CML_SUCCESS;
}

static double bias_correction(double beta, size_t step) {
  return beta == 0.0 ? 1.0 : -expm1((double)step * log(beta));
}

static CMLStatus update(CMLVector *candidate, CMLVector *gradient,
                        CMLVector *first, CMLVector *second, double rate,
                        size_t step, const CMLTrainingOptions *o,
                        double *change) {
  double norm;
  CMLStatus status = cml_vector_norm_l2(gradient, &norm);
  if (status != CML_SUCCESS)
    return status;
  double scale =
      o->clip_norm > 0.0 && norm > o->clip_norm ? o->clip_norm / norm : 1.0;
  double decay = 1.0, new_weight = 1.0, correction1 = 1.0, correction2 = 1.0;
  if (o->method == CML_OPTIMIZER_RMSPROP || o->method == CML_OPTIMIZER_ADAM) {
    decay = sqrt(o->beta2);
    new_weight = sqrt(1.0 - o->beta2);
  }
  if (o->method == CML_OPTIMIZER_ADAM) {
    correction1 = bias_correction(o->beta1, step);
    correction2 = sqrt(bias_correction(o->beta2, step));
  }
  for (size_t j = 0; j < candidate->size; ++j) {
    double direction = scale * gradient->data[j];
    if (second != NULL) {
      double moment = hypot(decay * second->data[j], new_weight * direction);
      if (!isfinite(moment))
        return CML_ERROR_NUMERICAL;
      second->data[j] = moment;
      if (o->method == CML_OPTIMIZER_ADAM) {
        direction = fma(o->beta1, first->data[j], (1.0 - o->beta1) * direction);
        first->data[j] = direction;
      }
      double epsilon = o->epsilon * correction2;
      /* Scale the denominator if its finite terms would overflow when added. */
      if (moment > DBL_MAX - epsilon)
        direction = (direction * 0.5) / (moment * 0.5 + epsilon * 0.5);
      else if (moment == 0.0 && epsilon == 0.0 && direction == 0.0)
        direction = 0.0;
      else
        direction /= moment + epsilon;
      direction *= correction2 / correction1;
    } else if (first != NULL) {
      direction = fma(o->momentum, first->data[j], direction);
      first->data[j] = direction;
    }
    double before = candidate->data[j];
    double after = fma(-rate, direction, before);
    if (!isfinite(direction) || !isfinite(after))
      return CML_ERROR_NUMERICAL;
    candidate->data[j] = after;
    *change = hypot(*change, after - before);
  }
  return CML_SUCCESS;
}

CMLStatus cml_train(CMLVector *parameters, CMLBatchFunction batch,
                    size_t training_count, void *training_context,
                    size_t validation_count, void *validation_context,
                    const CMLTrainingOptions *options, CMLTrainingResult *out) {
  CMLTrainingResult result = {
      .optimization = {.status = CML_OPTIMIZATION_ERROR,
                       .error = CML_ERROR_INVALID_ARGUMENT,
                       .initial_loss = NAN,
                       .final_loss = NAN,
                       .gradient_norm = NAN},
      .validation_loss = NAN};
  if (out != NULL)
    *out = result;
  if (parameters == NULL || batch == NULL || !valid_options(options))
    return CML_ERROR_INVALID_ARGUMENT;
  if (parameters->size == 0 || training_count == 0) {
    if (out != NULL)
      out->optimization.error = CML_ERROR_INVALID_DIMENSION;
    return CML_ERROR_INVALID_DIMENSION;
  }
  CMLStatus status = cml_vector_storage(parameters, parameters->size);
  if (status != CML_SUCCESS) {
    if (out != NULL)
      out->optimization.error = status;
    return status;
  }

  size_t batch_size = options->batch_size;
  if (batch_size == 0 || batch_size > training_count)
    batch_size = training_count;
  bool indexed = options->shuffle || batch_size < training_count;
  if (indexed && training_count > SIZE_MAX / sizeof(size_t)) {
    if (out != NULL)
      out->optimization.error = CML_ERROR_OVERFLOW;
    return CML_ERROR_OVERFLOW;
  }
  CMLVector *candidate = cml_vector_clone(parameters);
  CMLVector *gradient = cml_vector_create(parameters->size);
  bool need_first =
      options->momentum > 0.0 || options->method == CML_OPTIMIZER_ADAM;
  bool need_second = options->method != CML_OPTIMIZER_SGD;
  CMLVector *first = need_first ? cml_vector_create(parameters->size) : NULL;
  CMLVector *second = need_second ? cml_vector_create(parameters->size) : NULL;
  CMLVector *best = validation_count > 0 ? cml_vector_clone(parameters) : NULL;
  size_t *indices =
      indexed ? cml_calloc(training_count, sizeof(*indices)) : NULL;
  if (candidate == NULL || gradient == NULL || (need_first && first == NULL) ||
      (need_second && second == NULL) ||
      (validation_count > 0 && best == NULL) || (indexed && indices == NULL)) {
    status = CML_ERROR_OUT_OF_MEMORY;
    goto done;
  }
  if (!cml_values_finite(candidate->data, candidate->size)) {
    status = CML_ERROR_NUMERICAL;
    goto done;
  }

  for (size_t i = 0; indexed && i < training_count; ++i)
    indices[i] = i;
  size_t bytes = parameters->size * sizeof(double);
  uint64_t seed = options->seed;
  status = evaluate(batch, candidate, training_count, training_context,
                    &result.optimization.initial_loss, gradient,
                    &result.optimization.gradient_norm);
  if (status != CML_SUCCESS)
    goto done;
  result.optimization.final_loss = result.optimization.initial_loss;
  if (best != NULL) {
    status = evaluate(batch, candidate, validation_count, validation_context,
                      &result.validation_loss, NULL, NULL);
    if (status != CML_SUCCESS)
      goto done;
  }
  double patience_loss =
      best != NULL ? result.validation_loss : result.optimization.initial_loss;
  double checkpoint_loss = patience_loss;
  if (options->checkpoint != NULL) {
    CMLTrainingRecord initial = {
        .training_loss = result.optimization.initial_loss,
        .validation_loss = result.validation_loss,
        .gradient_norm = result.optimization.gradient_norm,
        .learning_rate = options->optimizer.learning_rate,
        .best = true};
    status = cml_vector_norm_l2(candidate, &initial.parameter_norm);
    if (status == CML_SUCCESS)
      status =
          options->checkpoint(&initial, candidate, options->checkpoint_context);
    if (status != CML_SUCCESS)
      goto done;
  }
  size_t stale_epochs = 0;
  const CMLOptimizerConfig *config = &options->optimizer;
  result.optimization.status = CML_OPTIMIZATION_MAX_ITERATIONS;
  if (options->min_epochs == 0 && config->gradient_tolerance > 0.0 &&
      result.optimization.gradient_norm <= config->gradient_tolerance)
    result.optimization.status = CML_OPTIMIZATION_CONVERGED;

  for (size_t epoch = 0;
       epoch < options->epochs &&
       result.optimization.iterations < config->max_iterations &&
       result.optimization.status != CML_OPTIMIZATION_CONVERGED;
       ++epoch) {
    double rate = learning_rate(options, epoch);
    if (options->shuffle)
      cml_shuffle_indices(indices, training_count, &seed);
    double change = 0.0;
    for (size_t start = 0;
         start < training_count &&
         result.optimization.iterations < config->max_iterations;) {
      size_t remaining = training_count - start;
      size_t count = batch_size < remaining ? batch_size : remaining;
      /* At the start of a full-batch epoch, the current gradient is cached. */
      if (indexed) {
        status = batch(candidate, indices + start, count, training_context,
                       NULL, gradient);
        if (status != CML_SUCCESS)
          goto done;
      }
      status = update(candidate, gradient, first, second, rate,
                      result.optimization.iterations + 1, options, &change);
      if (status != CML_SUCCESS)
        goto done;
      ++result.optimization.iterations;
      start += count;
    }
    double previous_loss = result.optimization.final_loss;
    status = evaluate(batch, candidate, training_count, training_context,
                      &result.optimization.final_loss, gradient,
                      &result.optimization.gradient_norm);
    if (status != CML_SUCCESS)
      goto done;
    result.epochs = epoch + 1;

    double validation_loss = NAN;
    bool improved = false;
    if (best != NULL) {
      status = evaluate(batch, candidate, validation_count, validation_context,
                        &validation_loss, NULL, NULL);
      if (status != CML_SUCCESS)
        goto done;
      improved = validation_loss < result.validation_loss;
      if (improved) {
        memcpy(best->data, candidate->data, bytes);
        result.validation_loss = validation_loss;
        result.best_epoch = result.epochs;
      }
    }
    double monitored =
        best != NULL ? validation_loss : result.optimization.final_loss;
    if (patience_loss - monitored > options->min_improvement) {
      patience_loss = monitored;
      stale_epochs = 0;
    } else {
      ++stale_epochs;
    }

    bool converged =
        (config->gradient_tolerance > 0.0 &&
         result.optimization.gradient_norm <= config->gradient_tolerance) ||
        (config->loss_tolerance > 0.0 &&
         result.optimization.final_loss <= previous_loss &&
         previous_loss - result.optimization.final_loss <=
             config->loss_tolerance) ||
        (options->parameter_tolerance > 0.0 &&
         change <= options->parameter_tolerance) ||
        (options->patience > 0 && stale_epochs >= options->patience);
    if (converged && result.epochs >= options->min_epochs)
      result.optimization.status = CML_OPTIMIZATION_CONVERGED;

    size_t frequency = options->record_every == 0 ? 1 : options->record_every;
    bool last = result.epochs == options->epochs ||
                result.optimization.iterations == config->max_iterations ||
                result.optimization.status == CML_OPTIMIZATION_CONVERGED;
    bool record_epoch = last || result.epochs % frequency == 0;
    bool notify_observer =
        options->observer != NULL && (improved || record_epoch);
    bool save_checkpoint =
        options->checkpoint != NULL && monitored < checkpoint_loss;
    if ((options->history != NULL && record_epoch) || notify_observer ||
        save_checkpoint) {
      CMLTrainingRecord record = {
          .epoch = result.epochs,
          .updates = result.optimization.iterations,
          .training_loss = result.optimization.final_loss,
          .validation_loss = validation_loss,
          .gradient_norm = result.optimization.gradient_norm,
          .learning_rate = rate,
          .best = improved || save_checkpoint};
      status = cml_vector_norm_l2(candidate, &record.parameter_norm);
      if (status != CML_SUCCESS)
        goto done;
      if (options->history != NULL && record_epoch) {
        status = append_record(options->history, &record);
        if (status != CML_SUCCESS)
          goto done;
      }
      if (notify_observer &&
          !options->observer(&record, candidate, options->observer_context)) {
        result.optimization.status = CML_OPTIMIZATION_CANCELLED;
        status = CML_ERROR_CANCELLED;
        goto done;
      }
      if (save_checkpoint) {
        status = options->checkpoint(&record, candidate,
                                     options->checkpoint_context);
        if (status != CML_SUCCESS)
          goto done;
        checkpoint_loss = monitored;
      }
    }
  }
  if (best != NULL && result.best_epoch != result.epochs) {
    memcpy(candidate->data, best->data, bytes);
    status = evaluate(batch, candidate, training_count, training_context,
                      &result.optimization.final_loss, gradient,
                      &result.optimization.gradient_norm);
    if (status != CML_SUCCESS)
      goto done;
  }
  memcpy(parameters->data, candidate->data, bytes);

done:
  if (status == CML_ERROR_NUMERICAL)
    result.optimization.status = CML_OPTIMIZATION_DIVERGED;
  else if (status == CML_ERROR_CANCELLED)
    result.optimization.status = CML_OPTIMIZATION_CANCELLED;
  else if (status != CML_SUCCESS)
    result.optimization.status = CML_OPTIMIZATION_ERROR;
  result.optimization.error = status;
  if (out != NULL)
    *out = result;
  cml_free(indices);
  cml_vector_destroy(best);
  cml_vector_destroy(second);
  cml_vector_destroy(first);
  cml_vector_destroy(gradient);
  cml_vector_destroy(candidate);
  return status;
}

typedef struct {
  CMLObjectiveFunction objective;
  CMLGradientFunction gradient;
  void *context;
} ObjectiveContext;

static CMLStatus objective_batch(const CMLVector *p, const size_t *indices,
                                 size_t count, void *context, double *loss,
                                 CMLVector *gradient) {
  (void)indices;
  (void)count;
  ObjectiveContext *objective = context;
  CMLStatus status = loss == NULL
                         ? CML_SUCCESS
                         : objective->objective(p, objective->context, loss);
  if (status != CML_SUCCESS || gradient == NULL)
    return status;
  return objective->gradient(p, objective->context, gradient);
}

CMLStatus cml_optimizer_minimize(const CMLOptimizer *optimizer,
                                 CMLVector *parameters,
                                 CMLObjectiveFunction objective,
                                 CMLGradientFunction gradient, void *context,
                                 CMLOptimizationResult *result) {
  if (optimizer == NULL || objective == NULL || gradient == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLTrainingOptions options = cml_training_options_default();
  options.optimizer = optimizer->config;
  options.epochs = optimizer->config.max_iterations;
  ObjectiveContext wrapper = {objective, gradient, context};
  CMLTrainingResult training;
  CMLStatus status = cml_train(parameters, objective_batch, 1, &wrapper, 0,
                               NULL, &options, &training);
  if (result != NULL)
    *result = training.optimization;
  return status;
}

CMLStatus cml_gradient_check(const CMLVector *parameters,
                             CMLObjectiveFunction objective,
                             CMLGradientFunction gradient, void *context,
                             double step, double abs_tolerance,
                             double rel_tolerance,
                             CMLGradientCheckResult *out) {
  if (parameters == NULL || objective == NULL || gradient == NULL ||
      out == NULL || !isfinite(step) || step <= 0.0 ||
      !isfinite(abs_tolerance) || abs_tolerance < 0.0 ||
      !isfinite(rel_tolerance) || rel_tolerance < 0.0)
    return CML_ERROR_INVALID_ARGUMENT;
  if (parameters->size == 0)
    return CML_ERROR_INVALID_DIMENSION;
  CMLStatus status = cml_vector_storage(parameters, parameters->size);
  if (status != CML_SUCCESS)
    return status;
  if (!cml_values_finite(parameters->data, parameters->size))
    return CML_ERROR_NUMERICAL;
  CMLVector *copy = cml_vector_clone(parameters);
  CMLVector *analytic = cml_vector_create(parameters->size);
  if (copy == NULL || analytic == NULL) {
    status = CML_ERROR_OUT_OF_MEMORY;
    goto done;
  }
  status = gradient(parameters, context, analytic);
  if (status != CML_SUCCESS)
    goto done;
  CMLGradientCheckResult result = {.passed = true};
  for (size_t i = 0; i < copy->size; ++i) {
    double original = copy->data[i];
    double increment = step * fmax(1.0, fabs(original));
    double upper = original + increment, lower = original - increment;
    double denominator = upper - lower;
    if (!isfinite(denominator) || upper == original || lower == original ||
        !isfinite(analytic->data[i])) {
      status = CML_ERROR_NUMERICAL;
      goto done;
    }
    double high_loss, low_loss;
    copy->data[i] = upper;
    status = objective(copy, context, &high_loss);
    if (status != CML_SUCCESS)
      goto done;
    copy->data[i] = lower;
    status = objective(copy, context, &low_loss);
    if (status != CML_SUCCESS)
      goto done;
    copy->data[i] = original;
    double numeric = (high_loss - low_loss) / denominator;
    if (!isfinite(numeric)) {
      status = CML_ERROR_NUMERICAL;
      goto done;
    }
    double magnitude = fmax(fabs(numeric), fabs(analytic->data[i]));
    double absolute = fabs(analytic->data[i] - numeric);
    double relative =
        magnitude == 0.0
            ? 0.0
            : fabs(analytic->data[i] / magnitude - numeric / magnitude);
    if (absolute > result.max_absolute_error) {
      result.max_absolute_error = absolute;
      result.worst_index = i;
    }
    result.max_relative_error = fmax(result.max_relative_error, relative);
    if (absolute > abs_tolerance && relative > rel_tolerance)
      result.passed = false;
  }
  *out = result;
done:
  cml_vector_destroy(analytic);
  cml_vector_destroy(copy);
  return status;
}
