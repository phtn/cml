#include "cml/metrics.h"
#include "cml/classification.h"
#include "cml/loss.h"
#include "cml/memory.h"
#include "internal.h"

#include <stdlib.h>

static double ratio(size_t numerator, size_t denominator) {
  return denominator == 0 ? 0.0 : (double)numerator / (double)denominator;
}

static CMLBinaryMetrics binary_rates(CMLConfusionMatrix m) {
  size_t tp = m.true_positive, tn = m.true_negative;
  size_t fp = m.false_positive, fn = m.false_negative;
  double recall = ratio(tp, tp + fn), specificity = ratio(tn, tn + fp);
  double f1_denominator = 2.0 * (double)tp + (double)fp + (double)fn;
  return (CMLBinaryMetrics){
      .confusion = m,
      .accuracy = ratio(tp + tn, tp + tn + fp + fn),
      .precision = ratio(tp, tp + fp),
      .recall = recall,
      .f1 = f1_denominator == 0.0 ? 0.0 : 2.0 * (double)tp / f1_denominator,
      .specificity = specificity,
      .balanced_accuracy = 0.5 * (recall + specificity)};
}

CMLStatus cml_binary_metrics(const CMLVector *actual,
                             const CMLVector *predicted,
                             CMLBinaryMetrics *out) {
  if (actual == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLStatus status = cml_vector_storage(predicted, actual->size);
  if (status != CML_SUCCESS)
    return status;
  status = cml_classification_validate_binary_labels(actual, NULL);
  if (status == CML_SUCCESS)
    status = cml_classification_validate_binary_labels(predicted, NULL);
  if (status != CML_SUCCESS)
    return status;
  CMLConfusionMatrix counts = {0};
  for (size_t i = 0; i < actual->size; ++i) {
    if (actual->data[i] == 1.0) {
      if (predicted->data[i] == 1.0)
        ++counts.true_positive;
      else
        ++counts.false_negative;
    } else if (predicted->data[i] == 1.0) {
      ++counts.false_positive;
    } else {
      ++counts.true_negative;
    }
  }
  *out = binary_rates(counts);
  return CML_SUCCESS;
}

typedef struct {
  double score;
  bool positive;
} RankedScore;

static int compare_scores(const void *first, const void *second) {
  double a = ((const RankedScore *)first)->score;
  double b = ((const RankedScore *)second)->score;
  return (a < b) - (a > b);
}

void cml_roc_curve_destroy(CMLRocCurve *curve) {
  if (curve != NULL) {
    cml_free(curve->points);
    cml_free(curve);
  }
}

CMLStatus cml_roc_curve(const CMLVector *actual, const CMLVector *scores,
                        CMLRocCurve **out) {
  if (out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLBinaryClassCounts classes;
  CMLStatus status =
      cml_classification_validate_binary_labels(actual, &classes);
  if (status != CML_SUCCESS)
    return status;
  status = cml_vector_storage(scores, actual->size);
  if (status != CML_SUCCESS)
    return status;
  if (classes.negative_count == 0 || classes.positive_count == 0)
    return CML_ERROR_INVALID_DIMENSION;
  if (scores->size >= SIZE_MAX / sizeof(CMLRocPoint) ||
      scores->size > SIZE_MAX / sizeof(RankedScore))
    return CML_ERROR_OVERFLOW;
  for (size_t i = 0; i < scores->size; ++i) {
    if (!isfinite(scores->data[i]))
      return CML_ERROR_NUMERICAL;
    if (scores->data[i] < 0.0 || scores->data[i] > 1.0)
      return CML_ERROR_INVALID_ARGUMENT;
  }
  RankedScore *ranked = cml_calloc(scores->size, sizeof(*ranked));
  CMLRocCurve *curve = cml_calloc(1, sizeof(*curve));
  if (ranked == NULL || curve == NULL) {
    cml_free(ranked);
    cml_free(curve);
    return CML_ERROR_OUT_OF_MEMORY;
  }
  curve->points = cml_calloc(scores->size + 1, sizeof(*curve->points));
  if (curve->points == NULL) {
    cml_free(ranked);
    cml_roc_curve_destroy(curve);
    return CML_ERROR_OUT_OF_MEMORY;
  }
  for (size_t i = 0; i < scores->size; ++i)
    ranked[i] = (RankedScore){scores->data[i], actual->data[i] == 1.0};
  qsort(ranked, scores->size, sizeof(*ranked), compare_scores);
  curve->points[curve->size++] = (CMLRocPoint){INFINITY, 0.0, 0.0};
  size_t tp = 0, fp = 0;
  for (size_t i = 0; i < scores->size;) {
    double threshold = ranked[i].score;
    do {
      if (ranked[i].positive)
        ++tp;
      else
        ++fp;
      ++i;
    } while (i < scores->size && ranked[i].score == threshold);
    CMLRocPoint previous = curve->points[curve->size - 1];
    CMLRocPoint point = {threshold, ratio(fp, classes.negative_count),
                         ratio(tp, classes.positive_count)};
    curve->auc +=
        (point.false_positive_rate - previous.false_positive_rate) *
        (0.5 * (point.true_positive_rate + previous.true_positive_rate));
    curve->points[curve->size++] = point;
    double f1 = 2.0 * (double)tp /
                ((double)tp + (double)fp + (double)classes.positive_count);
    if (f1 > curve->best_f1) {
      curve->best_f1 = f1;
      curve->best_threshold = threshold;
    }
  }
  cml_free(ranked);
  *out = curve;
  return CML_SUCCESS;
}

CMLStatus cml_r_squared(const CMLVector *actual, const CMLVector *predicted,
                        double *out) {
  if (actual == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (actual->size == 0)
    return CML_ERROR_INVALID_DIMENSION;
  CMLStatus status = cml_vector_storage(actual, actual->size);
  if (status == CML_SUCCESS)
    status = cml_vector_storage(predicted, actual->size);
  if (status != CML_SUCCESS)
    return status;

  double scale = 0.0;
  bool constant = true, exact = true;
  for (size_t i = 0; i < actual->size; ++i) {
    double target = actual->data[i], prediction = predicted->data[i];
    if (!isfinite(target) || !isfinite(prediction))
      return CML_ERROR_NUMERICAL;
    constant = constant && target == actual->data[0];
    exact = exact && target == prediction;
    scale = fmax(scale, fmax(fabs(target), fabs(prediction)));
  }
  if (constant || exact) {
    *out = exact ? 1.0 : 0.0;
    return CML_SUCCESS;
  }

  int exponent;
  frexp(scale, &exponent);
  double origin = scalbn(actual->data[0], -exponent), mean = 0.0;
  for (size_t i = 0; i < actual->size; ++i)
    mean += scalbn(actual->data[i], -exponent) - origin;
  mean /= (double)actual->size;
  double residual = 0.0, total = 0.0;
  for (size_t i = 0; i < actual->size; ++i) {
    double target = scalbn(actual->data[i], -exponent);
    double error = scalbn(predicted->data[i], -exponent) - target;
    double centered = (target - origin) - mean;
    residual = fma(error, error, residual);
    total = fma(centered, centered, total);
  }
  if (total == 0.0)
    return CML_ERROR_NUMERICAL;
  double score = 1.0 - residual / total;
  if (!isfinite(score))
    return CML_ERROR_NUMERICAL;
  *out = score;
  return CML_SUCCESS;
}

CMLStatus cml_regression_metrics(const CMLVector *actual,
                                 const CMLVector *predicted,
                                 CMLRegressionMetrics *out) {
  if (out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  CMLRegressionMetrics values = {0};
  CMLStatus status = cml_loss_mae(actual, predicted, &values.mae);
  if (status == CML_SUCCESS)
    status = cml_loss_mse(actual, predicted, &values.mse);
  if (status == CML_SUCCESS)
    status = cml_loss_rmse(actual, predicted, &values.rmse);
  if (status == CML_SUCCESS)
    status = cml_r_squared(actual, predicted, &values.r_squared);
  if (status == CML_SUCCESS) {
    values.samples = actual->size;
    *out = values;
  }
  return status;
}

void cml_multiclass_metrics_destroy(CMLMulticlassMetrics *report) {
  if (report != NULL) {
    cml_free(report->confusion);
    cml_free(report->per_class);
    cml_free(report);
  }
}

static bool valid_class(double value, size_t classes) {
  return value >= 0.0 && value < (double)classes && value == floor(value);
}

CMLStatus cml_multiclass_metrics(const CMLVector *actual,
                                 const CMLVector *predicted, size_t classes,
                                 CMLMulticlassMetrics **out) {
  if (actual == NULL || out == NULL)
    return CML_ERROR_INVALID_ARGUMENT;
  if (actual->size == 0 || classes < 2)
    return CML_ERROR_INVALID_DIMENSION;
  if (classes > SIZE_MAX / sizeof(size_t) / classes ||
      classes > SIZE_MAX / sizeof(CMLBinaryMetrics))
    return CML_ERROR_OVERFLOW;
  CMLStatus status = cml_vector_storage(actual, actual->size);
  if (status == CML_SUCCESS)
    status = cml_vector_storage(predicted, actual->size);
  if (status != CML_SUCCESS)
    return status;
  for (size_t i = 0; i < actual->size; ++i) {
    if (!isfinite(actual->data[i]) || !isfinite(predicted->data[i]))
      return CML_ERROR_NUMERICAL;
    if (!valid_class(actual->data[i], classes) ||
        !valid_class(predicted->data[i], classes))
      return CML_ERROR_INVALID_ARGUMENT;
  }

  CMLMulticlassMetrics *report = cml_calloc(1, sizeof(*report));
  if (report == NULL)
    return CML_ERROR_OUT_OF_MEMORY;
  report->classes = classes;
  report->confusion = cml_calloc(classes * classes, sizeof(*report->confusion));
  report->per_class = cml_calloc(classes, sizeof(*report->per_class));
  if (report->confusion == NULL || report->per_class == NULL) {
    cml_multiclass_metrics_destroy(report);
    return CML_ERROR_OUT_OF_MEMORY;
  }
  for (size_t i = 0; i < actual->size; ++i)
    ++report->confusion[(size_t)actual->data[i] * classes +
                        (size_t)predicted->data[i]];

  size_t correct = 0;
  for (size_t i = 0; i < classes; ++i) {
    size_t tp = report->confusion[i * classes + i], support = 0, guessed = 0;
    for (size_t j = 0; j < classes; ++j) {
      support += report->confusion[i * classes + j];
      guessed += report->confusion[j * classes + i];
    }
    correct += tp;
    size_t fn = support - tp, fp = guessed - tp;
    CMLConfusionMatrix counts = {actual->size - tp - fp - fn, fp, fn, tp};
    CMLBinaryMetrics rates = binary_rates(counts);
    report->per_class[i] = rates;
    double weight = ratio(support, actual->size);
    report->weighted.precision += rates.precision * weight;
    report->weighted.recall += rates.recall * weight;
    report->weighted.f1 += rates.f1 * weight;
    report->macro.precision += rates.precision / (double)classes;
    report->macro.recall += rates.recall / (double)classes;
    report->macro.f1 += rates.f1 / (double)classes;
  }
  report->accuracy = ratio(correct, actual->size);
  report->micro = (CMLClassificationRates){report->accuracy, report->accuracy,
                                           report->accuracy};
  *out = report;
  return CML_SUCCESS;
}
