#include "cml/metrics.h"
#include "cml/classification.h"
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
