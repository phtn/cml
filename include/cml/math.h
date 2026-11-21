#ifndef CML_MATH_H
#define CML_MATH_H

/* ==============================================================================
 * cml/math.h — CML Mathematical Utilities & Constants
 *
 * Provides:
 *   - Mathematical constants (PI, E, EPSILON).
 *   - Floating-point comparison with absolute/relative tolerance.
 *   - Utility functions for NaN/Inf/finite detection and clamping.
 * ------------------------------------------------------------------------------
 */

#include <stdbool.h>
#include <stddef.h>

#define CML_PI 3.14159265358979323846
#define CML_E 2.71828182845904523536
#define CML_EPSILON 1e-9

/* ==============================================================================
 * cml_is_close
 *
 * Compare two doubles for approximate equality using both absolute and
 * relative tolerance:
 *   |a - b| <= max(atol, rtol * max(|a|, |b|))
 * ------------------------------------------------------------------------------
 */
bool cml_is_close(double a, double b, double tol);

/* ==============================================================================
 * cml_clamp
 *
 * Clamp value `x` to the interval [min_val, max_val].
 * ------------------------------------------------------------------------------
 */
double cml_clamp(double x, double min_val, double max_val);

/* ==============================================================================
 * cml_max / cml_min / cml_sign
 * ------------------------------------------------------------------------------
 */
double cml_max(double a, double b);
double cml_min(double a, double b);
double cml_sign(double x);

#endif /* CML_MATH_H */
