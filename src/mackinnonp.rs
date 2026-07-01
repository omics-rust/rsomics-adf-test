//! MacKinnon (1994) approximate p-value for the ADF test statistic.
//!
//! Direct port of `statsmodels.tsa.adfvalues.mackinnonp(teststat, regression, N=1)`.
//!
//! The p-value is obtained by evaluating a polynomial in `teststat`, then
//! applying `Φ` (standard-normal CDF). Two polynomial families are used:
//! * `tau_smallp`: for `teststat ≤ tau_star` (left tail).
//! * `tau_largep`: for `teststat > tau_star` (right tail).
//!
//! Boundary behaviour:
//! * `teststat > tau_max` → 1.0
//! * `teststat < tau_min` → 0.0

use crate::Regression;

// ── cutoff tables (N=1 entries, index 0) ──────────────────────────────────────

const TAU_STAR_NC: f64 = -1.04;
const TAU_MIN_NC: f64 = -19.04;
const TAU_MAX_NC: f64 = f64::INFINITY;

const TAU_STAR_C: f64 = -1.61;
const TAU_MIN_C: f64 = -18.83;
const TAU_MAX_C: f64 = 2.74;

const TAU_STAR_CT: f64 = -2.89;
const TAU_MIN_CT: f64 = -16.18;
const TAU_MAX_CT: f64 = 0.7;

const TAU_STAR_CTT: f64 = -3.21;
const TAU_MIN_CTT: f64 = -17.17;
const TAU_MAX_CTT: f64 = 0.54;

// ── small-p polynomial coefficients (degree-2: [a0, a1, a2·1e-2]) ────────────
// Source: statsmodels adfvalues.py tau_{nc,c,ct,ctt}_smallp[0] * small_scaling
// small_scaling = [1, 1, 1e-2]

const NC_SMALL: [f64; 3] = [0.6344, 1.2378, 3.2496e-2];
const C_SMALL: [f64; 3] = [2.1659, 1.4412, 3.8269e-2];
const CT_SMALL: [f64; 3] = [3.2512, 1.6047, 4.9588e-2];
const CTT_SMALL: [f64; 3] = [4.0003, 1.658, 4.8288e-2];

// ── large-p polynomial coefficients (degree-3: [b0, b1·1e-1, b2·1e-1, b3·1e-2]) ─
// Source: statsmodels tau_{nc,c,ct,ctt}_largep[0] * large_scaling
// large_scaling = [1, 1e-1, 1e-1, 1e-2]

const NC_LARGE: [f64; 4] = [0.4797, 9.3557e-1, -0.6999e-1, 3.3066e-2];
const C_LARGE: [f64; 4] = [1.7339, 9.3202e-1, -1.2745e-1, -1.0368e-2];
const CT_LARGE: [f64; 4] = [2.5261, 6.1654e-1, -3.7956e-1, -6.0285e-2];
const CTT_LARGE: [f64; 4] = [3.0778, 4.9529e-1, -4.1477e-1, -5.9359e-2];

/// Evaluate polynomial `p[0] + p[1]*x + p[2]*x² + ...` (numpy `polyval` with reversed coeffs).
///
/// `numpy.polyval(coef, x)` treats `coef[0]` as the highest-degree coefficient.
/// statsmodels calls `polyval(tau_coef[::-1], teststat)` so `coef[::-1][0]` is
/// the degree-0 term — i.e. our arrays are stored degree-0 first, matching
/// Horner's method with the slice as-is.
fn polyval(coef: &[f64], x: f64) -> f64 {
    // Horner, degree-0 first.
    coef.iter().rev().fold(0.0_f64, |acc, &c| acc * x + c)
}

/// Standard normal CDF via Horner rational approximation (Hart, CACM 1968).
/// Matches `scipy.stats.norm.cdf` to ≈ 15 significant digits.
fn norm_cdf(x: f64) -> f64 {
    use std::f64::consts::SQRT_2;
    0.5 * libm_erfc(-x / SQRT_2)
}

/// `erfc(x)` via the complementary-error-function identity, matching libm.
fn libm_erfc(x: f64) -> f64 {
    // Delegate to the intrinsic.
    unsafe extern "C" {
        fn erfc(x: f64) -> f64;
    }
    unsafe { erfc(x) }
}

/// MacKinnon (1994) approximate p-value for the ADF statistic (N=1).
pub fn mackinnonp(teststat: f64, regression: Regression) -> f64 {
    let (tau_max, tau_min, tau_star, coef_small, coef_large) = match regression {
        Regression::N => (
            TAU_MAX_NC,
            TAU_MIN_NC,
            TAU_STAR_NC,
            NC_SMALL.as_ref(),
            NC_LARGE.as_ref(),
        ),
        Regression::C => (
            TAU_MAX_C,
            TAU_MIN_C,
            TAU_STAR_C,
            C_SMALL.as_ref(),
            C_LARGE.as_ref(),
        ),
        Regression::Ct => (
            TAU_MAX_CT,
            TAU_MIN_CT,
            TAU_STAR_CT,
            CT_SMALL.as_ref(),
            CT_LARGE.as_ref(),
        ),
        Regression::Ctt => (
            TAU_MAX_CTT,
            TAU_MIN_CTT,
            TAU_STAR_CTT,
            CTT_SMALL.as_ref(),
            CTT_LARGE.as_ref(),
        ),
    };

    if teststat > tau_max {
        return 1.0;
    }
    if teststat < tau_min {
        return 0.0;
    }

    let coef = if teststat <= tau_star {
        coef_small
    } else {
        coef_large
    };

    norm_cdf(polyval(coef, teststat))
}
