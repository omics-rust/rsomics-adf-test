//! MacKinnon (2010) critical values for the ADF test (N=1).
//!
//! Direct port of `statsmodels.tsa.adfvalues.mackinnoncrit(N=1, regression, nobs)`.
//!
//! Each regression case provides a 3×4 table `tau[level][coef]` where level
//! indexes 1%/5%/10% and the polynomial is evaluated in `1/nobs`:
//!   `crit[level] = tau[level][0] + tau[level][1]/nobs + tau[level][2]/nobs² + tau[level][3]/nobs³`
//!
//! Coefficients are from `statsmodels.tsa.adfvalues.tau_{nc,c,ct,ctt}_2010[0]`
//! (the N=1 slice).
//!
//! `polyval(val.T, 1./nobs)` in statsmodels reverses each row before evaluating,
//! so `tau[level]` is stored with `coef[0]` = degree-0 (asymptotic value),
//! matching `polyval(reversed_row, 1/nobs)` = `coef[0] + coef[1]/n + ...`.

use crate::Regression;

// Coefficients: [asymptote, c1, c2, c3] for each level (1%, 5%, 10%), N=1.
// Source: statsmodels tau_{nc,c,ct,ctt}_2010[0] (the N=1 block, rows 0..3, 4 cols each).
// statsmodels polyval call: `polyval(val.T, 1./nobs)` where val = tau[N-1,:,::-1]
// val[level] = tau[N-1, level, ::-1]  (reversed).
// polyval([c3,c2,c1,c0], z) = c3*z^3 + c2*z^2 + c1*z + c0  (highest-degree first).
// But stored in tau_*_2010 as [c0, c1, c2, c3] (degree-0 first), so after reverse:
//   polyval([c3,c2,c1,c0], z) = c0 + c1*z + c2*z^2 + c3*z^3.
// We store [c0,c1,c2,c3] below and evaluate via Horner from left.

// tau_nc_2010[0] rows: N=1, 1%/5%/10%
const NC: [[f64; 4]; 3] = [
    [-2.56574, -2.2358, -3.627, 0.0],
    [-1.94100, -0.2686, -3.365, 31.223],
    [-1.61682, 0.2656, -2.714, 25.364],
];

// tau_c_2010[0] rows: N=1
const C: [[f64; 4]; 3] = [
    [-3.43035, -6.5393, -16.786, -79.433],
    [-2.86154, -2.8903, -4.234, -40.040],
    [-2.56677, -1.5384, -2.809, 0.0],
];

// tau_ct_2010[0] rows: N=1
const CT: [[f64; 4]; 3] = [
    [-3.95877, -9.0531, -28.428, -134.155],
    [-3.41049, -4.3904, -9.036, -45.374],
    [-3.12705, -2.5856, -3.925, -22.380],
];

// tau_ctt_2010[0] rows: N=1
const CTT: [[f64; 4]; 3] = [
    [-4.37113, -11.5882, -35.819, -334.047],
    [-3.83239, -5.9057, -12.490, -118.284],
    [-3.55326, -3.6596, -5.293, -63.559],
];

/// Evaluate `c[0] + c[1]*z + c[2]*z² + c[3]*z³` (degree-0 first, Horner).
fn polyval4(c: &[f64; 4], z: f64) -> f64 {
    c[0] + z * (c[1] + z * (c[2] + z * c[3]))
}

/// MacKinnon (2010) critical values at 1%, 5%, 10% for ADF (N=1).
///
/// Returns `[crit_1pct, crit_5pct, crit_10pct]`.
pub fn mackinnoncrit(regression: Regression, nobs: usize) -> [f64; 3] {
    let table = match regression {
        Regression::N => &NC,
        Regression::C => &C,
        Regression::Ct => &CT,
        Regression::Ctt => &CTT,
    };

    let z = 1.0 / nobs as f64;
    [
        polyval4(&table[0], z),
        polyval4(&table[1], z),
        polyval4(&table[2], z),
    ]
}
