//! OLS helpers used by the ADF implementation.

use nalgebra::{DMatrix, DVector};

/// Solve `X'X β = X'y` and return `(β, (X'X)⁻¹)`.
///
/// A well-conditioned design uses the direct LU solve plus explicit inverse — the
/// path that is value-exact against statsmodels' `pinv` fit. A rank-deficient `X'X`
/// (collinear deterministic + lag columns: a perfect linear trend, an alternating
/// series, a repeating block) has no inverse; there we fall back to the Moore-Penrose
/// pseudo-inverse, which is what statsmodels' `pinv` computes, so the fit returns a
/// defined value instead of failing.
pub(crate) fn xtx_solve(xtx: DMatrix<f64>, xty: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
    if let (Some(beta), Some(inv)) = (xtx.clone().lu().solve(xty), xtx.clone().try_inverse()) {
        if beta.iter().all(|v| v.is_finite()) && inv.iter().all(|v| v.is_finite()) {
            return (beta, inv);
        }
    }
    let pinv = pseudo_inverse(xtx);
    let beta = &pinv * xty;
    (beta, pinv)
}

/// Moore-Penrose pseudo-inverse via SVD, with singular values below `max_sv · 1e-12`
/// treated as zero (numpy `pinv`'s relative-cutoff convention).
fn pseudo_inverse(m: DMatrix<f64>) -> DMatrix<f64> {
    let svd = m.svd(true, true);
    let max_sv = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    svd.pseudo_inverse(max_sv * 1e-12)
        .expect("pseudo_inverse of a finite matrix")
}

/// OLS t-statistic on column 0 (`y_{t-1}`).
///
/// Used at the final refit step — column layout is [y_{t-1}, lag_diffs..., trend_cols...].
pub fn ols_tstat_col0(endog: &[f64], exog: &[f64], nobs: usize, ncols: usize) -> f64 {
    let y = DVector::from_column_slice(endog);
    let x = DMatrix::from_row_slice(nobs, ncols, &exog[..nobs * ncols]);
    let xtx = x.tr_mul(&x);
    let xty = x.tr_mul(&y);
    let (beta, xtx_inv) = xtx_solve(xtx, &xty);
    let resid = &y - &x * &beta;
    let ssr = resid.dot(&resid);
    let df = (nobs - ncols) as f64;
    let sigma2 = ssr / df;
    let se0 = (sigma2 * xtx_inv[(0, 0)]).sqrt();
    beta[0] / se0
}

/// OLS residual sum of squares, for IC computation.
pub fn ols_ssr(endog: &[f64], exog: &[f64], nobs: usize, ncols: usize) -> f64 {
    let y = DVector::from_column_slice(endog);
    let x = DMatrix::from_row_slice(nobs, ncols, &exog[..nobs * ncols]);
    let xtx = x.tr_mul(&x);
    let xty = x.tr_mul(&y);
    let (beta, _) = xtx_solve(xtx, &xty);
    let resid = &y - &x * &beta;
    resid.dot(&resid)
}
