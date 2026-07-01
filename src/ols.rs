//! OLS helpers used by the ADF implementation.

use nalgebra::{DMatrix, DVector};

/// OLS t-statistic on column 0 (`y_{t-1}`).
///
/// Used at the final refit step — column layout is [y_{t-1}, lag_diffs..., trend_cols...].
pub fn ols_tstat_col0(endog: &[f64], exog: &[f64], nobs: usize, ncols: usize) -> f64 {
    let y = DVector::from_column_slice(endog);
    let x = DMatrix::from_row_slice(nobs, ncols, &exog[..nobs * ncols]);
    let xtx = x.tr_mul(&x);
    let xty = x.tr_mul(&y);
    let beta = xtx.clone().lu().solve(&xty).expect("OLS: X'X singular");
    let resid = &y - &x * &beta;
    let ssr = resid.dot(&resid);
    let df = (nobs - ncols) as f64;
    let sigma2 = ssr / df;
    let xtx_inv = xtx.try_inverse().expect("OLS: X'X not invertible");
    let se0 = (sigma2 * xtx_inv[(0, 0)]).sqrt();
    beta[0] / se0
}

/// OLS residual sum of squares, for IC computation.
pub fn ols_ssr(endog: &[f64], exog: &[f64], nobs: usize, ncols: usize) -> f64 {
    let y = DVector::from_column_slice(endog);
    let x = DMatrix::from_row_slice(nobs, ncols, &exog[..nobs * ncols]);
    let xtx = x.tr_mul(&x);
    let xty = x.tr_mul(&y);
    let beta = xtx
        .lu()
        .solve(&xty)
        .unwrap_or_else(|| panic!("ols_ssr: X'X singular at nobs={nobs} ncols={ncols}"));
    let resid = &y - &x * &beta;
    resid.dot(&resid)
}
