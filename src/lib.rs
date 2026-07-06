//! Augmented Dickey-Fuller unit-root test — value-exact port of
//! `statsmodels.tsa.stattools.adfuller` (statsmodels 0.14.x).
//!
//! # Algorithm overview
//! 1. Compute `Δy_t = y_t − y_{t−1}` and build a "lag matrix" from it.
//! 2. Build the autolag search matrix `fullRHS` using `add_trend(..., prepend=True)`:
//!    columns are `[trend_cols, y_{t-1}, Δy_{t-1}, ..., Δy_{t-maxlag}]`.
//! 3. Select lag order by minimising AIC/BIC over `startlag..=(startlag+maxlag)` columns,
//!    or by t-stat rule, or use the fixed `maxlag`.
//! 4. Refit with `add_trend` **without** prepend, giving column order
//!    `[y_{t-1}, Δy_{t-1}, ..., Δy_{t-usedlag}, trend_cols]`.
//! 5. `adf_stat = tvalues[0]` (t-statistic on the first column = lagged level).
//! 6. p-value via MacKinnon (1994); critical values via MacKinnon (2010).

mod mackinnoncrit;
mod mackinnonp;
mod ols;

pub use mackinnoncrit::mackinnoncrit;
pub use mackinnonp::mackinnonp;
pub(crate) use ols::{ols_ssr, ols_tstat_col0, xtx_solve};

/// Reasons `adfuller` refuses an input, mirroring the `ValueError`/`MissingDataError`
/// cases statsmodels raises for the same inputs.
#[derive(Debug, Clone, PartialEq)]
pub enum AdfError {
    /// The series contains a NaN or infinity.
    NonFinite,
    /// Every element is identical (statsmodels: "Invalid input, x is constant").
    Constant,
    /// Too few observations for the regression component (`nobs/2 - ntrend - 1 < 0`).
    TooShort,
    /// A caller-supplied `maxlag` exceeds `nobs/2 - 1 - ntrend`.
    MaxlagTooLarge { maxlag: usize, hard_max: i64 },
}

impl std::fmt::Display for AdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdfError::NonFinite => write!(f, "series contains a non-finite value (NaN or inf)"),
            AdfError::Constant => write!(f, "series is constant"),
            AdfError::TooShort => write!(
                f,
                "sample size is too short to use the selected regression component"
            ),
            AdfError::MaxlagTooLarge { maxlag, hard_max } => {
                write!(
                    f,
                    "maxlag {maxlag} exceeds nobs/2 - 1 - ntrend = {hard_max}"
                )
            }
        }
    }
}

impl std::error::Error for AdfError {}

/// Regression specification: which deterministic terms to include.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Regression {
    /// Constant only (default).
    C,
    /// Constant + linear trend.
    Ct,
    /// Constant + linear + quadratic trend.
    Ctt,
    /// No deterministic terms.
    N,
}

impl Regression {
    /// Number of deterministic columns added by `add_trend`.
    pub fn ntrend(self) -> usize {
        match self {
            Regression::C => 1,
            Regression::Ct => 2,
            Regression::Ctt => 3,
            Regression::N => 0,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Regression::C => "c",
            Regression::Ct => "ct",
            Regression::Ctt => "ctt",
            Regression::N => "n",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "c" => Some(Regression::C),
            "ct" => Some(Regression::Ct),
            "ctt" => Some(Regression::Ctt),
            "n" => Some(Regression::N),
            _ => None,
        }
    }
}

/// Lag-selection method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AutoLag {
    Aic,
    Bic,
    /// Drop from `maxlag` down until last-lag |t-stat| ≥ 1.6449.
    TStat,
    /// Fixed: use exactly `maxlag` lags.
    None,
}

impl AutoLag {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "aic" => Some(AutoLag::Aic),
            "bic" => Some(AutoLag::Bic),
            "t-stat" | "tstat" => Some(AutoLag::TStat),
            "none" => Some(AutoLag::None),
            _ => Option::None,
        }
    }
}

/// Result of the Augmented Dickey-Fuller test.
#[derive(Debug)]
pub struct AdfResult {
    pub adf_stat: f64,
    pub pvalue: f64,
    pub usedlag: usize,
    pub nobs: usize,
    pub crit_1pct: f64,
    pub crit_5pct: f64,
    pub crit_10pct: f64,
    pub icbest: Option<f64>,
}

/// Augmented Dickey-Fuller unit-root test.
///
/// Value-exact port of `statsmodels.tsa.stattools.adfuller(x, maxlag, regression, autolag)`.
pub fn adfuller(
    x: &[f64],
    maxlag: Option<usize>,
    regression: Regression,
    autolag: AutoLag,
) -> Result<AdfResult, AdfError> {
    let orig_nobs = x.len();

    if x.iter().any(|v| !v.is_finite()) {
        return Err(AdfError::NonFinite);
    }
    if let Some(&first) = x.first() {
        if x.iter().all(|&v| v == first) {
            return Err(AdfError::Constant);
        }
    }

    let ntrend = regression.ntrend();

    // Signed so a too-short series is caught explicitly rather than wrapping the
    // unsigned `nobs/2 - ntrend - 1` into a huge maxlag.
    let hard_max = orig_nobs as i64 / 2 - ntrend as i64 - 1;
    let maxlag: usize = match maxlag {
        Some(m) => {
            if m as i64 > hard_max {
                return Err(AdfError::MaxlagTooLarge {
                    maxlag: m,
                    hard_max,
                });
            }
            m
        }
        None => {
            let schwert = (12.0 * (orig_nobs as f64 / 100.0).powf(0.25)).ceil() as i64;
            let ml = schwert.min(hard_max);
            if ml < 0 {
                return Err(AdfError::TooShort);
            }
            ml as usize
        }
    };

    // xdiff = diff(x), length = orig_nobs - 1.
    let xdiff: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();

    // nobs_max: rows after trimming `maxlag` lags from xdiff (statsmodels "both" trim).
    // lagmat(xdiff, maxlag, trim="both", original="in") → shape (xdiff.len()-maxlag, maxlag+1)
    let nobs_max = xdiff.len() - maxlag; // orig_nobs - 1 - maxlag

    // Build `fullRHS` for the autolag search — add_trend with prepend=True:
    // columns: [trend_cols | y_{t-1} | Δy_{t-1} | ... | Δy_{t-maxlag}]
    //
    // statsmodels layout:
    //   xdall = lagmat(xdiff[:,None], maxlag, trim="both", original="in")
    //   xdall[:,0] = x[-nobs_max-1:-1]   ← replace first col with level
    //   fullRHS = add_trend(xdall, regression, prepend=True)
    //
    // add_trend(prepend=True) → [trend | xdall] so:
    //   c:   [1 | y_{t-1} | Δy_{t-1} | ... | Δy_{t-maxlag}]
    //   ct:  [1 t | y_{t-1} | Δy_{t-1} | ...]   (trend = 1..nobs_max, 1-based)
    //   ctt: [1 t t² | y_{t-1} | ...]
    //   n:   fullRHS = xdall (no add_trend call)
    //
    // startlag = ntrend + 1  (index of first lag-diff column in fullRHS)
    let startlag = ntrend + 1;
    let ncols_full = startlag + maxlag;
    let mut full_rhs = vec![0.0_f64; nobs_max * ncols_full];

    // Trend columns (1-based, matching numpy.vander + fliplr).
    match regression {
        Regression::C => {
            for r in 0..nobs_max {
                full_rhs[r * ncols_full] = 1.0;
            }
        }
        Regression::Ct => {
            for r in 0..nobs_max {
                let t = (r + 1) as f64; // 1-based
                full_rhs[r * ncols_full] = 1.0;
                full_rhs[r * ncols_full + 1] = t;
            }
        }
        Regression::Ctt => {
            for r in 0..nobs_max {
                let t = (r + 1) as f64;
                full_rhs[r * ncols_full] = 1.0;
                full_rhs[r * ncols_full + 1] = t;
                full_rhs[r * ncols_full + 2] = t * t;
            }
        }
        Regression::N => {}
    }

    // Lagged level y_{t-1}: x[-nobs_max-1..-1].
    let level_start = orig_nobs - nobs_max - 1;
    for r in 0..nobs_max {
        full_rhs[r * ncols_full + ntrend] = x[level_start + r];
    }

    // Lagged differences Δy_{t-j}: column (ntrend+j) for j=1..maxlag.
    // lagmat(xdiff, maxlag, trim="both") row r, lag-col j (0-indexed in lagmat):
    //   = xdiff[maxlag - 1 - j + r]
    for r in 0..nobs_max {
        for j in 0..maxlag {
            let col = ntrend + 1 + j;
            full_rhs[r * ncols_full + col] = xdiff[(maxlag - 1 - j) + r];
        }
    }

    // endog for the autolag search: xdiff[-nobs_max..].
    let xdshort = &xdiff[xdiff.len() - nobs_max..];

    if autolag == AutoLag::None {
        // Fixed lag: refit at maxlag with the correct (non-prepend) column order.
        let (adf_stat, nobs) = refit(x, &xdiff, maxlag, regression, orig_nobs);
        let pvalue = mackinnonp(adf_stat, regression);
        let [crit_1pct, crit_5pct, crit_10pct] = mackinnoncrit(regression, nobs);
        return Ok(AdfResult {
            adf_stat,
            pvalue,
            usedlag: maxlag,
            nobs,
            crit_1pct,
            crit_5pct,
            crit_10pct,
            icbest: Option::None,
        });
    }

    // Autolag: iterate over column counts startlag..=startlag+maxlag.
    let mut best_lag_col = startlag;
    let mut icbest = f64::INFINITY;

    if autolag == AutoLag::TStat {
        let stop = 1.6448536269514722_f64;
        best_lag_col = startlag + maxlag;
        icbest = 0.0;
        for lag_col in (startlag..=startlag + maxlag).rev() {
            let last_t = ols_last_tstat(xdshort, &full_rhs, nobs_max, lag_col, ncols_full);
            icbest = last_t.abs();
            best_lag_col = lag_col;
            if last_t.abs() >= stop {
                break;
            }
        }
    } else {
        for lag_col in startlag..=startlag + maxlag {
            let ic = compute_ic(xdshort, &full_rhs, nobs_max, lag_col, ncols_full, autolag);
            if ic < icbest {
                icbest = ic;
                best_lag_col = lag_col;
            }
        }
    }

    let usedlag = best_lag_col - startlag;

    // Refit with the selected lag — using add_trend WITHOUT prepend (statsmodels default).
    let (adf_stat, nobs) = refit(x, &xdiff, usedlag, regression, orig_nobs);
    let pvalue = mackinnonp(adf_stat, regression);
    let [crit_1pct, crit_5pct, crit_10pct] = mackinnoncrit(regression, nobs);

    Ok(AdfResult {
        adf_stat,
        pvalue,
        usedlag,
        nobs,
        crit_1pct,
        crit_5pct,
        crit_10pct,
        icbest: Some(icbest),
    })
}

/// Final OLS fit, matching statsmodels:
///   `OLS(xdshort, add_trend(xdall[:, :usedlag+1], regression)).fit()`
///   where `add_trend` uses default `prepend=False`.
///
/// Column layout (prepend=False = append):
///   n:   [y_{t-1}, Δy_{t-1}, ..., Δy_{t-usedlag}]
///   c:   [y_{t-1}, Δy_{t-1}, ..., Δy_{t-usedlag}, 1]
///   ct:  [y_{t-1}, Δy_{t-1}, ..., Δy_{t-usedlag}, 1, t]
///   ctt: [y_{t-1}, Δy_{t-1}, ..., Δy_{t-usedlag}, 1, t, t²]
///
/// `tvalues[0]` (col 0 = y_{t-1}) is the ADF statistic.
fn refit(
    x: &[f64],
    xdiff: &[f64],
    usedlag: usize,
    regression: Regression,
    orig_nobs: usize,
) -> (f64, usize) {
    // nobs for this lag: lagmat(xdiff, usedlag, trim="both") → xdiff.len()-usedlag rows.
    // For usedlag=0, lagmat with 0 lags gives the original array (no trimming needed):
    //   lagmat(xdiff[:,None], 0, trim="both", original="in") → xdiff as column, nobs=len(xdiff)=199
    let nobs = if usedlag == 0 {
        xdiff.len()
    } else {
        xdiff.len() - usedlag
    };
    let ntrend = regression.ntrend();
    let ncols = 1 + usedlag + ntrend; // level + lag_diffs + trend

    let mut rhs = vec![0.0_f64; nobs * ncols];

    // Column 0: y_{t-1}  (lagged level)
    let level_start = orig_nobs - nobs - 1;
    for r in 0..nobs {
        rhs[r * ncols] = x[level_start + r];
    }

    // Columns 1..usedlag: Δy_{t-j}, j=1..usedlag (same lagmat ordering as before)
    for r in 0..nobs {
        for j in 0..usedlag {
            let col = 1 + j;
            rhs[r * ncols + col] = xdiff[(usedlag - 1 - j) + r];
        }
    }

    // Append trend columns (add_trend with prepend=False appends):
    // c: append const=1
    // ct: append 1, t (1-based)
    // ctt: append 1, t, t²
    let trend_start = 1 + usedlag;
    match regression {
        Regression::N => {}
        Regression::C => {
            for r in 0..nobs {
                rhs[r * ncols + trend_start] = 1.0;
            }
        }
        Regression::Ct => {
            for r in 0..nobs {
                let t = (r + 1) as f64;
                rhs[r * ncols + trend_start] = 1.0;
                rhs[r * ncols + trend_start + 1] = t;
            }
        }
        Regression::Ctt => {
            for r in 0..nobs {
                let t = (r + 1) as f64;
                rhs[r * ncols + trend_start] = 1.0;
                rhs[r * ncols + trend_start + 1] = t;
                rhs[r * ncols + trend_start + 2] = t * t;
            }
        }
    }

    // endog: xdiff[-nobs..] = xdiff[xdiff.len()-nobs..]
    let endog = &xdiff[xdiff.len() - nobs..];

    // A degenerate design that fits exactly (zero residual variance) makes the standard
    // error zero and the t-statistic ±∞. Valid data never fits exactly, so this only
    // touches numerically-degenerate inputs; report them as NaN rather than ±∞.
    let t0 = ols_tstat_col0(endog, &rhs, nobs, ncols);
    let t0 = if t0.is_finite() { t0 } else { f64::NAN };
    (t0, nobs)
}

/// Extract a packed (contiguous) submatrix from a row-major matrix with stride `row_stride`.
///
/// `full_rhs` has rows of length `row_stride`; we want only the first `ncols` columns.
fn extract_packed(exog: &[f64], nobs: usize, ncols: usize, row_stride: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(nobs * ncols);
    for r in 0..nobs {
        out.extend_from_slice(&exog[r * row_stride..r * row_stride + ncols]);
    }
    out
}

/// ADF IC computation — replicates `OLS(...).fit().aic` / `.bic`.
///
/// `k_params = df_model + k_constant = ncols` (all columns including const).
/// `llf = −nobs/2 · (1 + ln(2π) + ln(ssr/nobs))`
/// `AIC = −2·llf + 2·k_params`
/// `BIC = −2·llf + ln(nobs)·k_params`
///
/// `exog` is row-major with stride `row_stride`; only the first `ncols` cols are used.
fn compute_ic(
    endog: &[f64],
    exog: &[f64],
    nobs: usize,
    ncols: usize,
    row_stride: usize,
    method: AutoLag,
) -> f64 {
    let packed = extract_packed(exog, nobs, ncols, row_stride);
    let ssr = ols_ssr(endog, &packed, nobs, ncols);
    let nobs_f = nobs as f64;
    let llf = -nobs_f / 2.0 * (1.0 + (2.0 * std::f64::consts::PI).ln() + (ssr / nobs_f).ln());
    let k = ncols as f64;
    match method {
        AutoLag::Aic => -2.0 * llf + 2.0 * k,
        AutoLag::Bic => -2.0 * llf + nobs_f.ln() * k,
        _ => unreachable!(),
    }
}

/// Absolute t-statistic on the last column (for t-stat autolag).
///
/// `exog` is row-major with stride `row_stride`; only the first `ncols` cols are used.
fn ols_last_tstat(
    endog: &[f64],
    exog: &[f64],
    nobs: usize,
    ncols: usize,
    row_stride: usize,
) -> f64 {
    use nalgebra::{DMatrix, DVector};
    let packed = extract_packed(exog, nobs, ncols, row_stride);
    let y = DVector::from_column_slice(endog);
    let x = DMatrix::from_row_slice(nobs, ncols, &packed);
    let xtx = x.tr_mul(&x);
    let xty = x.tr_mul(&y);
    let (beta, xtx_inv) = xtx_solve(xtx, &xty);
    let resid = &y - &x * &beta;
    let ssr = resid.dot(&resid);
    let df = (nobs - ncols) as f64;
    let sigma2 = ssr / df;
    let last = ncols - 1;
    let se = (sigma2 * xtx_inv[(last, last)]).sqrt();
    beta[last] / se
}
