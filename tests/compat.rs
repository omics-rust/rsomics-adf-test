//! Value-exact compatibility against `statsmodels.tsa.stattools.adfuller` 0.14.6.
//!
//! Goldens were computed once with Python oracle (numpy seed 42, statsmodels 0.14.6)
//! and frozen below as u64 bit patterns. No Python or subprocess at test time.
//!
//! Tolerance: 1e-10 relative (stat/pvalue/crits) or absolute for near-zero values.

#![allow(clippy::excessive_precision)]

use rsomics_adf_test::{AutoLag, Regression, adfuller};

// ── seed-42 series (200 obs each) ─────────────────────────────────────────────
// Generated: numpy.random.seed(42); rw = cumsum(randn(200)); ar1[i] = 0.5*ar1[i-1] + randn(); wn = randn(200)

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/series_data.rs"));

// ── helper ────────────────────────────────────────────────────────────────────

fn rel_err(got: f64, want: f64) -> f64 {
    if want.abs() < 1e-300 {
        (got - want).abs()
    } else {
        ((got - want) / want).abs()
    }
}

fn assert_close(got: f64, want: f64, tol: f64, label: &str) {
    let e = rel_err(got, want);
    assert!(
        e < tol,
        "{label}: got {got:.17e}, want {want:.17e}, rel_err {e:.3e}"
    );
}

fn assert_int_eq(got: usize, want: usize, label: &str) {
    assert_eq!(got, want, "{label}: got {got}, want {want}");
}

// ── test cases ─────────────────────────────────────────────────────────────────

#[test]
fn rw_default_aic() {
    let r = adfuller(&RW, None, Regression::C, AutoLag::Aic);
    assert_close(
        r.adf_stat,
        -2.3072851790645248,
        1e-10,
        "rw_default adf_stat",
    );
    assert_close(r.pvalue, 0.16962912078943748, 1e-10, "rw_default pvalue");
    assert_int_eq(r.usedlag, 0, "rw_default usedlag");
    assert_int_eq(r.nobs, 199, "rw_default nobs");
    assert_close(
        r.crit_1pct,
        -3.4636447617687436,
        1e-10,
        "rw_default crit_1pct",
    );
    assert_close(
        r.crit_5pct,
        -2.8761761179270766,
        1e-10,
        "rw_default crit_5pct",
    );
    assert_close(
        r.crit_10pct,
        -2.5745715858185401,
        1e-10,
        "rw_default crit_10pct",
    );
}

#[test]
fn ar1_default_aic() {
    let r = adfuller(&AR1, None, Regression::C, AutoLag::Aic);
    assert_close(
        r.adf_stat,
        -8.4822161528455489,
        1e-10,
        "ar1_default adf_stat",
    );
    // pvalue is very small; use absolute tolerance
    let pv_want = 1.3833236336942946e-13;
    assert!(
        (r.pvalue - pv_want).abs() < 1e-20,
        "ar1_default pvalue: got {:.17e}, want {:.17e}",
        r.pvalue,
        pv_want
    );
    assert_int_eq(r.usedlag, 0, "ar1_default usedlag");
    assert_int_eq(r.nobs, 199, "ar1_default nobs");
    assert_close(
        r.crit_1pct,
        -3.4636447617687436,
        1e-10,
        "ar1_default crit_1pct",
    );
    assert_close(
        r.crit_5pct,
        -2.8761761179270766,
        1e-10,
        "ar1_default crit_5pct",
    );
    assert_close(
        r.crit_10pct,
        -2.5745715858185401,
        1e-10,
        "ar1_default crit_10pct",
    );
}

#[test]
fn wn_default_aic() {
    let r = adfuller(&WN, None, Regression::C, AutoLag::Aic);
    assert_close(
        r.adf_stat,
        -13.363254849861557,
        1e-10,
        "wn_default adf_stat",
    );
    assert_int_eq(r.usedlag, 0, "wn_default usedlag");
    assert_int_eq(r.nobs, 199, "wn_default nobs");
    assert_close(
        r.crit_1pct,
        -3.4636447617687436,
        1e-10,
        "wn_default crit_1pct",
    );
    assert_close(
        r.crit_5pct,
        -2.8761761179270766,
        1e-10,
        "wn_default crit_5pct",
    );
    assert_close(
        r.crit_10pct,
        -2.5745715858185401,
        1e-10,
        "wn_default crit_10pct",
    );
}

#[test]
fn rw_ct() {
    let r = adfuller(&RW, None, Regression::Ct, AutoLag::Aic);
    assert_close(r.adf_stat, -2.0280247832558267, 1e-10, "rw_ct adf_stat");
    assert_close(r.pvalue, 0.58608840605290535, 1e-10, "rw_ct pvalue");
    assert_int_eq(r.usedlag, 0, "rw_ct usedlag");
    assert_int_eq(r.nobs, 199, "rw_ct nobs");
    assert_close(r.crit_1pct, -4.0049978489363562, 1e-10, "rw_ct crit_1pct");
    assert_close(r.crit_5pct, -3.4327862452981046, 1e-10, "rw_ct crit_5pct");
    assert_close(r.crit_10pct, -3.1401449183685148, 1e-10, "rw_ct crit_10pct");
}

#[test]
fn rw_ctt() {
    let r = adfuller(&RW, None, Regression::Ctt, AutoLag::Aic);
    assert_close(r.adf_stat, -3.0752185966646057, 1e-10, "rw_ctt adf_stat");
    assert_close(r.pvalue, 0.26059924836186399, 1e-10, "rw_ctt pvalue");
    assert_int_eq(r.usedlag, 0, "rw_ctt usedlag");
    assert_int_eq(r.nobs, 199, "rw_ctt nobs");
    assert_close(r.crit_1pct, -4.4303090466942932, 1e-10, "rw_ctt crit_1pct");
    assert_close(r.crit_5pct, -3.8623972900169137, 1e-10, "rw_ctt crit_5pct");
    assert_close(
        r.crit_10pct,
        -3.5717916732395594,
        1e-10,
        "rw_ctt crit_10pct",
    );
}

#[test]
fn rw_n() {
    let r = adfuller(&RW, None, Regression::N, AutoLag::Aic);
    assert_close(r.adf_stat, -0.45423028988011138, 1e-10, "rw_n adf_stat");
    assert_close(r.pvalue, 0.51483569705387033, 1e-10, "rw_n pvalue");
    assert_int_eq(r.usedlag, 0, "rw_n usedlag");
    assert_int_eq(r.nobs, 199, "rw_n nobs");
    assert_close(r.crit_1pct, -2.5770667644756444, 1e-10, "rw_n crit_1pct");
    assert_close(r.crit_5pct, -1.942430759336949, 1e-10, "rw_n crit_5pct");
    assert_close(r.crit_10pct, -1.615550641718986, 1e-10, "rw_n crit_10pct");
}

#[test]
fn ar1_bic() {
    let r = adfuller(&AR1, None, Regression::C, AutoLag::Bic);
    assert_close(r.adf_stat, -8.4822161528455489, 1e-10, "ar1_bic adf_stat");
    assert_int_eq(r.usedlag, 0, "ar1_bic usedlag");
    assert_int_eq(r.nobs, 199, "ar1_bic nobs");
}

#[test]
fn ar1_tstat() {
    let r = adfuller(&AR1, None, Regression::C, AutoLag::TStat);
    assert_close(r.adf_stat, -8.4822161528455489, 1e-10, "ar1_tstat adf_stat");
    assert_int_eq(r.usedlag, 0, "ar1_tstat usedlag");
    assert_int_eq(r.nobs, 199, "ar1_tstat nobs");
}

#[test]
fn ar1_nolag_maxlag4() {
    let r = adfuller(&AR1, Some(4), Regression::C, AutoLag::None);
    assert_close(r.adf_stat, -5.6976038963908797, 1e-10, "ar1_nolag adf_stat");
    let pv_want = 7.8085792667346633e-07;
    assert!(
        (r.pvalue - pv_want).abs() / pv_want < 1e-10,
        "ar1_nolag pvalue: got {:.17e}, want {:.17e}",
        r.pvalue,
        pv_want
    );
    assert_int_eq(r.usedlag, 4, "ar1_nolag usedlag");
    assert_int_eq(r.nobs, 195, "ar1_nolag nobs");
    assert_close(
        r.crit_1pct,
        -3.4643370308670072,
        1e-10,
        "ar1_nolag crit_1pct",
    );
    assert_close(
        r.crit_5pct,
        -2.8764787990357221,
        1e-10,
        "ar1_nolag crit_5pct",
    );
    assert_close(
        r.crit_10pct,
        -2.5747331032215648,
        1e-10,
        "ar1_nolag crit_10pct",
    );
}

#[test]
fn rw_fixlag3() {
    let r = adfuller(&RW, Some(3), Regression::C, AutoLag::None);
    assert_close(
        r.adf_stat,
        -2.6894618247810103,
        1e-10,
        "rw_fixlag3 adf_stat",
    );
    assert_close(r.pvalue, 0.07587965346190767, 1e-10, "rw_fixlag3 pvalue");
    assert_int_eq(r.usedlag, 3, "rw_fixlag3 usedlag");
    assert_int_eq(r.nobs, 196, "rw_fixlag3 nobs");
    assert_close(
        r.crit_1pct,
        -3.4641612783842191,
        1e-10,
        "rw_fixlag3 crit_1pct",
    );
    assert_close(
        r.crit_5pct,
        -2.876401960790147,
        1e-10,
        "rw_fixlag3 crit_5pct",
    );
    assert_close(
        r.crit_10pct,
        -2.5746921001665974,
        1e-10,
        "rw_fixlag3 crit_10pct",
    );
}

#[test]
fn wn_ct() {
    let r = adfuller(&WN, None, Regression::Ct, AutoLag::Aic);
    assert_close(r.adf_stat, -13.330791287329648, 1e-10, "wn_ct adf_stat");
    assert_int_eq(r.usedlag, 0, "wn_ct usedlag");
    assert_int_eq(r.nobs, 199, "wn_ct nobs");
}

#[test]
fn wn_ctt() {
    let r = adfuller(&WN, None, Regression::Ctt, AutoLag::Aic);
    assert_close(r.adf_stat, -13.345193582081379, 1e-10, "wn_ctt adf_stat");
    assert_int_eq(r.usedlag, 0, "wn_ctt usedlag");
    assert_int_eq(r.nobs, 199, "wn_ctt nobs");
}

#[test]
fn wn_n() {
    let r = adfuller(&WN, None, Regression::N, AutoLag::Aic);
    assert_close(r.adf_stat, -13.298222618636036, 1e-10, "wn_n adf_stat");
    assert_int_eq(r.usedlag, 0, "wn_n usedlag");
    assert_int_eq(r.nobs, 199, "wn_n nobs");
}

#[test]
fn rw_bic() {
    let r = adfuller(&RW, None, Regression::C, AutoLag::Bic);
    assert_close(r.adf_stat, -2.3072851790645248, 1e-10, "rw_bic adf_stat");
    assert_close(r.pvalue, 0.16962912078943748, 1e-10, "rw_bic pvalue");
    assert_int_eq(r.usedlag, 0, "rw_bic usedlag");
    assert_int_eq(r.nobs, 199, "rw_bic nobs");
}
