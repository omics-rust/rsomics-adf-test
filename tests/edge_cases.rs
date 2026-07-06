//! Adversarial inputs, each reproduced against statsmodels 0.14.6 adfuller.
//!
//! statsmodels raises for non-finite / constant / too-short series; we mirror those
//! as loud errors. For rank-deficient designs (perfect trend, alternating series,
//! repeating block) statsmodels' `pinv` fit returns a numerically-degenerate but
//! defined value — we must likewise return a defined (finite-or-nan) value, never panic.

use std::io::Write;
use std::process::{Command, Stdio};

use rsomics_adf_test::{AdfError, AutoLag, Regression, adfuller};

// ── library API: statsmodels raises → we return Err ──────────────────────────

#[test]
fn nan_is_rejected() {
    let x = [0.0, 1.0, f64::NAN, 3.0, 4.0, 5.0];
    assert_eq!(
        adfuller(&x, None, Regression::C, AutoLag::Aic).unwrap_err(),
        AdfError::NonFinite
    );
}

#[test]
fn inf_is_rejected() {
    let x = [0.0, 1.0, f64::INFINITY, 3.0, 4.0, 5.0];
    assert_eq!(
        adfuller(&x, None, Regression::C, AutoLag::Aic).unwrap_err(),
        AdfError::NonFinite
    );
}

#[test]
fn constant_is_rejected() {
    let x = [3.0; 50];
    assert_eq!(
        adfuller(&x, None, Regression::C, AutoLag::Aic).unwrap_err(),
        AdfError::Constant
    );
}

#[test]
fn too_short_is_rejected() {
    // n=3, regression "c": nobs/2 - ntrend - 1 = 1 - 1 - 1 = -1 < 0.
    let x = [1.0, 2.0, 3.0];
    assert_eq!(
        adfuller(&x, None, Regression::C, AutoLag::Aic).unwrap_err(),
        AdfError::TooShort
    );
}

#[test]
fn maxlag_too_large_is_rejected() {
    let x = [1.0, 2.0, 3.5, 3.0];
    let got = adfuller(&x, Some(5), Regression::C, AutoLag::None);
    assert!(
        matches!(got, Err(AdfError::MaxlagTooLarge { .. })),
        "got {got:?}"
    );
}

// ── rank-deficient designs: statsmodels returns a defined value → so must we ──

fn assert_defined(tag: &str, x: &[f64], reg: Regression) {
    let r = adfuller(x, None, reg, AutoLag::Aic)
        .unwrap_or_else(|e| panic!("{tag}: unexpected error {e}"));
    assert!(
        r.adf_stat.is_finite() || r.adf_stat.is_nan(),
        "{tag}: adf_stat must be finite-or-nan, got {}",
        r.adf_stat
    );
}

#[test]
fn perfect_trend_no_panic() {
    let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    assert_defined("perfect_trend", &x, Regression::C);
}

#[test]
fn alternating_no_panic() {
    let x: Vec<f64> = (0..50)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    assert_defined("alternating", &x, Regression::C);
}

#[test]
fn repeating_block_no_panic() {
    let x: Vec<f64> = (0..51).map(|i| [1.0, 2.0, 3.0][i % 3]).collect();
    assert_defined("repeating_block", &x, Regression::C);
}

// ── binary: fail-loud cases exit non-zero with a stderr message ───────────────

fn run_bin(input: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rsomics-adf-test"))
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn binary");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().expect("wait")
}

fn assert_fail_loud(tag: &str, input: &str) {
    let out = run_bin(input);
    assert!(
        !out.status.success(),
        "{tag}: expected non-zero exit, got {}",
        out.status
    );
    assert!(
        !out.stderr.is_empty(),
        "{tag}: expected a stderr message on failure"
    );
}

#[test]
fn bin_nan_fails_loud() {
    assert_fail_loud("nan", "0\n1\nnan\n3\n4\n5\n");
}

#[test]
fn bin_inf_fails_loud() {
    assert_fail_loud("inf", "0\n1\ninf\n3\n4\n5\n");
}

#[test]
fn bin_constant_fails_loud() {
    let s: String = std::iter::repeat_n("3.0\n", 50).collect();
    assert_fail_loud("constant", &s);
}

#[test]
fn bin_too_short_fails_loud() {
    assert_fail_loud("too_short", "1\n2\n3\n");
}

#[test]
fn bin_perfect_trend_defined_no_panic() {
    let s: String = (0..50).map(|i| format!("{i}.0\n")).collect();
    let out = run_bin(&s);
    assert!(
        out.status.success(),
        "perfect_trend: expected success, got {} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}
