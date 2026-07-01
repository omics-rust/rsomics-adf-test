use std::io::{BufRead, stdin};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use rsomics_common::{CommonFlags, RsomicsError, ToolMeta, run};
use serde::Serialize;

use rsomics_adf_test::{AutoLag, Regression, adfuller};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

/// Augmented Dickey-Fuller unit-root test (statsmodels.tsa.stattools.adfuller equivalent).
///
/// Reads one floating-point value per line from SERIES (or stdin with `-`).
/// Tests whether the series has a unit root; small p-value → reject unit root (stationary).
#[derive(Parser, Debug)]
#[command(name = "rsomics-adf-test", version, about, long_about = None)]
pub struct Cli {
    /// Input series: one float per line; `-` reads stdin.
    #[arg(value_name = "SERIES")]
    pub series: Option<PathBuf>,

    /// Regression type: c (constant), ct (+trend), ctt (+quadratic), n (none).
    #[arg(long, default_value = "c", value_name = "TYPE")]
    pub regression: String,

    /// Lag selection: AIC, BIC, t-stat, none.
    #[arg(long, default_value = "AIC", value_name = "METHOD")]
    pub autolag: String,

    /// Fixed maximum lag (overrides lag search when --autolag none; sets upper bound otherwise).
    #[arg(long, value_name = "N")]
    pub maxlag: Option<usize>,

    #[command(flatten)]
    pub common: CommonFlags,
}

#[derive(Serialize)]
struct Output {
    adf_stat: f64,
    pvalue: f64,
    usedlag: usize,
    nobs: usize,
    crit_1pct: f64,
    crit_5pct: f64,
    crit_10pct: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    icbest: Option<f64>,
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common;
        run(&common, META, move || {
            let regression = Regression::parse(&self.regression).ok_or_else(|| {
                RsomicsError::InvalidInput(format!(
                    "unknown regression '{}'; expected c/ct/ctt/n",
                    self.regression
                ))
            })?;

            let autolag = AutoLag::parse(&self.autolag).ok_or_else(|| {
                RsomicsError::InvalidInput(format!(
                    "unknown autolag '{}'; expected AIC/BIC/t-stat/none",
                    self.autolag
                ))
            })?;

            let x = read_series(self.series.as_ref())?;
            let result = adfuller(&x, self.maxlag, regression, autolag);

            Ok(Output {
                adf_stat: result.adf_stat,
                pvalue: result.pvalue,
                usedlag: result.usedlag,
                nobs: result.nobs,
                crit_1pct: result.crit_1pct,
                crit_5pct: result.crit_5pct,
                crit_10pct: result.crit_10pct,
                icbest: result.icbest,
            })
        })
    }
}

fn read_series(path: Option<&PathBuf>) -> rsomics_common::Result<Vec<f64>> {
    let reader: Box<dyn BufRead> = match path {
        None => Box::new(stdin().lock()),
        Some(p) if p.as_os_str() == "-" => Box::new(stdin().lock()),
        Some(p) => Box::new(std::io::BufReader::new(
            std::fs::File::open(p).map_err(RsomicsError::Io)?,
        )),
    };

    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(RsomicsError::Io)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: f64 = fast_float2::parse(trimmed)
            .map_err(|_| RsomicsError::InvalidInput(format!("cannot parse float: '{trimmed}'")))?;
        values.push(v);
    }
    Ok(values)
}

#[test]
fn cli_debug_assert() {
    use clap::CommandFactory;
    Cli::command().debug_assert();
}
