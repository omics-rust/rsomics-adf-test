# rsomics-adf-test

Augmented Dickey-Fuller unit-root test — a value-exact Rust reimplementation of
`statsmodels.tsa.stattools.adfuller`.

```
rsomics-adf-test [OPTIONS] [SERIES]
```

Reads one floating-point value per line from `SERIES` (or stdin with `-`).
Returns the ADF statistic, p-value, number of lags used, sample size, and MacKinnon (2010)
critical values at 1%, 5%, 10%.

## Usage

```
Options:
  --regression <TYPE>    c (constant), ct (+trend), ctt (+quadratic), n (none)  [default: c]
  --autolag <METHOD>     AIC, BIC, t-stat, none  [default: AIC]
  --maxlag <N>           Fixed maximum lag (or search upper bound)
  --json                 Emit JSON envelope
```

## Performance

Measured on mini_m2 (aarch64-apple-darwin), 2000-observation random walk, `--regression c --autolag AIC`:

| | wall time |
|---|---|
| rsomics-adf-test 0.1.0 | 5.3 ms |
| statsmodels 0.14.6 | 1145 ms |

**216× end-to-end; 6.7× compute-only** (Rust pays process startup + file I/O; Python number excludes interpreter startup).

## Install

```
cargo install rsomics-adf-test
```

## Origin

This crate is an independent Rust reimplementation of `statsmodels.tsa.stattools.adfuller` based on:

- Said, S.E. & Dickey, D.A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. _Biometrika_ 71(3), 599–607.
- MacKinnon, J.G. (1994). Approximate asymptotic distribution functions for unit-root and cointegration tests. _Journal of Business & Economic Statistics_ 12(2), 167–176.
- MacKinnon, J.G. (2010). Critical values for cointegration tests. _Queen's University Economics Department Working Paper No. 1227_.
- The statsmodels 0.14.6 source (BSD-3-Clause) was read to ensure exact algorithmic compatibility (column ordering, trend indexing, IC formula, two-branch p-value polynomial).

License: MIT OR Apache-2.0.
Upstream credit: [statsmodels](https://www.statsmodels.org/) (BSD-3-Clause).
