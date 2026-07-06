use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_adf_test::{AutoLag, Regression, adfuller};

fn bench_adf(c: &mut Criterion) {
    // 2000-observation random walk — matches the perf comparison scenario.
    let mut rng_state: u64 = 0x_dead_beef_1234_5678;
    let mut randn = || -> f64 {
        // xorshift64 → Box-Muller
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let u = (rng_state as f64 + 0.5) / (u64::MAX as f64 + 1.0);
        // approximate normal via 12 uniform summing trick (not perfect but reproducible)
        let _ = u;
        // simpler: cast bits to a reasonable range
        (rng_state as i64 as f64) / (i64::MAX as f64)
    };

    let rw: Vec<f64> = {
        let mut acc = 0.0_f64;
        (0..2000)
            .map(|_| {
                acc += randn();
                acc
            })
            .collect()
    };

    c.bench_function("adf_rw_2000_aic", |b| {
        b.iter(|| adfuller(&rw, None, Regression::C, AutoLag::Aic).unwrap())
    });

    c.bench_function("adf_rw_2000_bic", |b| {
        b.iter(|| adfuller(&rw, None, Regression::C, AutoLag::Bic).unwrap())
    });

    c.bench_function("adf_rw_2000_ct_aic", |b| {
        b.iter(|| adfuller(&rw, None, Regression::Ct, AutoLag::Aic).unwrap())
    });
}

criterion_group!(benches, bench_adf);
criterion_main!(benches);
