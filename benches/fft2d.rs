use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use ndfft_test::ndfft_with_ndrustfft::fft2d_with_ndrustfft;
use ndfft_test::ndfft_with_transpose::fft2d_with_transpose;
use ndfft_test::test_array::test_array2;
use ndfft_test::test_array::test_vec;
use ndrustfft::FftHandler;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;
const AXIS: usize = 0;
const FFT_SIZES: [usize; 4] = [128, 256, 512, 1024];

pub fn bench_fft2d_with_ndrustfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2d (ndrustfft) ");
    for n in FFT_SIZES.into_iter() {
        let name = format!("Size: {}", n);
        let v = test_array2(n);
        let mut vhat = Array2::<Complex<f64>>::zeros((n, n));
        let mut handler: FftHandler<f64> = FftHandler::new(n);
        group.bench_function(&name, |b| {
            b.iter(|| fft2d_with_ndrustfft(&v, &mut vhat, &mut handler, AXIS))
        });
    }
    group.finish();
}

pub fn bench_fft2d_with_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft2d (transpose) ");
    for n in FFT_SIZES.into_iter() {
        let name = format!("Size: {}", n);
        let mut v = test_vec(n * n);
        let mut scratch = vec![Complex::default(); n * n];
        let mut planner = FftPlanner::<f64>::new();
        let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(n);
        group.bench_function(&name, |b| {
            b.iter(|| fft2d_with_transpose(&mut v, &mut scratch, n, n, &fft, AXIS))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fft2d_with_ndrustfft,
    bench_fft2d_with_transpose
);
criterion_main!(benches);
