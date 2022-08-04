#![allow(dead_code)]

use ndarray::Array2;
use ndfft_test::ndfft_with_ndrustfft::fft2d_with_ndrustfft;
use ndfft_test::ndfft_with_transpose::fft2d_with_transpose;
use ndfft_test::test_array::test_array2;
use ndfft_test::test_array::test_vec;
use ndrustfft::FftHandler;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

fn main() {
    let n = 28;
    let axis = 0;

    // Ndrustfft
    let v = test_array2(n);
    let mut vhat = Array2::<Complex<f64>>::zeros((n, n));
    let mut handler: FftHandler<f64> = FftHandler::new(n);
    fft2d_with_ndrustfft(&v, &mut vhat, &mut handler, axis);
    //println!("{:?}", vhat);
    //println!("");

    let mut v: Vec<Complex<f64>> = test_vec(n * n);
    let mut scratch = vec![Complex::default(); n * n];
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    fft2d_with_transpose(&mut v, &mut scratch, n, n, &fft, axis);
    //println!("{:?}", v);

    let tol = 1e-6;
    for (a, b) in v.iter().zip(vhat.iter()) {
        assert!((a.re - b.re).abs() < tol);
        assert!((a.im - b.im).abs() < tol);
    }
    println!("Test successfull");
}
