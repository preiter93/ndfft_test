use ndarray::Array2;
use ndrustfft::{ndfft, Complex, FftHandler};

pub fn fft2d_with_ndrustfft(
    v: &Array2<Complex<f64>>,
    vhat: &mut Array2<Complex<f64>>,
    handler: &mut FftHandler<f64>,
    axis: usize,
) {
    ndfft(v, vhat, handler, axis);
}
