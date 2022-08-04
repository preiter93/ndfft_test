use crate::transpose::oop_transpose;
use rustfft::{num_complex::Complex, Fft};
use std::sync::Arc;

pub fn fft2d_with_transpose(
    v: &mut [Complex<f64>],
    scratch: &mut [Complex<f64>],
    m: usize,
    n: usize,
    fft: &Arc<dyn Fft<f64>>,
    axis: usize,
) {
    assert!(v.len() == m * n);
    if axis == 1 {
        let scratch_len = fft.get_inplace_scratch_len();
        assert!(scratch.len() >= scratch_len);
        fft.process_with_scratch(v, scratch);
    } else {
        let scratch_len = m * n;
        assert!(scratch.len() >= scratch_len);
        oop_transpose(&v, scratch, n, n);
        fft.process_with_scratch(scratch, v);
        oop_transpose(&scratch, v, n, n);
    }
}
