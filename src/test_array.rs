use ndarray::Array2;
use ndrustfft::Complex;

pub fn test_array2(n: usize) -> Array2<Complex<f64>> {
    Array2::from_shape_vec((n, n), test_vec(n * n)).unwrap()
}

pub fn test_vec(n: usize) -> Vec<Complex<f64>> {
    (0..n).map(|x| Complex::new(x as f64, x as f64)).collect()
}
