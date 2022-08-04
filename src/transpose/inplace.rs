//! In-Place Transpose Algorithm
//!
//! # Reference
//!
//! Based on:
//! F. Gustavson and D. Walker - Algorithms for in-place matrix transposition (2018)
use super::oop_transpose;

/// In-Place transpose of square and rectangular matrices
///
/// The input array `src` is overwritten by its transpose.
/// A work-space must be provided with a minimum size of 2 and a maximum
/// size of `rows` * `cols` which is used internally for out-of-place transposes.
/// The larger the provided workspace, the more efficient the transpose should be.
///
///  # Parameters
///
/// * `a`: Matrix of size cols x rows, with cols <= rows
/// * `w`: work-space. Used for out-of-place transpose of submatrices.
/// * `rows`: Number of rows
/// * `cols`: Number of cols
pub fn ip_transpose<T: Copy>(src: &mut [T], w: &mut [T], rows: usize, cols: usize) {
    assert!(src.len() == rows * cols, "{} != {}", src.len(), rows * cols);
    let iw = w.len();
    if rows >= cols {
        row_transpose(src, cols, rows, w, iw);
    } else {
        column_transpose(src, cols, rows, w, iw);
    }
}

/// In-Place transpose of a square matrix
///
/// # Parameters
///
/// * src - Square matrix of size n x n
/// * n - Number of rows and cols
pub fn square_transpose<T: Copy>(src: &mut [T], n: usize) {
    assert!(src.len() == n * n, "{} != {}", src.len(), n * n);
    for c in 0..n - 1 {
        for r in c + 1..n {
            let i = r * n + c;
            let j = c * n + r;
            src.swap(i, j);
        }
    }
}

/// Exchange operation takes two contiguous vectors
/// of length p and q and reverses their order in-place:
/// a1 .. ap b1 .. bq
/// ->
/// b1 .. bq a1 .. ap
///
/// # Parameters
///
/// * v - Vector of length (p + q)
/// * p - Size of first vector a
/// * q - Size of second vector b
fn exchange<T: Copy>(v: &mut [T], p: usize, q: usize) {
    if p >= q {
        for i in 0..q {
            v.swap(i, i + p);
        }
        if p != q {
            exchange(&mut v[q..], p - q, q);
        }
    } else {
        for i in 0..p {
            v.swap(i, i + q);
        }
        exchange(&mut v[..q], p, q - p);
    }
}
/// Return largest power of 2 of *m*.
///
/// For example, returns 4 for *m*=7
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn largest_power_of_two(m: usize) -> usize {
    2_usize.pow((m as f64 - 1.).log2() as u32)
}

/// Unshuffle pairs of shuffled vectors
///
/// # Parameters
///
/// * v - Vector of length (la + lb)m made up of m shuffled pairs of vectors
/// of length la and lb
///
/// # Reference
/// F. Gustavson and D. Walker - Algorithms for in-place matrix transposition (2018)
fn unshuffle<T: Copy>(v: &mut [T], la: usize, lb: usize, m: usize) {
    if m > 1 {
        let m1 = largest_power_of_two(m);
        unshuffle(v, la, lb, m1);
        unshuffle(&mut v[(la + lb) * m1..], la, lb, m - m1);
        if (la * (m - m1) > 0) & (lb * m1 > 0) {
            exchange(&mut v[la * m1..], lb * m1, la * (m - m1));
        }
    }
}

/// Shuffle pairs of unshuffled vectors
///
/// # Parameters
///
/// * v - Vector of length (la + lb)m made up of m shuffled pairs of vectors
/// of length la and lb
fn shuffle<T: Copy>(v: &mut [T], la: usize, lb: usize, m: usize) {
    if m > 1 {
        let m1 = largest_power_of_two(m);
        if (la * (m - m1) > 0) & (lb * m1 > 0) {
            exchange(&mut v[la * m1..], la * (m - m1), lb * m1);
        }
        shuffle(v, la, lb, m1);
        shuffle(&mut v[(la + lb) * m1..], la, lb, m - m1);
    }
}

/// Swap-Bases Matrix Transpose of Panel of Square Matrices
///
/// # Parameters
///
/// * a: Matrix A of size qn x n
fn partition<T: Copy>(a: &mut [T], q: usize, n: usize) {
    if q == 1 {
        square_transpose(&mut a[..n * n], n);
    } else {
        let q2 = q / 2;
        let q1 = q - q2;
        unshuffle(a, q1 * n, q2 * n, n);
        partition(a, q1, n);
        partition(&mut a[q1 * n * n..], q2, n);
    }
}

/// Swap-Bases Matrix Transpose of Panel of Square Matrices
///
/// # Parameters
///
/// * a: Matrix A of size n x qn
fn join<T: Copy>(a: &mut [T], q: usize, n: usize) {
    if q == 1 {
        square_transpose(&mut a[..n * n], n);
    } else {
        let q2 = q / 2;
        let q1 = q - q2;
        join(a, q1, n);
        join(&mut a[q1 * n * n..], q2, n);
        shuffle(a, q1 * n, q2 * n, n);
    }
}

///  In-Place Swap-Based Matrix Transpose
///
///  # Parameters
///
/// * `a`: Matrix of size cols x rows, with cols >= rows
/// * `rows`: Number of rows
/// * `cols`: Number of cols
/// * `w`: work-space of size iw
/// * `iw`: work-space size
///
///  # Reference
///
/// F. Gustavson and D. Walker - Algorithms for in-place matrix
/// transposition (2018)
fn column_transpose<T: Copy>(a: &mut [T], rows: usize, cols: usize, w: &mut [T], iw: usize) {
    if rows * cols <= iw {
        oop_transpose(a, &mut w[..rows * cols], rows, cols);
    } else {
        let q = rows / cols;
        let r = rows % cols;
        unshuffle(a, q * cols, r, cols);
        partition(a, q, cols);
        row_transpose(&mut a[q * cols * cols..], r, cols, w, iw);
    }
}

///  In-Place Swap-Based Matrix Transpose
///
///  # Parameters
///
/// * `a`: Matrix of size cols x rows, with cols <= rows
/// * `rows`: Number of rows
/// * `cols`: Number of cols
/// * `w`: work-space of size iw
/// * `iw`: work-space size
///
///  # Reference
///
/// F. Gustavson and D. Walker - Algorithms for in-place matrix
/// transposition (2018)
fn row_transpose<T: Copy>(a: &mut [T], rows: usize, cols: usize, w: &mut [T], iw: usize) {
    if rows * cols <= iw {
        oop_transpose(a, &mut w[..rows * cols], rows, cols);
    } else {
        let q = cols / rows;
        let r = cols % rows;
        column_transpose(&mut a[q * rows * rows..], rows, r, w, iw);
        join(a, q, rows);
        shuffle(a, q * rows, r, rows);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::Array2;

    /// Make an rows x cols test array
    fn test_array(rows: usize, cols: usize) -> Array2<f64> {
        let mut array: Array2<f64> = Array2::zeros((rows, cols));
        for (i, v) in array.iter_mut().enumerate() {
            *v = i as f64;
        }
        array
    }

    #[test]
    fn test_square_transpose() {
        let n = 4;
        let mut src = test_array(n, n);
        let cmp = test_array(n, n);

        let src_sl = src.as_slice_mut().unwrap();
        square_transpose(src_sl, n);
        assert!(src == cmp.t());
    }

    #[test]
    fn test_inplace_transpose() {
        // Bunch of arbitraty sizes
        let sizes = [4, 5, 13, 16, 54, 67, 512, 813];

        for rows in sizes {
            for cols in sizes {
                let (n, m) = (rows, cols);
                let mut src = test_array(n, m);
                let cmp = test_array(n, m);
                let src_sl = src.as_slice_mut().unwrap();
                let iw = 4;
                let mut w: Vec<f64> = vec![0.; iw];
                ip_transpose(src_sl, &mut w, n, m);
                assert!(src.into_shape((m, n)).unwrap() == cmp.t());
            }
        }
    }
}
