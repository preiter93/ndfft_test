//! Out-of-Place Transpose Algorithms

/// Block size of tiling transpose
const BLOCK_SIZE: usize = 16;

/// Size for simple transpose
const SIZE_SIMPLE: usize = 16 * 16;

/// Size for tile-based transpose
const SIZE_TILE: usize = 512 * 512;

/// Recusrive limit of recursive transpose
const RECURSION_LIMIT: usize = 128;

/// Out-of Place transpose
///
/// Uses simple transpose algorithm for small matrix sizes,
/// a loop blocking algorithm for medium matrices and
/// recursive cache oblivious algorithm for larger matrices.
///
/// # Arguments
///
/// * `src` - Flattened 2D array with rows * cols elements, input
/// * `dst` - Flattened 2D array with rows * cols elements, output
/// * `rows` - Number of rows
/// * `cols` - Number of cols
pub fn oop_transpose<T: Copy>(src: &[T], dst: &mut [T], rows: usize, cols: usize) {
    if rows * cols <= SIZE_SIMPLE {
        oop_transpose_small(src, dst, rows, cols);
    } else if rows * cols <= SIZE_TILE {
        oop_transpose_medium(src, dst, rows, cols, BLOCK_SIZE);
    } else {
        oop_transpose_large(src, dst, rows, cols, BLOCK_SIZE);
    }
}

/// Simple out-of-place transpose
///
/// # Arguments
///
/// * `src` - Flattened 2D array with rows * cols elements, input
/// * `dst` - Flattened 2D array with rows * cols elements, output
/// * `rows` - Number of rows
/// * `cols` - Number of cols
///
/// # Unsafe
///
/// src.len() and dst.len() must equal rows * cols
pub fn oop_transpose_small<T: Copy>(src: &[T], dst: &mut [T], rows: usize, cols: usize) {
    assert!(src.len() == rows * cols, "{} != {}", src.len(), rows * cols);
    assert!(dst.len() == rows * cols, "{} != {}", dst.len(), rows * cols);

    for r in 0..rows {
        for c in 0..cols {
            let i = c + r * cols;
            let j = r + c * rows;
            unsafe {
                *dst.get_unchecked_mut(j) = *src.get_unchecked(i);
            }
        }
    }
}

/// Transpose with loop blocking optimzation #2
///
/// Splits rows and columns into blocks of size `block_size`.
/// This enhances cache locality of the transpose and is known als
/// loop blocking or tiling optimization.
///
/// # Arguments
///
/// * `src` - Flattened 2D array with rows * cols elements, input
/// * `dst` - Flattened 2D array with rows * cols elements, output
/// * `rows` - Number of rows
/// * `cols` - Number of cols
/// * `block_size` - Size of each block, its total length is `block_size` * `block_size`
///
/// # Unsafe
///
/// src.len() and dst.len() must equal rows * cols
pub fn oop_transpose_medium<T: Copy>(
    src: &[T],
    dst: &mut [T],
    rows: usize,
    cols: usize,
    block_size: usize,
) {
    assert!(src.len() == rows * cols, "{} != {}", src.len(), rows * cols);
    assert!(dst.len() == rows * cols, "{} != {}", dst.len(), rows * cols);
    // Number of blocks needed
    let block_rows = rows / block_size;
    let block_cols = cols / block_size;
    let remain_rows = rows - block_rows * block_size;
    let remain_cols = cols - block_cols * block_size;
    //
    // Loop over blocks
    //
    for block_col in 0..block_cols {
        for block_row in 0..block_rows {
            //
            // Loop over block entries
            //
            unsafe {
                transpose_tile(
                    src,
                    dst,
                    rows,
                    cols,
                    block_row * block_size,
                    block_col * block_size,
                    block_size,
                    block_size,
                );
            }
        }
    }

    //
    // Loop over remainders
    //
    if remain_cols > 0 {
        for block_row in 0..block_rows {
            unsafe {
                transpose_tile(
                    src,
                    dst,
                    rows,
                    cols,
                    block_row * block_size,
                    cols - remain_cols,
                    block_size,
                    remain_cols,
                );
            }
        }
    }

    if remain_rows > 0 {
        for block_col in 0..block_cols {
            unsafe {
                transpose_tile(
                    src,
                    dst,
                    rows,
                    cols,
                    rows - remain_rows,
                    block_col * block_size,
                    remain_rows,
                    block_size,
                );
            }
        }
    }

    if remain_cols > 0 && remain_rows > 0 {
        unsafe {
            transpose_tile(
                src,
                dst,
                rows,
                cols,
                rows - remain_rows,
                cols - remain_cols,
                remain_rows,
                remain_cols,
            );
        }
    }
}

/// Transpose a single sub-Tile
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_tile<T: Copy>(
    src: &[T],
    dst: &mut [T],
    rows: usize,
    cols: usize,
    first_row: usize,
    first_col: usize,
    num_rows_per_block: usize,
    num_cols_per_block: usize,
) {
    for tile_col in 0..num_cols_per_block {
        for tile_row in 0..num_rows_per_block {
            let mat_row = first_row + tile_row;
            let mat_col = first_col + tile_col;
            let i = mat_col + mat_row * cols;
            let j = mat_row + mat_col * rows;
            *dst.get_unchecked_mut(j) = *src.get_unchecked(i);
        }
    }
}

/// Transpose based on recursion and loop-blocking
///
/// Divide matrix recursively into smaller submatrixes until number of rows
/// and columns falls below a treshold. These submatrices are
/// then transposed using the loop-blocking based approach, see [`transpose_tiling`].
///
/// # Arguments
///
/// * `src` - Flattened 2D array with rows * cols elements, input
/// * `dst` - Flattened 2D array with rows * cols elements, output
/// * `rows` - Number of rows
/// * `cols` - Number of cols
/// * `block_size` - Size of each block, its total length is `block_size` * `block_size`
///
/// # Unsafe
///
/// src.len() and dst.len() must equal rows * cols
pub fn oop_transpose_large<T: Copy>(
    src: &[T],
    dst: &mut [T],
    rows: usize,
    cols: usize,
    block_size: usize,
) {
    assert!(src.len() == rows * cols, "{} != {}", src.len(), rows * cols);
    assert!(dst.len() == rows * cols, "{} != {}", dst.len(), rows * cols);
    transpose_recursive(src, dst, 0, 0, rows, cols, rows, cols, block_size);
}

/// Transpose based on recursive division of rows and cols
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn transpose_recursive<T: Copy>(
    src: &[T],
    dst: &mut [T],
    first_row: usize,
    first_col: usize,
    num_rows: usize,
    num_cols: usize,
    total_rows: usize,
    total_cols: usize,
    block_size: usize,
) {
    if (num_rows <= RECURSION_LIMIT) & (num_cols < RECURSION_LIMIT) {
        // Number of blocks needed
        let block_rows = num_rows / block_size;
        let block_cols = num_cols / block_size;
        let remain_rows = num_rows - block_rows * block_size;
        let remain_cols = num_cols - block_cols * block_size;
        //
        // Loop over blocks
        //
        for block_col in 0..block_cols {
            for block_row in 0..block_rows {
                //
                // Loop over block entries
                //
                unsafe {
                    transpose_tile(
                        src,
                        dst,
                        total_rows,
                        total_cols,
                        block_row * block_size + first_row,
                        block_col * block_size + first_col,
                        block_size,
                        block_size,
                    );
                }
            }
        }

        //
        // Loop over remainders
        //
        if remain_cols > 0 {
            for block_row in 0..block_rows {
                unsafe {
                    transpose_tile(
                        src,
                        dst,
                        total_rows,
                        total_cols,
                        block_row * block_size + first_row,
                        num_cols - remain_cols + first_col,
                        block_size,
                        remain_cols,
                    );
                }
            }
        }

        if remain_rows > 0 {
            for block_col in 0..block_cols {
                unsafe {
                    transpose_tile(
                        src,
                        dst,
                        total_rows,
                        total_cols,
                        num_rows - remain_rows + first_row,
                        block_col * block_size + first_col,
                        remain_rows,
                        block_size,
                    );
                }
            }
        }

        if remain_cols > 0 && remain_rows > 0 {
            unsafe {
                transpose_tile(
                    src,
                    dst,
                    total_rows,
                    total_cols,
                    num_rows - remain_rows + first_row,
                    num_cols - remain_cols + first_col,
                    remain_rows,
                    remain_cols,
                );
            }
        }
    //
    // Subdivide rows
    //
    } else if num_rows >= num_cols {
        transpose_recursive(
            src,
            dst,
            first_row,
            first_col,
            num_rows / 2,
            num_cols,
            total_rows,
            total_cols,
            block_size,
        );
        transpose_recursive(
            src,
            dst,
            first_row + num_rows / 2,
            first_col,
            num_rows - num_rows / 2,
            num_cols,
            total_rows,
            total_cols,
            block_size,
        );
    //
    // Subdivide cols
    //
    } else {
        transpose_recursive(
            src,
            dst,
            first_row,
            first_col,
            num_rows,
            num_cols / 2,
            total_rows,
            total_cols,
            block_size,
        );
        transpose_recursive(
            src,
            dst,
            first_row,
            first_col + num_cols / 2,
            num_rows,
            num_cols - num_cols / 2,
            total_rows,
            total_cols,
            block_size,
        );
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
    fn test_transposes() {
        // Bunch of arbitraty sizes
        let sizes = [4, 5, 13, 16, 54, 67, 813];

        for rows in sizes {
            for cols in sizes {
                let src = test_array(rows, cols);
                let src_sl = src.as_slice().unwrap();

                // F #1
                let mut dst = Array2::<f64>::zeros((cols, rows));
                let dst_sl = dst.as_slice_mut().unwrap();
                oop_transpose_small(src_sl, dst_sl, rows, cols);
                assert!(src.t() == dst);

                // F #1
                let mut dst = Array2::<f64>::zeros((cols, rows));
                let dst_sl = dst.as_slice_mut().unwrap();
                oop_transpose_medium(src_sl, dst_sl, rows, cols, 16);
                assert!(src.t() == dst);

                // F #1
                let mut dst = Array2::<f64>::zeros((cols, rows));
                let dst_sl = dst.as_slice_mut().unwrap();
                oop_transpose_large(src_sl, dst_sl, rows, cols, 16);
                assert!(src.t() == dst);
            }
        }
    }
}
