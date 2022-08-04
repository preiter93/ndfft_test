# ndfft_test

Test multidimensional FFT transforms

## Discussion
<https://github.com/ejmahler/RustFFT/issues/85>

## Methods
Assuming we have a two-dimensional array of size n x n, the two approaches are:

"transpose"
- Transpose data, if necessary
- Compute 1 fft of size n x n
- Transpose back, if necessary

"ndrustfft"
- Iterate over 1d lanes -> Array1View
- Copy data into slice, if 1d view is not contiguous,
- Compute n ffts of size n

## Benches
Time in ms

FFT along outer axis (transpose is not necessary)
| N 	| ndrustfft | transpose |
| :---  |  	:---:   |   ---: 	|
| 128 	| 	0.09 	| 	0.04	|
| 256 	| 	0.4 	| 	0.24 	|
| 512 	| 	2.22 	| 	1.64 	|
| 1024 	| 	10.3 	| 	4.77	|

FFT along inner axis (transpose is necessary)
| N 	| ndrustfft | transpose |
| :---  |  	:---:   |   ---: 	|
| 128   |	0.24 	| 	0.1     |
| 256	|  	1.12 	| 	0.56    |
| 512	| 	8.41 	| 	5.08    |
| 1024	| 	42.92	| 	19.49   |

Advantages of "transpose"-approach
- Performance (~50%)
- No additional dependencies
Disadvantages
- Larger memory requirements
- Not easily parallelizable
