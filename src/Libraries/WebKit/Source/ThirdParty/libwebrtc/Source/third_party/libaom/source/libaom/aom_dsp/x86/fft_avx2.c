/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include <immintrin.h>

#include "config/aom_dsp_rtcd.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/fft_common.h"

extern void aom_transpose_float_sse2(const float *A, float *B, int n);
extern void aom_fft_unpack_2d_output_sse2(const float *col_fft, float *output,
                                          int n);

// Generate the 1d forward transforms for float using _mm256
GEN_FFT_8(static inline void, avx2, float, __m256, _mm256_load_ps,
          _mm256_store_ps, _mm256_set1_ps, _mm256_add_ps, _mm256_sub_ps,
          _mm256_mul_ps)
GEN_FFT_16(static inline void, avx2, float, __m256, _mm256_load_ps,
           _mm256_store_ps, _mm256_set1_ps, _mm256_add_ps, _mm256_sub_ps,
           _mm256_mul_ps)
GEN_FFT_32(static inline void, avx2, float, __m256, _mm256_load_ps,
           _mm256_store_ps, _mm256_set1_ps, _mm256_add_ps, _mm256_sub_ps,
           _mm256_mul_ps)

void aom_fft8x8_float_avx2(const float *input, float *temp, float *output) {
  aom_fft_2d_gen(input, temp, output, 8, aom_fft1d_8_avx2,
                 aom_transpose_float_sse2, aom_fft_unpack_2d_output_sse2, 8);
}

void aom_fft16x16_float_avx2(const float *input, float *temp, float *output) {
  aom_fft_2d_gen(input, temp, output, 16, aom_fft1d_16_avx2,
                 aom_transpose_float_sse2, aom_fft_unpack_2d_output_sse2, 8);
}

void aom_fft32x32_float_avx2(const float *input, float *temp, float *output) {
  aom_fft_2d_gen(input, temp, output, 32, aom_fft1d_32_avx2,
                 aom_transpose_float_sse2, aom_fft_unpack_2d_output_sse2, 8);
}

// Generate the 1d inverse transforms for float using _mm256
GEN_IFFT_8(static inline void, avx2, float, __m256, _mm256_load_ps,
           _mm256_store_ps, _mm256_set1_ps, _mm256_add_ps, _mm256_sub_ps,
           _mm256_mul_ps)
GEN_IFFT_16(static inline void, avx2, float, __m256, _mm256_load_ps,
            _mm256_store_ps, _mm256_set1_ps, _mm256_add_ps, _mm256_sub_ps,
            _mm256_mul_ps)
GEN_IFFT_32(static inline void, avx2, float, __m256, _mm256_load_ps,
            _mm256_store_ps, _mm256_set1_ps, _mm256_add_ps, _mm256_sub_ps,
            _mm256_mul_ps)

void aom_ifft8x8_float_avx2(const float *input, float *temp, float *output) {
  aom_ifft_2d_gen(input, temp, output, 8, aom_fft1d_8_float, aom_fft1d_8_avx2,
                  aom_ifft1d_8_avx2, aom_transpose_float_sse2, 8);
}

void aom_ifft16x16_float_avx2(const float *input, float *temp, float *output) {
  aom_ifft_2d_gen(input, temp, output, 16, aom_fft1d_16_float,
                  aom_fft1d_16_avx2, aom_ifft1d_16_avx2,
                  aom_transpose_float_sse2, 8);
}

void aom_ifft32x32_float_avx2(const float *input, float *temp, float *output) {
  aom_ifft_2d_gen(input, temp, output, 32, aom_fft1d_32_float,
                  aom_fft1d_32_avx2, aom_ifft1d_32_avx2,
                  aom_transpose_float_sse2, 8);
}
