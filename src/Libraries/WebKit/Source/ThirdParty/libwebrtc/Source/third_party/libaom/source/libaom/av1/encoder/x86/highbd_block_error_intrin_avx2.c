/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
#include <stdio.h>
#include "aom/aom_integer.h"
#include "av1/common/common.h"
#include "config/av1_rtcd.h"

int64_t av1_highbd_block_error_avx2(const tran_low_t *coeff,
                                    const tran_low_t *dqcoeff,
                                    intptr_t block_size, int64_t *ssz,
                                    int bps) {
  int i;
  int64_t temp1[8];
  int64_t error = 0, sqcoeff = 0;
  const int shift = 2 * (bps - 8);
  const int rounding = shift > 0 ? 1 << (shift - 1) : 0;

  for (i = 0; i < block_size; i += 16) {
    __m256i mm256_coeff = _mm256_loadu_si256((__m256i *)(coeff + i));
    __m256i mm256_coeff2 = _mm256_loadu_si256((__m256i *)(coeff + i + 8));
    __m256i mm256_dqcoeff = _mm256_loadu_si256((__m256i *)(dqcoeff + i));
    __m256i mm256_dqcoeff2 = _mm256_loadu_si256((__m256i *)(dqcoeff + i + 8));

    __m256i diff1 = _mm256_sub_epi32(mm256_coeff, mm256_dqcoeff);
    __m256i diff2 = _mm256_sub_epi32(mm256_coeff2, mm256_dqcoeff2);
    __m256i diff1h = _mm256_srli_epi64(diff1, 32);
    __m256i diff2h = _mm256_srli_epi64(diff2, 32);
    __m256i res = _mm256_mul_epi32(diff1, diff1);
    __m256i res1 = _mm256_mul_epi32(diff1h, diff1h);
    __m256i res2 = _mm256_mul_epi32(diff2, diff2);
    __m256i res3 = _mm256_mul_epi32(diff2h, diff2h);
    __m256i res_diff = _mm256_add_epi64(_mm256_add_epi64(res, res1),
                                        _mm256_add_epi64(res2, res3));
    __m256i mm256_coeffh = _mm256_srli_epi64(mm256_coeff, 32);
    __m256i mm256_coeffh2 = _mm256_srli_epi64(mm256_coeff2, 32);
    res = _mm256_mul_epi32(mm256_coeff, mm256_coeff);
    res1 = _mm256_mul_epi32(mm256_coeffh, mm256_coeffh);
    res2 = _mm256_mul_epi32(mm256_coeff2, mm256_coeff2);
    res3 = _mm256_mul_epi32(mm256_coeffh2, mm256_coeffh2);
    __m256i res_sqcoeff = _mm256_add_epi64(_mm256_add_epi64(res, res1),
                                           _mm256_add_epi64(res2, res3));
    _mm256_storeu_si256((__m256i *)temp1, res_diff);
    _mm256_storeu_si256((__m256i *)temp1 + 1, res_sqcoeff);

    error += temp1[0] + temp1[1] + temp1[2] + temp1[3];
    sqcoeff += temp1[4] + temp1[5] + temp1[6] + temp1[7];
  }
  assert(error >= 0 && sqcoeff >= 0);
  error = (error + rounding) >> shift;
  sqcoeff = (sqcoeff + rounding) >> shift;

  *ssz = sqcoeff;
  return error;
}
