/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#include "libyuv/basic_types.h"

#include "libyuv/compare_row.h"
#include "libyuv/row.h"

// This module is for GCC MSA
#if !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
#include "libyuv/macros_msa.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

uint32_t HammingDistance_MSA(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count) {
  uint32_t diff = 0u;
  int i;
  v16u8 src0, src1, src2, src3;
  v2i64 vec0 = {0}, vec1 = {0};

  for (i = 0; i < count; i += 32) {
    src0 = (v16u8)__msa_ld_b((v16i8*)src_a, 0);
    src1 = (v16u8)__msa_ld_b((v16i8*)src_a, 16);
    src2 = (v16u8)__msa_ld_b((v16i8*)src_b, 0);
    src3 = (v16u8)__msa_ld_b((v16i8*)src_b, 16);
    src0 ^= src2;
    src1 ^= src3;
    vec0 += __msa_pcnt_d((v2i64)src0);
    vec1 += __msa_pcnt_d((v2i64)src1);
    src_a += 32;
    src_b += 32;
  }

  vec0 += vec1;
  diff = (uint32_t)__msa_copy_u_w((v4i32)vec0, 0);
  diff += (uint32_t)__msa_copy_u_w((v4i32)vec0, 2);
  return diff;
}

uint32_t SumSquareError_MSA(const uint8_t* src_a,
                            const uint8_t* src_b,
                            int count) {
  uint32_t sse = 0u;
  int i;
  v16u8 src0, src1, src2, src3;
  v8i16 vec0, vec1, vec2, vec3;
  v4i32 reg0 = {0}, reg1 = {0}, reg2 = {0}, reg3 = {0};
  v2i64 tmp0;

  for (i = 0; i < count; i += 32) {
    src0 = (v16u8)__msa_ld_b((v16i8*)src_a, 0);
    src1 = (v16u8)__msa_ld_b((v16i8*)src_a, 16);
    src2 = (v16u8)__msa_ld_b((v16i8*)src_b, 0);
    src3 = (v16u8)__msa_ld_b((v16i8*)src_b, 16);
    vec0 = (v8i16)__msa_ilvr_b((v16i8)src2, (v16i8)src0);
    vec1 = (v8i16)__msa_ilvl_b((v16i8)src2, (v16i8)src0);
    vec2 = (v8i16)__msa_ilvr_b((v16i8)src3, (v16i8)src1);
    vec3 = (v8i16)__msa_ilvl_b((v16i8)src3, (v16i8)src1);
    vec0 = __msa_hsub_u_h((v16u8)vec0, (v16u8)vec0);
    vec1 = __msa_hsub_u_h((v16u8)vec1, (v16u8)vec1);
    vec2 = __msa_hsub_u_h((v16u8)vec2, (v16u8)vec2);
    vec3 = __msa_hsub_u_h((v16u8)vec3, (v16u8)vec3);
    reg0 = __msa_dpadd_s_w(reg0, vec0, vec0);
    reg1 = __msa_dpadd_s_w(reg1, vec1, vec1);
    reg2 = __msa_dpadd_s_w(reg2, vec2, vec2);
    reg3 = __msa_dpadd_s_w(reg3, vec3, vec3);
    src_a += 32;
    src_b += 32;
  }

  reg0 += reg1;
  reg2 += reg3;
  reg0 += reg2;
  tmp0 = __msa_hadd_s_d(reg0, reg0);
  sse = (uint32_t)__msa_copy_u_w((v4i32)tmp0, 0);
  sse += (uint32_t)__msa_copy_u_w((v4i32)tmp0, 2);
  return sse;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
