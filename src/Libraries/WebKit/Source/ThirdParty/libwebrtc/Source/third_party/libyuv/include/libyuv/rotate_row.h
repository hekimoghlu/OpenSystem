/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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
#ifndef INCLUDE_LIBYUV_ROTATE_ROW_H_
#define INCLUDE_LIBYUV_ROTATE_ROW_H_

#include "libyuv/basic_types.h"
#include "libyuv/cpu_support.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// The following are available for Visual C 32 bit:
#if !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86) && defined(_MSC_VER) && \
    !defined(__clang__)
#define HAS_TRANSPOSEWX8_SSSE3
#define HAS_TRANSPOSEUVWX8_SSE2
#endif

// The following are available for GCC 32 or 64 bit:
#if !defined(LIBYUV_DISABLE_X86) && (defined(__i386__) || defined(__x86_64__))
#define HAS_TRANSPOSEWX8_SSSE3
#define HAS_TRANSPOSE4X4_32_SSE2
#define HAS_TRANSPOSE4X4_32_AVX2
#endif

// The following are available for 64 bit GCC:
#if !defined(LIBYUV_DISABLE_X86) && defined(__x86_64__)
#define HAS_TRANSPOSEWX8_FAST_SSSE3
#define HAS_TRANSPOSEUVWX8_SSE2
#endif

#if !defined(LIBYUV_DISABLE_NEON) && \
    (defined(__ARM_NEON__) || defined(LIBYUV_NEON) || defined(__aarch64__))
#if defined(__aarch64__)
#define HAS_TRANSPOSEWX16_NEON
#else
#define HAS_TRANSPOSEWX8_NEON
#endif
#define HAS_TRANSPOSEUVWX8_NEON
#define HAS_TRANSPOSE4X4_32_NEON
#endif

#if !defined(LIBYUV_DISABLE_SME) && defined(CLANG_HAS_SME) && \
    defined(__aarch64__)
#define HAS_TRANSPOSEWXH_SME
#define HAS_TRANSPOSEUVWXH_SME
#endif

#if !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
#define HAS_TRANSPOSEWX16_MSA
#define HAS_TRANSPOSEUVWX16_MSA
#endif

#if !defined(LIBYUV_DISABLE_LSX) && defined(__loongarch_sx)
#define HAS_TRANSPOSEWX16_LSX
#define HAS_TRANSPOSEUVWX16_LSX
#endif

void TransposeWxH_C(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width,
                    int height);

void TransposeWx8_C(const uint8_t* src,
                    int src_stride,
                    uint8_t* dst,
                    int dst_stride,
                    int width);
void TransposeWx16_C(const uint8_t* src,
                     int src_stride,
                     uint8_t* dst,
                     int dst_stride,
                     int width);
void TransposeWx8_NEON(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst,
                       int dst_stride,
                       int width);
void TransposeWx16_NEON(const uint8_t* src,
                        int src_stride,
                        uint8_t* dst,
                        int dst_stride,
                        int width);
void TransposeWxH_SME(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst,
                      int dst_stride,
                      int width,
                      int height);
void TransposeWx8_SSSE3(const uint8_t* src,
                        int src_stride,
                        uint8_t* dst,
                        int dst_stride,
                        int width);
void TransposeWx8_Fast_SSSE3(const uint8_t* src,
                             int src_stride,
                             uint8_t* dst,
                             int dst_stride,
                             int width);
void TransposeWx16_MSA(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst,
                       int dst_stride,
                       int width);
void TransposeWx16_LSX(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst,
                       int dst_stride,
                       int width);

void TransposeWx8_Any_NEON(const uint8_t* src,
                           int src_stride,
                           uint8_t* dst,
                           int dst_stride,
                           int width);
void TransposeWx16_Any_NEON(const uint8_t* src,
                            int src_stride,
                            uint8_t* dst,
                            int dst_stride,
                            int width);
void TransposeWx8_Any_SSSE3(const uint8_t* src,
                            int src_stride,
                            uint8_t* dst,
                            int dst_stride,
                            int width);
void TransposeWx8_Fast_Any_SSSE3(const uint8_t* src,
                                 int src_stride,
                                 uint8_t* dst,
                                 int dst_stride,
                                 int width);
void TransposeWx16_Any_MSA(const uint8_t* src,
                           int src_stride,
                           uint8_t* dst,
                           int dst_stride,
                           int width);
void TransposeWx16_Any_LSX(const uint8_t* src,
                           int src_stride,
                           uint8_t* dst,
                           int dst_stride,
                           int width);

void TransposeUVWxH_C(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width,
                      int height);

void TransposeUVWx8_C(const uint8_t* src,
                      int src_stride,
                      uint8_t* dst_a,
                      int dst_stride_a,
                      uint8_t* dst_b,
                      int dst_stride_b,
                      int width);
void TransposeUVWx16_C(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst_a,
                       int dst_stride_a,
                       uint8_t* dst_b,
                       int dst_stride_b,
                       int width);
void TransposeUVWx8_SSE2(const uint8_t* src,
                         int src_stride,
                         uint8_t* dst_a,
                         int dst_stride_a,
                         uint8_t* dst_b,
                         int dst_stride_b,
                         int width);
void TransposeUVWx8_NEON(const uint8_t* src,
                         int src_stride,
                         uint8_t* dst_a,
                         int dst_stride_a,
                         uint8_t* dst_b,
                         int dst_stride_b,
                         int width);
void TransposeUVWxH_SME(const uint8_t* src,
                        int src_stride,
                        uint8_t* dst_a,
                        int dst_stride_a,
                        uint8_t* dst_b,
                        int dst_stride_b,
                        int width,
                        int height);
void TransposeUVWx16_MSA(const uint8_t* src,
                         int src_stride,
                         uint8_t* dst_a,
                         int dst_stride_a,
                         uint8_t* dst_b,
                         int dst_stride_b,
                         int width);
void TransposeUVWx16_LSX(const uint8_t* src,
                         int src_stride,
                         uint8_t* dst_a,
                         int dst_stride_a,
                         uint8_t* dst_b,
                         int dst_stride_b,
                         int width);

void TransposeUVWx8_Any_SSE2(const uint8_t* src,
                             int src_stride,
                             uint8_t* dst_a,
                             int dst_stride_a,
                             uint8_t* dst_b,
                             int dst_stride_b,
                             int width);
void TransposeUVWx8_Any_NEON(const uint8_t* src,
                             int src_stride,
                             uint8_t* dst_a,
                             int dst_stride_a,
                             uint8_t* dst_b,
                             int dst_stride_b,
                             int width);
void TransposeUVWx16_Any_MSA(const uint8_t* src,
                             int src_stride,
                             uint8_t* dst_a,
                             int dst_stride_a,
                             uint8_t* dst_b,
                             int dst_stride_b,
                             int width);
void TransposeUVWx16_Any_LSX(const uint8_t* src,
                             int src_stride,
                             uint8_t* dst_a,
                             int dst_stride_a,
                             uint8_t* dst_b,
                             int dst_stride_b,
                             int width);
void TransposeWxH_16_C(const uint16_t* src,
                       int src_stride,
                       uint16_t* dst,
                       int dst_stride,
                       int width,
                       int height);

void TransposeWx8_16_C(const uint16_t* src,
                       int src_stride,
                       uint16_t* dst,
                       int dst_stride,
                       int width);
void TransposeWx1_16_C(const uint16_t* src,
                       int src_stride,
                       uint16_t* dst,
                       int dst_stride,
                       int width);

// Transpose 32 bit values (ARGB)
void Transpose4x4_32_NEON(const uint8_t* src,
                          int src_stride,
                          uint8_t* dst,
                          int dst_stride,
                          int width);

void Transpose4x4_32_SSE2(const uint8_t* src,
                          int src_stride,
                          uint8_t* dst,
                          int dst_stride,
                          int width);

void Transpose4x4_32_AVX2(const uint8_t* src,
                          int src_stride,
                          uint8_t* dst,
                          int dst_stride,
                          int width);

void Transpose4x4_32_C(const uint8_t* src,
                       int src_stride,
                       uint8_t* dst,
                       int dst_stride,
                       int width);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_ROTATE_ROW_H_
