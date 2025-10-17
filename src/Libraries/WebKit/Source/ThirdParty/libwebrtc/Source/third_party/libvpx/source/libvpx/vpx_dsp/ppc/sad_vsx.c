/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include <stdlib.h>

#include "./vpx_dsp_rtcd.h"

#include "vpx_dsp/ppc/types_vsx.h"

#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

#define PROCESS16(offset)      \
  v_a = vec_vsx_ld(offset, a); \
  v_b = vec_vsx_ld(offset, b); \
  v_abs = vec_absd(v_a, v_b);  \
  v_sad = vec_sum4s(v_abs, v_sad);

#define SAD8(height)                                                     \
  unsigned int vpx_sad8x##height##_vsx(const uint8_t *a, int a_stride,   \
                                       const uint8_t *b, int b_stride) { \
    int y = 0;                                                           \
    uint8x16_t v_a, v_b, v_abs;                                          \
    uint32x4_t v_sad = vec_zeros_u32;                                    \
                                                                         \
    do {                                                                 \
      PROCESS16(0)                                                       \
                                                                         \
      a += a_stride;                                                     \
      b += b_stride;                                                     \
      y++;                                                               \
    } while (y < height);                                                \
                                                                         \
    return v_sad[1] + v_sad[0];                                          \
  }

#define SAD16(height)                                                     \
  unsigned int vpx_sad16x##height##_vsx(const uint8_t *a, int a_stride,   \
                                        const uint8_t *b, int b_stride) { \
    int y = 0;                                                            \
    uint8x16_t v_a, v_b, v_abs;                                           \
    uint32x4_t v_sad = vec_zeros_u32;                                     \
                                                                          \
    do {                                                                  \
      PROCESS16(0);                                                       \
                                                                          \
      a += a_stride;                                                      \
      b += b_stride;                                                      \
      y++;                                                                \
    } while (y < height);                                                 \
                                                                          \
    return v_sad[3] + v_sad[2] + v_sad[1] + v_sad[0];                     \
  }

#define SAD32(height)                                                     \
  unsigned int vpx_sad32x##height##_vsx(const uint8_t *a, int a_stride,   \
                                        const uint8_t *b, int b_stride) { \
    int y = 0;                                                            \
    uint8x16_t v_a, v_b, v_abs;                                           \
    uint32x4_t v_sad = vec_zeros_u32;                                     \
                                                                          \
    do {                                                                  \
      PROCESS16(0);                                                       \
      PROCESS16(16);                                                      \
                                                                          \
      a += a_stride;                                                      \
      b += b_stride;                                                      \
      y++;                                                                \
    } while (y < height);                                                 \
                                                                          \
    return v_sad[3] + v_sad[2] + v_sad[1] + v_sad[0];                     \
  }

#define SAD64(height)                                                     \
  unsigned int vpx_sad64x##height##_vsx(const uint8_t *a, int a_stride,   \
                                        const uint8_t *b, int b_stride) { \
    int y = 0;                                                            \
    uint8x16_t v_a, v_b, v_abs;                                           \
    uint32x4_t v_sad = vec_zeros_u32;                                     \
                                                                          \
    do {                                                                  \
      PROCESS16(0);                                                       \
      PROCESS16(16);                                                      \
      PROCESS16(32);                                                      \
      PROCESS16(48);                                                      \
                                                                          \
      a += a_stride;                                                      \
      b += b_stride;                                                      \
      y++;                                                                \
    } while (y < height);                                                 \
                                                                          \
    return v_sad[3] + v_sad[2] + v_sad[1] + v_sad[0];                     \
  }

SAD8(4);
SAD8(8);
SAD8(16);
SAD16(8);
SAD16(16);
SAD16(32);
SAD32(16);
SAD32(32);
SAD32(64);
SAD64(32);
SAD64(64);

#define SAD16AVG(height)                                                      \
  unsigned int vpx_sad16x##height##_avg_vsx(                                  \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      const uint8_t *second_pred) {                                           \
    DECLARE_ALIGNED(16, uint8_t, comp_pred[16 * (height)]);                   \
    vpx_comp_avg_pred_vsx(comp_pred, second_pred, 16, height, ref,            \
                          ref_stride);                                        \
                                                                              \
    return vpx_sad16x##height##_vsx(src, src_stride, comp_pred, 16);          \
  }

#define SAD32AVG(height)                                                      \
  unsigned int vpx_sad32x##height##_avg_vsx(                                  \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      const uint8_t *second_pred) {                                           \
    DECLARE_ALIGNED(32, uint8_t, comp_pred[32 * (height)]);                   \
    vpx_comp_avg_pred_vsx(comp_pred, second_pred, 32, height, ref,            \
                          ref_stride);                                        \
                                                                              \
    return vpx_sad32x##height##_vsx(src, src_stride, comp_pred, 32);          \
  }

#define SAD64AVG(height)                                                      \
  unsigned int vpx_sad64x##height##_avg_vsx(                                  \
      const uint8_t *src, int src_stride, const uint8_t *ref, int ref_stride, \
      const uint8_t *second_pred) {                                           \
    DECLARE_ALIGNED(64, uint8_t, comp_pred[64 * (height)]);                   \
    vpx_comp_avg_pred_vsx(comp_pred, second_pred, 64, height, ref,            \
                          ref_stride);                                        \
    return vpx_sad64x##height##_vsx(src, src_stride, comp_pred, 64);          \
  }

SAD16AVG(8);
SAD16AVG(16);
SAD16AVG(32);
SAD32AVG(16);
SAD32AVG(32);
SAD32AVG(64);
SAD64AVG(32);
SAD64AVG(64);

#define PROCESS16_4D(offset, ref, v_h, v_l) \
  v_b = vec_vsx_ld(offset, ref);            \
  v_bh = unpack_to_s16_h(v_b);              \
  v_bl = unpack_to_s16_l(v_b);              \
  v_subh = vec_sub(v_h, v_bh);              \
  v_subl = vec_sub(v_l, v_bl);              \
  v_absh = vec_abs(v_subh);                 \
  v_absl = vec_abs(v_subl);                 \
  v_sad = vec_sum4s(v_absh, v_sad);         \
  v_sad = vec_sum4s(v_absl, v_sad);

#define UNPACK_SRC(offset, srcv_h, srcv_l) \
  v_a = vec_vsx_ld(offset, src);           \
  srcv_h = unpack_to_s16_h(v_a);           \
  srcv_l = unpack_to_s16_l(v_a);

#define SAD16_4D(height)                                                  \
  void vpx_sad16x##height##x4d_vsx(const uint8_t *src, int src_stride,    \
                                   const uint8_t *const ref_array[],      \
                                   int ref_stride, uint32_t *sad_array) { \
    int i;                                                                \
    int y;                                                                \
    unsigned int sad[4];                                                  \
    uint8x16_t v_a, v_b;                                                  \
    int16x8_t v_ah, v_al, v_bh, v_bl, v_absh, v_absl, v_subh, v_subl;     \
                                                                          \
    for (i = 0; i < 4; i++) sad_array[i] = 0;                             \
                                                                          \
    for (y = 0; y < height; y++) {                                        \
      UNPACK_SRC(y *src_stride, v_ah, v_al);                              \
      for (i = 0; i < 4; i++) {                                           \
        int32x4_t v_sad = vec_splat_s32(0);                               \
        PROCESS16_4D(y *ref_stride, ref_array[i], v_ah, v_al);            \
                                                                          \
        vec_vsx_st((uint32x4_t)v_sad, 0, sad);                            \
        sad_array[i] += (sad[3] + sad[2] + sad[1] + sad[0]);              \
      }                                                                   \
    }                                                                     \
  }

#define SAD32_4D(height)                                                  \
  void vpx_sad32x##height##x4d_vsx(const uint8_t *src, int src_stride,    \
                                   const uint8_t *const ref_array[],      \
                                   int ref_stride, uint32_t *sad_array) { \
    int i;                                                                \
    int y;                                                                \
    unsigned int sad[4];                                                  \
    uint8x16_t v_a, v_b;                                                  \
    int16x8_t v_ah1, v_al1, v_ah2, v_al2, v_bh, v_bl;                     \
    int16x8_t v_absh, v_absl, v_subh, v_subl;                             \
                                                                          \
    for (i = 0; i < 4; i++) sad_array[i] = 0;                             \
                                                                          \
    for (y = 0; y < height; y++) {                                        \
      UNPACK_SRC(y *src_stride, v_ah1, v_al1);                            \
      UNPACK_SRC(y *src_stride + 16, v_ah2, v_al2);                       \
      for (i = 0; i < 4; i++) {                                           \
        int32x4_t v_sad = vec_splat_s32(0);                               \
        PROCESS16_4D(y *ref_stride, ref_array[i], v_ah1, v_al1);          \
        PROCESS16_4D(y *ref_stride + 16, ref_array[i], v_ah2, v_al2);     \
                                                                          \
        vec_vsx_st((uint32x4_t)v_sad, 0, sad);                            \
        sad_array[i] += (sad[3] + sad[2] + sad[1] + sad[0]);              \
      }                                                                   \
    }                                                                     \
  }

#define SAD64_4D(height)                                                  \
  void vpx_sad64x##height##x4d_vsx(const uint8_t *src, int src_stride,    \
                                   const uint8_t *const ref_array[],      \
                                   int ref_stride, uint32_t *sad_array) { \
    int i;                                                                \
    int y;                                                                \
    unsigned int sad[4];                                                  \
    uint8x16_t v_a, v_b;                                                  \
    int16x8_t v_ah1, v_al1, v_ah2, v_al2, v_bh, v_bl;                     \
    int16x8_t v_ah3, v_al3, v_ah4, v_al4;                                 \
    int16x8_t v_absh, v_absl, v_subh, v_subl;                             \
                                                                          \
    for (i = 0; i < 4; i++) sad_array[i] = 0;                             \
                                                                          \
    for (y = 0; y < height; y++) {                                        \
      UNPACK_SRC(y *src_stride, v_ah1, v_al1);                            \
      UNPACK_SRC(y *src_stride + 16, v_ah2, v_al2);                       \
      UNPACK_SRC(y *src_stride + 32, v_ah3, v_al3);                       \
      UNPACK_SRC(y *src_stride + 48, v_ah4, v_al4);                       \
      for (i = 0; i < 4; i++) {                                           \
        int32x4_t v_sad = vec_splat_s32(0);                               \
        PROCESS16_4D(y *ref_stride, ref_array[i], v_ah1, v_al1);          \
        PROCESS16_4D(y *ref_stride + 16, ref_array[i], v_ah2, v_al2);     \
        PROCESS16_4D(y *ref_stride + 32, ref_array[i], v_ah3, v_al3);     \
        PROCESS16_4D(y *ref_stride + 48, ref_array[i], v_ah4, v_al4);     \
                                                                          \
        vec_vsx_st((uint32x4_t)v_sad, 0, sad);                            \
        sad_array[i] += (sad[3] + sad[2] + sad[1] + sad[0]);              \
      }                                                                   \
    }                                                                     \
  }

SAD16_4D(8);
SAD16_4D(16);
SAD16_4D(32);
SAD32_4D(16);
SAD32_4D(32);
SAD32_4D(64);
SAD64_4D(32);
SAD64_4D(64);
