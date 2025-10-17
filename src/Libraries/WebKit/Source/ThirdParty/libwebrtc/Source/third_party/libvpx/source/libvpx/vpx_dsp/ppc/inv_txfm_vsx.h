/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
#ifndef VPX_VPX_DSP_PPC_INV_TXFM_VSX_H_
#define VPX_VPX_DSP_PPC_INV_TXFM_VSX_H_

#include "vpx_dsp/ppc/types_vsx.h"

void vpx_round_store4x4_vsx(int16x8_t *in, int16x8_t *out, uint8_t *dest,
                            int stride);
void vpx_idct4_vsx(int16x8_t *in, int16x8_t *out);
void vp9_iadst4_vsx(int16x8_t *in, int16x8_t *out);

void vpx_round_store8x8_vsx(int16x8_t *in, uint8_t *dest, int stride);
void vpx_idct8_vsx(int16x8_t *in, int16x8_t *out);
void vp9_iadst8_vsx(int16x8_t *in, int16x8_t *out);

#define LOAD_INPUT16(load, source, offset, step, in) \
  in[0] = load(offset, source);                      \
  in[1] = load((step) + (offset), source);           \
  in[2] = load(2 * (step) + (offset), source);       \
  in[3] = load(3 * (step) + (offset), source);       \
  in[4] = load(4 * (step) + (offset), source);       \
  in[5] = load(5 * (step) + (offset), source);       \
  in[6] = load(6 * (step) + (offset), source);       \
  in[7] = load(7 * (step) + (offset), source);       \
  in[8] = load(8 * (step) + (offset), source);       \
  in[9] = load(9 * (step) + (offset), source);       \
  in[10] = load(10 * (step) + (offset), source);     \
  in[11] = load(11 * (step) + (offset), source);     \
  in[12] = load(12 * (step) + (offset), source);     \
  in[13] = load(13 * (step) + (offset), source);     \
  in[14] = load(14 * (step) + (offset), source);     \
  in[15] = load(15 * (step) + (offset), source);

void vpx_round_store16x16_vsx(int16x8_t *src0, int16x8_t *src1, uint8_t *dest,
                              int stride);
void vpx_idct16_vsx(int16x8_t *src0, int16x8_t *src1);
void vpx_iadst16_vsx(int16x8_t *src0, int16x8_t *src1);

#endif  // VPX_VPX_DSP_PPC_INV_TXFM_VSX_H_
