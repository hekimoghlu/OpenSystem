/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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
#ifndef VPX_VP8_ENCODER_ENCODEMB_H_
#define VPX_VP8_ENCODER_ENCODEMB_H_

#include "onyx_int.h"

#ifdef __cplusplus
extern "C" {
#endif
void vp8_encode_inter16x16(MACROBLOCK *x);

void vp8_subtract_b(BLOCK *be, BLOCKD *bd, int pitch);
void vp8_subtract_mbuv(short *diff, unsigned char *usrc, unsigned char *vsrc,
                       int src_stride, unsigned char *upred,
                       unsigned char *vpred, int pred_stride);
void vp8_subtract_mby(short *diff, unsigned char *src, int src_stride,
                      unsigned char *pred, int pred_stride);

void vp8_build_dcblock(MACROBLOCK *b);
void vp8_transform_mb(MACROBLOCK *mb);
void vp8_transform_mbuv(MACROBLOCK *x);
void vp8_transform_intra_mby(MACROBLOCK *x);

void vp8_optimize_mby(MACROBLOCK *x);
void vp8_optimize_mbuv(MACROBLOCK *x);
void vp8_encode_inter16x16y(MACROBLOCK *x);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_ENCODEMB_H_
