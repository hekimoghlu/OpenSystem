/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#ifndef VPX_VP8_ENCODER_ENCODEINTRA_H_
#define VPX_VP8_ENCODER_ENCODEINTRA_H_
#include "onyx_int.h"

#ifdef __cplusplus
extern "C" {
#endif

int vp8_encode_intra(MACROBLOCK *x, int use_dc_pred);
void vp8_encode_intra16x16mby(MACROBLOCK *x);
void vp8_encode_intra16x16mbuv(MACROBLOCK *x);
void vp8_encode_intra4x4mby(MACROBLOCK *mb);
void vp8_encode_intra4x4block(MACROBLOCK *x, int ib);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_ENCODEINTRA_H_
