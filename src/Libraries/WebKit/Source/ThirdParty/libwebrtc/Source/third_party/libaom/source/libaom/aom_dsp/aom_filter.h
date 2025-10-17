/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 18, 2024.
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
#ifndef AOM_AOM_DSP_AOM_FILTER_H_
#define AOM_AOM_DSP_AOM_FILTER_H_

#include "aom/aom_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FILTER_BITS 7

#define SUBPEL_BITS 4
#define SUBPEL_MASK ((1 << SUBPEL_BITS) - 1)
#define SUBPEL_SHIFTS (1 << SUBPEL_BITS)
#define SUBPEL_TAPS 8

#define SCALE_SUBPEL_BITS 10
#define SCALE_SUBPEL_SHIFTS (1 << SCALE_SUBPEL_BITS)
#define SCALE_SUBPEL_MASK (SCALE_SUBPEL_SHIFTS - 1)
#define SCALE_EXTRA_BITS (SCALE_SUBPEL_BITS - SUBPEL_BITS)
#define SCALE_EXTRA_OFF ((1 << SCALE_EXTRA_BITS) / 2)

#define RS_SUBPEL_BITS 6
#define RS_SUBPEL_MASK ((1 << RS_SUBPEL_BITS) - 1)
#define RS_SCALE_SUBPEL_BITS 14
#define RS_SCALE_SUBPEL_MASK ((1 << RS_SCALE_SUBPEL_BITS) - 1)
#define RS_SCALE_EXTRA_BITS (RS_SCALE_SUBPEL_BITS - RS_SUBPEL_BITS)
#define RS_SCALE_EXTRA_OFF (1 << (RS_SCALE_EXTRA_BITS - 1))

typedef int16_t InterpKernel[SUBPEL_TAPS];

#define BIL_SUBPEL_BITS 3
#define BIL_SUBPEL_SHIFTS (1 << BIL_SUBPEL_BITS)

// 2 tap bilinear filters
static const uint8_t bilinear_filters_2t[BIL_SUBPEL_SHIFTS][2] = {
  { 128, 0 }, { 112, 16 }, { 96, 32 }, { 80, 48 },
  { 64, 64 }, { 48, 80 },  { 32, 96 }, { 16, 112 },
};

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_DSP_AOM_FILTER_H_
