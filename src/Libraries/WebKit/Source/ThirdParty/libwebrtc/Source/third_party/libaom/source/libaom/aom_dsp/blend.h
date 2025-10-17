/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#ifndef AOM_AOM_DSP_BLEND_H_
#define AOM_AOM_DSP_BLEND_H_

#include "aom_ports/mem.h"

// Various blending functions and macros.
// See also the aom_blend_* functions in aom_dsp_rtcd.h

// Alpha blending with alpha values from the range [0, 64], where 64
// means use the first input and 0 means use the second input.

#define AOM_BLEND_A64_ROUND_BITS 6
#define AOM_BLEND_A64_MAX_ALPHA (1 << AOM_BLEND_A64_ROUND_BITS)  // 64

#define AOM_BLEND_A64(a, v0, v1)                                          \
  ROUND_POWER_OF_TWO((a) * (v0) + (AOM_BLEND_A64_MAX_ALPHA - (a)) * (v1), \
                     AOM_BLEND_A64_ROUND_BITS)

// Alpha blending with alpha values from the range [0, 256], where 256
// means use the first input and 0 means use the second input.
#define AOM_BLEND_A256_ROUND_BITS 8
#define AOM_BLEND_A256_MAX_ALPHA (1 << AOM_BLEND_A256_ROUND_BITS)  // 256

#define AOM_BLEND_A256(a, v0, v1)                                          \
  ROUND_POWER_OF_TWO((a) * (v0) + (AOM_BLEND_A256_MAX_ALPHA - (a)) * (v1), \
                     AOM_BLEND_A256_ROUND_BITS)

// Blending by averaging.
#define AOM_BLEND_AVG(v0, v1) ROUND_POWER_OF_TWO((v0) + (v1), 1)

#define DIFF_FACTOR_LOG2 4
#define DIFF_FACTOR (1 << DIFF_FACTOR_LOG2)

#endif  // AOM_AOM_DSP_BLEND_H_
