/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
/*
 * This file contains implementations of the functions
 * WebRtcSpl_ScaleAndAddVectorsWithRound_mips()
 */

#include "common_audio/signal_processing/include/signal_processing_library.h"

int WebRtcSpl_ScaleAndAddVectorsWithRound_mips(const int16_t* in_vector1,
                                               int16_t in_vector1_scale,
                                               const int16_t* in_vector2,
                                               int16_t in_vector2_scale,
                                               int right_shifts,
                                               int16_t* out_vector,
                                               size_t length) {
  int16_t r0 = 0, r1 = 0;
  int16_t *in1 = (int16_t*)in_vector1;
  int16_t *in2 = (int16_t*)in_vector2;
  int16_t *out = out_vector;
  size_t i = 0;
  int value32 = 0;

  if (in_vector1 == NULL || in_vector2 == NULL || out_vector == NULL ||
      length == 0 || right_shifts < 0) {
    return -1;
  }
  for (i = 0; i < length; i++) {
    __asm __volatile (
      "lh         %[r0],          0(%[in1])                               \n\t"
      "lh         %[r1],          0(%[in2])                               \n\t"
      "mult       %[r0],          %[in_vector1_scale]                     \n\t"
      "madd       %[r1],          %[in_vector2_scale]                     \n\t"
      "extrv_r.w  %[value32],     $ac0,               %[right_shifts]     \n\t"
      "addiu      %[in1],         %[in1],             2                   \n\t"
      "addiu      %[in2],         %[in2],             2                   \n\t"
      "sh         %[value32],     0(%[out])                               \n\t"
      "addiu      %[out],         %[out],             2                   \n\t"
      : [value32] "=&r" (value32), [out] "+r" (out), [in1] "+r" (in1),
        [in2] "+r" (in2), [r0] "=&r" (r0), [r1] "=&r" (r1)
      : [in_vector1_scale] "r" (in_vector1_scale),
        [in_vector2_scale] "r" (in_vector2_scale),
        [right_shifts] "r" (right_shifts)
      : "hi", "lo", "memory"
    );
  }
  return 0;
}
