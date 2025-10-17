/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
#ifndef AOM_AV1_ENCODER_ARM_TXFM_NEON_H_
#define AOM_AV1_ENCODER_ARM_TXFM_NEON_H_

#include <stdint.h>

static inline void ud_adjust_input_and_stride(int ud_flip,
                                              const int16_t **input,
                                              int *stride, int out_size) {
  if (ud_flip) {
    *input = *input + (out_size - 1) * *stride;
    *stride = -*stride;
  }
}

#endif  // AOM_AV1_ENCODER_ARM_TXFM_NEON_H_
