/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
#ifndef AOM_AV1_ENCODER_AV1_FWD_TXFM1D_H_
#define AOM_AV1_ENCODER_AV1_FWD_TXFM1D_H_

#include "av1/common/av1_txfm.h"

#ifdef __cplusplus
extern "C" {
#endif

void av1_fdct4(const int32_t *input, int32_t *output, int8_t cos_bit,
               const int8_t *stage_range);
void av1_fdct8(const int32_t *input, int32_t *output, int8_t cos_bit,
               const int8_t *stage_range);
void av1_fdct16(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *stage_range);
void av1_fdct32(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *stage_range);
void av1_fdct64(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *stage_range);
void av1_fadst4(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *stage_range);
void av1_fadst8(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *stage_range);
void av1_fadst16(const int32_t *input, int32_t *output, int8_t cos_bit,
                 const int8_t *stage_range);
void av1_fidentity4_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                      const int8_t *stage_range);
void av1_fidentity8_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                      const int8_t *stage_range);
void av1_fidentity16_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                       const int8_t *stage_range);
void av1_fidentity32_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                       const int8_t *stage_range);
#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_ENCODER_AV1_FWD_TXFM1D_H_
