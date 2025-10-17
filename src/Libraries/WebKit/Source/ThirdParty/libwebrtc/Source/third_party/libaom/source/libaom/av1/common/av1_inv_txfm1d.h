/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#ifndef AOM_AV1_COMMON_AV1_INV_TXFM1D_H_
#define AOM_AV1_COMMON_AV1_INV_TXFM1D_H_

#include "av1/common/av1_txfm.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline int32_t clamp_value(int32_t value, int8_t bit) {
  if (bit <= 0) return value;  // Do nothing for invalid clamp bit.
  const int64_t max_value = (1LL << (bit - 1)) - 1;
  const int64_t min_value = -(1LL << (bit - 1));
  return (int32_t)clamp64(value, min_value, max_value);
}

static inline void clamp_buf(int32_t *buf, int32_t size, int8_t bit) {
  for (int i = 0; i < size; ++i) buf[i] = clamp_value(buf[i], bit);
}

void av1_idct4(const int32_t *input, int32_t *output, int8_t cos_bit,
               const int8_t *stage_range);
void av1_idct8(const int32_t *input, int32_t *output, int8_t cos_bit,
               const int8_t *stage_range);
void av1_idct16(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *stage_range);
void av1_idct32(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *stage_range);
void av1_idct64(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *stage_range);
void av1_iadst4(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *stage_range);
void av1_iadst8(const int32_t *input, int32_t *output, int8_t cos_bit,
                const int8_t *stage_range);
void av1_iadst16(const int32_t *input, int32_t *output, int8_t cos_bit,
                 const int8_t *stage_range);
void av1_iidentity4_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                      const int8_t *stage_range);
void av1_iidentity8_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                      const int8_t *stage_range);
void av1_iidentity16_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                       const int8_t *stage_range);
void av1_iidentity32_c(const int32_t *input, int32_t *output, int8_t cos_bit,
                       const int8_t *stage_range);

#ifdef __cplusplus
}
#endif

#endif  // AOM_AV1_COMMON_AV1_INV_TXFM1D_H_
