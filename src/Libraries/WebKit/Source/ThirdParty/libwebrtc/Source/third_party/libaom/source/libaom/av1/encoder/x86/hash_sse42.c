/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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
#include <stdint.h>
#include <smmintrin.h>

#include "config/av1_rtcd.h"

// Byte-boundary alignment issues
#define ALIGN_SIZE 8
#define ALIGN_MASK (ALIGN_SIZE - 1)

#define CALC_CRC(op, crc, type, buf, len) \
  while ((len) >= sizeof(type)) {         \
    (crc) = op((crc), *(type *)(buf));    \
    (len) -= sizeof(type);                \
    buf += sizeof(type);                  \
  }

/**
 * Calculates 32-bit CRC for the input buffer
 * polynomial is 0x11EDC6F41
 * @return A 32-bit unsigned integer representing the CRC
 */
uint32_t av1_get_crc32c_value_sse4_2(void *crc_calculator, uint8_t *p,
                                     size_t len) {
  (void)crc_calculator;
  const uint8_t *buf = p;
  uint32_t crc = 0xFFFFFFFF;

  // Align the input to the word boundary
  for (; (len > 0) && ((intptr_t)buf & ALIGN_MASK); len--, buf++) {
    crc = _mm_crc32_u8(crc, *buf);
  }

#ifdef __x86_64__
  uint64_t crc64 = crc;
  CALC_CRC(_mm_crc32_u64, crc64, uint64_t, buf, len)
  crc = (uint32_t)crc64;
#endif
  CALC_CRC(_mm_crc32_u32, crc, uint32_t, buf, len)
  CALC_CRC(_mm_crc32_u16, crc, uint16_t, buf, len)
  CALC_CRC(_mm_crc32_u8, crc, uint8_t, buf, len)
  return (crc ^ 0xFFFFFFFF);
}
