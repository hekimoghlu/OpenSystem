/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <arm_acle.h>
#endif

#include <stddef.h>
#include <stdint.h>

#include "config/aom_config.h"
#include "config/av1_rtcd.h"

#define CRC_LOOP(op, crc, type, buf, len) \
  while ((len) >= sizeof(type)) {         \
    (crc) = op((crc), *(type *)(buf));    \
    (len) -= sizeof(type);                \
    buf += sizeof(type);                  \
  }

#define CRC_SINGLE(op, crc, type, buf, len) \
  if ((len) >= sizeof(type)) {              \
    (crc) = op((crc), *(type *)(buf));      \
    (len) -= sizeof(type);                  \
    buf += sizeof(type);                    \
  }

/* Return 32-bit CRC for the input buffer.
 * Polynomial is 0x1EDC6F41.
 */

uint32_t av1_get_crc32c_value_arm_crc32(void *crc_calculator, uint8_t *p,
                                        size_t len) {
  (void)crc_calculator;
  const uint8_t *buf = p;
  uint32_t crc = 0xFFFFFFFF;

#if !AOM_ARCH_AARCH64
  // Align input to 8-byte boundary (only necessary for 32-bit builds.)
  while (len && ((uintptr_t)buf & 7)) {
    crc = __crc32cb(crc, *buf++);
    len--;
  }
#endif

  CRC_LOOP(__crc32cd, crc, uint64_t, buf, len)
  CRC_SINGLE(__crc32cw, crc, uint32_t, buf, len)
  CRC_SINGLE(__crc32ch, crc, uint16_t, buf, len)
  CRC_SINGLE(__crc32cb, crc, uint8_t, buf, len)

  return ~crc;
}
