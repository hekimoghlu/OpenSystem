/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
#ifndef AOM_AV1_ENCODER_HASH_H_
#define AOM_AV1_ENCODER_HASH_H_

#include "config/aom_config.h"

#include "aom/aom_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _crc_calculator {
  uint32_t remainder;
  uint32_t trunc_poly;
  uint32_t bits;
  uint32_t table[256];
  uint32_t final_result_mask;
} CRC_CALCULATOR;

// Initialize the crc calculator. It must be executed at least once before
// calling av1_get_crc_value().
void av1_crc_calculator_init(CRC_CALCULATOR *p_crc_calculator, uint32_t bits,
                             uint32_t truncPoly);
uint32_t av1_get_crc_value(CRC_CALCULATOR *p_crc_calculator, uint8_t *p,
                           int length);

// CRC32C: POLY = 0x82f63b78;
typedef struct _CRC32C {
  /* Table for a quadword-at-a-time software crc. */
  uint32_t table[8][256];
} CRC32C;

// init table for software version crc32c
void av1_crc32c_calculator_init(CRC32C *p_crc32c);

#define AOM_BUFFER_SIZE_FOR_BLOCK_HASH (4096)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_HASH_H_
