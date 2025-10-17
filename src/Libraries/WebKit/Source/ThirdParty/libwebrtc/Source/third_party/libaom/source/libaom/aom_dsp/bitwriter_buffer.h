/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#ifndef AOM_AOM_DSP_BITWRITER_BUFFER_H_
#define AOM_AOM_DSP_BITWRITER_BUFFER_H_

#include "aom/aom_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct aom_write_bit_buffer {
  uint8_t *bit_buffer;
  uint32_t bit_offset;
};

int aom_wb_is_byte_aligned(const struct aom_write_bit_buffer *wb);

uint32_t aom_wb_bytes_written(const struct aom_write_bit_buffer *wb);

void aom_wb_write_bit(struct aom_write_bit_buffer *wb, int bit);

void aom_wb_write_literal(struct aom_write_bit_buffer *wb, int data, int bits);

void aom_wb_write_unsigned_literal(struct aom_write_bit_buffer *wb,
                                   uint32_t data, int bits);

void aom_wb_overwrite_literal(struct aom_write_bit_buffer *wb, int data,
                              int bits);

void aom_wb_write_inv_signed_literal(struct aom_write_bit_buffer *wb, int data,
                                     int bits);

void aom_wb_write_uvlc(struct aom_write_bit_buffer *wb, uint32_t v);

void aom_wb_write_signed_primitive_refsubexpfin(struct aom_write_bit_buffer *wb,
                                                uint16_t n, uint16_t k,
                                                int16_t ref, int16_t v);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_DSP_BITWRITER_BUFFER_H_
