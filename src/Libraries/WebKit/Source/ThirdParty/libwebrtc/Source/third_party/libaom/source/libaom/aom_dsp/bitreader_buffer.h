/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#ifndef AOM_AOM_DSP_BITREADER_BUFFER_H_
#define AOM_AOM_DSP_BITREADER_BUFFER_H_

#include <limits.h>

#include "aom/aom_integer.h"
#include "config/aom_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*aom_rb_error_handler)(void *data);

struct aom_read_bit_buffer {
  const uint8_t *bit_buffer;
  const uint8_t *bit_buffer_end;
  uint32_t bit_offset;

  void *error_handler_data;
  aom_rb_error_handler error_handler;
};

size_t aom_rb_bytes_read(const struct aom_read_bit_buffer *rb);

int aom_rb_read_bit(struct aom_read_bit_buffer *rb);

int aom_rb_read_literal(struct aom_read_bit_buffer *rb, int bits);

uint32_t aom_rb_read_uvlc(struct aom_read_bit_buffer *rb);

#if CONFIG_AV1_DECODER
uint32_t aom_rb_read_unsigned_literal(struct aom_read_bit_buffer *rb, int bits);

int aom_rb_read_inv_signed_literal(struct aom_read_bit_buffer *rb, int bits);

int16_t aom_rb_read_signed_primitive_refsubexpfin(
    struct aom_read_bit_buffer *rb, uint16_t n, uint16_t k, int16_t ref);
#endif  // CONFIG_AV1_DECODER

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_DSP_BITREADER_BUFFER_H_
