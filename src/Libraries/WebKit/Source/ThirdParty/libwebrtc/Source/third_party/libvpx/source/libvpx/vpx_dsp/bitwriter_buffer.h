/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
#ifndef VPX_VPX_DSP_BITWRITER_BUFFER_H_
#define VPX_VPX_DSP_BITWRITER_BUFFER_H_

#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct vpx_write_bit_buffer {
  // Whether there has been an error.
  int error;
  // We maintain the invariant that bit_offset <= size * CHAR_BIT, i.e., we
  // never write beyond the end of bit_buffer. If bit_offset would be
  // incremented to be greater than size * CHAR_BIT, leave bit_offset unchanged
  // and set error to 1.
  size_t bit_offset;
  // Size of bit_buffer in bytes.
  size_t size;
  uint8_t *bit_buffer;
};

void vpx_wb_init(struct vpx_write_bit_buffer *wb, uint8_t *bit_buffer,
                 size_t size);

int vpx_wb_has_error(const struct vpx_write_bit_buffer *wb);

// Must not be called if vpx_wb_has_error(wb) returns true.
size_t vpx_wb_bytes_written(const struct vpx_write_bit_buffer *wb);

void vpx_wb_write_bit(struct vpx_write_bit_buffer *wb, int bit);

void vpx_wb_write_literal(struct vpx_write_bit_buffer *wb, int data, int bits);

void vpx_wb_write_inv_signed_literal(struct vpx_write_bit_buffer *wb, int data,
                                     int bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_BITWRITER_BUFFER_H_
