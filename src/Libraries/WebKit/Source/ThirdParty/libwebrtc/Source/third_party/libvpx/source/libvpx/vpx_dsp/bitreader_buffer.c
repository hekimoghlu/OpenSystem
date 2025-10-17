/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
#include "./vpx_config.h"
#include "./bitreader_buffer.h"

size_t vpx_rb_bytes_read(struct vpx_read_bit_buffer *rb) {
  return (rb->bit_offset + 7) >> 3;
}

int vpx_rb_read_bit(struct vpx_read_bit_buffer *rb) {
  const size_t off = rb->bit_offset;
  const size_t p = off >> 3;
  const int q = 7 - (int)(off & 0x7);
  if (rb->bit_buffer + p < rb->bit_buffer_end) {
    const int bit = (rb->bit_buffer[p] >> q) & 1;
    rb->bit_offset = off + 1;
    return bit;
  } else {
    if (rb->error_handler != NULL) rb->error_handler(rb->error_handler_data);
    return 0;
  }
}

int vpx_rb_read_literal(struct vpx_read_bit_buffer *rb, int bits) {
  int value = 0, bit;
  for (bit = bits - 1; bit >= 0; bit--) value |= vpx_rb_read_bit(rb) << bit;
  return value;
}

int vpx_rb_read_signed_literal(struct vpx_read_bit_buffer *rb, int bits) {
  const int value = vpx_rb_read_literal(rb, bits);
  return vpx_rb_read_bit(rb) ? -value : value;
}

int vpx_rb_read_inv_signed_literal(struct vpx_read_bit_buffer *rb, int bits) {
  return vpx_rb_read_signed_literal(rb, bits);
}
