/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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
#include <assert.h>
#include <limits.h>
#include <stdlib.h>

#include "./vpx_config.h"
#include "./bitwriter_buffer.h"

void vpx_wb_init(struct vpx_write_bit_buffer *wb, uint8_t *bit_buffer,
                 size_t size) {
  wb->error = 0;
  wb->bit_offset = 0;
  wb->size = size;
  wb->bit_buffer = bit_buffer;
}

int vpx_wb_has_error(const struct vpx_write_bit_buffer *wb) {
  return wb->error;
}

size_t vpx_wb_bytes_written(const struct vpx_write_bit_buffer *wb) {
  assert(!wb->error);
  return wb->bit_offset / CHAR_BIT + (wb->bit_offset % CHAR_BIT > 0);
}

void vpx_wb_write_bit(struct vpx_write_bit_buffer *wb, int bit) {
  if (wb->error) return;
  const int off = (int)wb->bit_offset;
  const int p = off / CHAR_BIT;
  const int q = CHAR_BIT - 1 - off % CHAR_BIT;
  if ((size_t)p >= wb->size) {
    wb->error = 1;
    return;
  }
  if (q == CHAR_BIT - 1) {
    wb->bit_buffer[p] = bit << q;
  } else {
    assert((wb->bit_buffer[p] & (1 << q)) == 0);
    wb->bit_buffer[p] |= bit << q;
  }
  wb->bit_offset = off + 1;
}

void vpx_wb_write_literal(struct vpx_write_bit_buffer *wb, int data, int bits) {
  int bit;
  for (bit = bits - 1; bit >= 0; bit--) vpx_wb_write_bit(wb, (data >> bit) & 1);
}

void vpx_wb_write_inv_signed_literal(struct vpx_write_bit_buffer *wb, int data,
                                     int bits) {
  vpx_wb_write_literal(wb, abs(data), bits);
  vpx_wb_write_bit(wb, data < 0);
}
