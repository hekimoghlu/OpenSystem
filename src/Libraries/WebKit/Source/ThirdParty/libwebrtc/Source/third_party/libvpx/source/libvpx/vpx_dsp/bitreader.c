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
#include <stdlib.h>

#include "./vpx_config.h"

#include "vpx_dsp/bitreader.h"
#include "vpx_dsp/prob.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_ports/mem.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_util/endian_inl.h"

int vpx_reader_init(vpx_reader *r, const uint8_t *buffer, size_t size,
                    vpx_decrypt_cb decrypt_cb, void *decrypt_state) {
  if (size && !buffer) {
    return 1;
  } else {
    r->buffer_end = buffer + size;
    r->buffer = buffer;
    r->value = 0;
    r->count = -8;
    r->range = 255;
    r->decrypt_cb = decrypt_cb;
    r->decrypt_state = decrypt_state;
    vpx_reader_fill(r);
    return vpx_read_bit(r) != 0;  // marker bit
  }
}

void vpx_reader_fill(vpx_reader *r) {
  const uint8_t *const buffer_end = r->buffer_end;
  const uint8_t *buffer = r->buffer;
  const uint8_t *buffer_start = buffer;
  BD_VALUE value = r->value;
  int count = r->count;
  const size_t bytes_left = buffer_end - buffer;
  const size_t bits_left = bytes_left * CHAR_BIT;
  int shift = BD_VALUE_SIZE - CHAR_BIT - (count + CHAR_BIT);

  if (r->decrypt_cb) {
    size_t n = VPXMIN(sizeof(r->clear_buffer), bytes_left);
    r->decrypt_cb(r->decrypt_state, buffer, r->clear_buffer, (int)n);
    buffer = r->clear_buffer;
    buffer_start = r->clear_buffer;
  }
  if (bits_left > BD_VALUE_SIZE) {
    const int bits = (shift & 0xfffffff8) + CHAR_BIT;
    BD_VALUE nv;
    BD_VALUE big_endian_values;
    memcpy(&big_endian_values, buffer, sizeof(BD_VALUE));
#if SIZE_MAX == 0xffffffffffffffffULL
    big_endian_values = HToBE64(big_endian_values);
#else
    big_endian_values = HToBE32(big_endian_values);
#endif
    nv = big_endian_values >> (BD_VALUE_SIZE - bits);
    count += bits;
    buffer += (bits >> 3);
    value = r->value | (nv << (shift & 0x7));
  } else {
    const int bits_over = (int)(shift + CHAR_BIT - (int)bits_left);
    int loop_end = 0;
    if (bits_over >= 0) {
      count += LOTS_OF_BITS;
      loop_end = bits_over;
    }

    if (bits_over < 0 || bits_left) {
      while (shift >= loop_end) {
        count += CHAR_BIT;
        value |= (BD_VALUE)*buffer++ << shift;
        shift -= CHAR_BIT;
      }
    }
  }

  // NOTE: Variable 'buffer' may not relate to 'r->buffer' after decryption,
  // so we increase 'r->buffer' by the amount that 'buffer' moved, rather than
  // assign 'buffer' to 'r->buffer'.
  r->buffer += buffer - buffer_start;
  r->value = value;
  r->count = count;
}

const uint8_t *vpx_reader_find_end(vpx_reader *r) {
  // Find the end of the coded buffer
  while (r->count > CHAR_BIT && r->count < BD_VALUE_SIZE) {
    r->count -= CHAR_BIT;
    r->buffer--;
  }
  return r->buffer;
}
