/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#include "dboolhuff.h"
#include "vp8/common/common.h"
#include "vpx_dsp/vpx_dsp_common.h"

int vp8dx_start_decode(BOOL_DECODER *br, const unsigned char *source,
                       unsigned int source_sz, vpx_decrypt_cb decrypt_cb,
                       void *decrypt_state) {
  if (source_sz && !source) return 1;

  // To simplify calling code this fuction can be called with |source| == null
  // and |source_sz| == 0. This and vp8dx_bool_decoder_fill() are essentially
  // no-ops in this case.
  // Work around a ubsan warning with a ternary to avoid adding 0 to null.
  br->user_buffer_end = source ? source + source_sz : source;
  br->user_buffer = source;
  br->value = 0;
  br->count = -8;
  br->range = 255;
  br->decrypt_cb = decrypt_cb;
  br->decrypt_state = decrypt_state;

  /* Populate the buffer */
  vp8dx_bool_decoder_fill(br);

  return 0;
}

void vp8dx_bool_decoder_fill(BOOL_DECODER *br) {
  const unsigned char *bufptr = br->user_buffer;
  VP8_BD_VALUE value = br->value;
  int count = br->count;
  int shift = VP8_BD_VALUE_SIZE - CHAR_BIT - (count + CHAR_BIT);
  size_t bytes_left = br->user_buffer_end - bufptr;
  size_t bits_left = bytes_left * CHAR_BIT;
  int x = shift + CHAR_BIT - (int)bits_left;
  int loop_end = 0;
  unsigned char decrypted[sizeof(VP8_BD_VALUE) + 1];

  if (br->decrypt_cb) {
    size_t n = VPXMIN(sizeof(decrypted), bytes_left);
    br->decrypt_cb(br->decrypt_state, bufptr, decrypted, (int)n);
    bufptr = decrypted;
  }

  if (x >= 0) {
    count += VP8_LOTS_OF_BITS;
    loop_end = x;
  }

  if (x < 0 || bits_left) {
    while (shift >= loop_end) {
      count += CHAR_BIT;
      value |= (VP8_BD_VALUE)*bufptr << shift;
      ++bufptr;
      ++br->user_buffer;
      shift -= CHAR_BIT;
    }
  }

  br->value = value;
  br->count = count;
}
