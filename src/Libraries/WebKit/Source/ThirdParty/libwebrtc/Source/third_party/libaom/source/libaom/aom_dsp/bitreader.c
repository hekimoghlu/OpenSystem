/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
#include "aom_dsp/bitreader.h"

int aom_reader_init(aom_reader *r, const uint8_t *buffer, size_t size) {
  if (size && !buffer) {
    return 1;
  }
  r->buffer_end = buffer + size;
  r->buffer = buffer;
  od_ec_dec_init(&r->ec, buffer, (uint32_t)size);
#if CONFIG_ACCOUNTING
  r->accounting = NULL;
#endif
  return 0;
}

const uint8_t *aom_reader_find_begin(aom_reader *r) { return r->buffer; }

const uint8_t *aom_reader_find_end(aom_reader *r) { return r->buffer_end; }

uint32_t aom_reader_tell(const aom_reader *r) { return od_ec_dec_tell(&r->ec); }

uint32_t aom_reader_tell_frac(const aom_reader *r) {
  return od_ec_dec_tell_frac(&r->ec);
}

int aom_reader_has_overflowed(const aom_reader *r) {
  const uint32_t tell_bits = aom_reader_tell(r);
  const uint32_t tell_bytes = (tell_bits + 7) >> 3;
  return ((ptrdiff_t)tell_bytes > r->buffer_end - r->buffer);
}
