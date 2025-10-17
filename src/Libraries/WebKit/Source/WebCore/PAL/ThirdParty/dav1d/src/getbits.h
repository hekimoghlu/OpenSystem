/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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
#ifndef DAV1D_SRC_GETBITS_H
#define DAV1D_SRC_GETBITS_H

#include <stddef.h>
#include <stdint.h>

typedef struct GetBits {
    int error, eof;
    uint64_t state;
    unsigned bits_left;
    const uint8_t *ptr, *ptr_start, *ptr_end;
} GetBits;

void dav1d_init_get_bits(GetBits *c, const uint8_t *data, size_t sz);
unsigned dav1d_get_bits(GetBits *c, unsigned n);
int dav1d_get_sbits(GetBits *c, unsigned n);
unsigned dav1d_get_uleb128(GetBits *c);

// Output in range 0..max-1
unsigned dav1d_get_uniform(GetBits *c, unsigned max);
unsigned dav1d_get_vlc(GetBits *c);
int dav1d_get_bits_subexp(GetBits *c, int ref, unsigned n);

// Discard bits from the buffer until we're next byte-aligned.
void dav1d_bytealign_get_bits(GetBits *c);

// Return the current bit position relative to the start of the buffer.
static inline unsigned dav1d_get_bits_pos(const GetBits *c) {
    return (unsigned) (c->ptr - c->ptr_start) * 8 - c->bits_left;
}

#endif /* DAV1D_SRC_GETBITS_H */
