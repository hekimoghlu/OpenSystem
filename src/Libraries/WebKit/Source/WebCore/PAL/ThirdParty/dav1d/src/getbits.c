/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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
#include "config.h"

#include <limits.h>

#include "common/intops.h"

#include "src/getbits.h"

void dav1d_init_get_bits(GetBits *const c, const uint8_t *const data,
                         const size_t sz)
{
    // If sz were 0, c->eof would need to be initialized to 1.
    assert(sz);
    c->ptr = c->ptr_start = data;
    c->ptr_end = &c->ptr_start[sz];
    c->bits_left = 0;
    c->state = 0;
    c->error = 0;
    c->eof = 0;
}

static void refill(GetBits *const c, const unsigned n) {
    assert(c->bits_left <= 56);
    uint64_t state = 0;
    do {
        state <<= 8;
        c->bits_left += 8;
        if (!c->eof)
            state |= *c->ptr++;
        if (c->ptr >= c->ptr_end) {
            c->error = c->eof;
            c->eof = 1;
        }
    } while (n > c->bits_left);
    c->state |= state << (64 - c->bits_left);
}

unsigned dav1d_get_bits(GetBits *const c, const unsigned n) {
    assert(n <= 32 /* can go up to 57 if we change return type */);
    assert(n /* can't shift state by 64 */);

    if (n > c->bits_left) refill(c, n);

    const uint64_t state = c->state;
    c->bits_left -= n;
    c->state <<= n;

    return (unsigned) (state >> (64 - n));
}

int dav1d_get_sbits(GetBits *const c, const unsigned n) {
    const int shift = 31 - n;
    const int res = dav1d_get_bits(c, n + 1) << shift;
    return res >> shift;
}

unsigned dav1d_get_uleb128(GetBits *const c) {
    uint64_t val = 0;
    unsigned i = 0, more;

    do {
        const int v = dav1d_get_bits(c, 8);
        more = v & 0x80;
        val |= ((uint64_t) (v & 0x7F)) << i;
        i += 7;
    } while (more && i < 56);

    if (val > UINT_MAX || more) {
        c->error = 1;
        return 0;
    }

    return (unsigned) val;
}

unsigned dav1d_get_uniform(GetBits *const c, const unsigned max) {
    // Output in range [0..max-1]
    // max must be > 1, or else nothing is read from the bitstream
    assert(max > 1);
    const int l = ulog2(max) + 1;
    assert(l > 1);
    const unsigned m = (1U << l) - max;
    const unsigned v = dav1d_get_bits(c, l - 1);
    return v < m ? v : (v << 1) - m + dav1d_get_bits(c, 1);
}

unsigned dav1d_get_vlc(GetBits *const c) {
    int n_bits = 0;
    while (!dav1d_get_bits(c, 1))
        if (++n_bits == 32)
            return 0xFFFFFFFFU;
    return n_bits ? ((1U << n_bits) - 1) + dav1d_get_bits(c, n_bits) : 0;
}

static unsigned get_bits_subexp_u(GetBits *const c, const unsigned ref,
                                  const unsigned n)
{
    unsigned v = 0;

    for (int i = 0;; i++) {
        const int b = i ? 3 + i - 1 : 3;

        if (n < v + 3 * (1 << b)) {
            v += dav1d_get_uniform(c, n - v + 1);
            break;
        }

        if (!dav1d_get_bits(c, 1)) {
            v += dav1d_get_bits(c, b);
            break;
        }

        v += 1 << b;
    }

    return ref * 2 <= n ? inv_recenter(ref, v) : n - inv_recenter(n - ref, v);
}

int dav1d_get_bits_subexp(GetBits *const c, const int ref, const unsigned n) {
    return (int) get_bits_subexp_u(c, ref + (1 << n), 2 << n) - (1 << n);
}

void dav1d_bytealign_get_bits(GetBits *c) {
    // bits_left is never more than 7, because it is only incremented
    // by refill(), called by dav1d_get_bits and that never reads more
    // than 7 bits more than it needs.
    //
    // If this wasn't true, we would need to work out how many bits to
    // discard (bits_left % 8), subtract that from bits_left and then
    // shift state right by that amount.
    assert(c->bits_left <= 7);

    c->bits_left = 0;
    c->state = 0;
}
