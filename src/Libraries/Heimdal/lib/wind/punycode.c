/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <string.h>
#include "windlocl.h"

static const unsigned base         = 36;
static const unsigned t_min        = 1;
static const unsigned t_max        = 26;
static const unsigned skew         = 38;
static const unsigned damp         = 700;
static const unsigned initial_n    = 128;
static const unsigned initial_bias = 72;

static unsigned
digit(unsigned n)
{
    return "abcdefghijklmnopqrstuvwxyz0123456789"[n];
}

static unsigned
adapt(unsigned delta, unsigned numpoints, int first)
{
    unsigned k;

    if (first)
	delta = delta / damp;
    else
	delta /= 2;
    delta += delta / numpoints;
    k = 0;
    while (delta > ((base - t_min) * t_max) / 2) {
	delta /= base - t_min;
	k += base;
    }
    return k + (((base - t_min + 1) * delta) / (delta + skew));
}

/**
 * Convert an UCS4 string to a puny-coded DNS label string suitable
 * when combined with delimiters and other labels for DNS lookup.
 *
 * @param in an UCS4 string to convert
 * @param in_len the length of in.
 * @param out the resulting puny-coded string. The string is not NUL
 * terminatied.
 * @param out_len before processing out_len should be the length of
 * the out variable, after processing it will be the length of the out
 * string.
 *
 * @return returns 0 on success, an wind error code otherwise
 * @ingroup wind
 */

int
wind_punycode_label_toascii(const uint32_t *in, size_t in_len,
			    char *out, size_t *out_len)
{
    unsigned n     = initial_n;
    unsigned delta = 0;
    unsigned bias  = initial_bias;
    unsigned h = 0;
    unsigned b;
    unsigned i;
    unsigned o = 0;
    unsigned m;

    for (i = 0; i < in_len; ++i) {
	if (in[i] < 0x80) {
	    ++h;
	    if (o >= *out_len)
		return WIND_ERR_OVERRUN;
	    out[o++] = in[i];
	}
    }
    b = h;
    if (b > 0) {
	if (o >= *out_len)
	    return WIND_ERR_OVERRUN;
	out[o++] = 0x2D;
    }
    /* is this string punycoded */
    if (h < in_len) {
	if (o + 4 >= *out_len)
	    return WIND_ERR_OVERRUN;
	memmove(out + 4, out, o);
	memcpy(out, "xn--", 4);
	o += 4;
    }

    while (h < in_len) {
	m = (unsigned)-1;
	for (i = 0; i < in_len; ++i)
	    if(in[i] < m && in[i] >= n)
		m = in[i];

	delta += (m - n) * (h + 1);
	n = m;
	for (i = 0; i < in_len; ++i) {
	    if (in[i] < n) {
		++delta;
	    } else if (in[i] == n) {
		unsigned q = delta;
		unsigned k;
		for (k = base; ; k += base) {
		    unsigned t;
		    if (k <= bias)
			t = t_min;
		    else if (k >= bias + t_max)
			t = t_max;
		    else
			t = k - bias;
		    if (q < t)
			break;
		    if (o >= *out_len)
			return WIND_ERR_OVERRUN;
		    out[o++] = digit(t + ((q - t) % (base - t)));
		    q = (q - t) / (base - t);
		}
		if (o >= *out_len)
		    return WIND_ERR_OVERRUN;
		out[o++] = digit(q);
		/* output */
		bias = adapt(delta, h + 1, h == b);
		delta = 0;
		++h;
	    }
	}
	++delta;
	++n;
    }

    *out_len = o;
    return 0;
}
