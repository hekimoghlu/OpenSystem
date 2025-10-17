/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#include <sys/endian.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <strings.h>

#include "math_private.h"

/*
 * Scan a string of hexadecimal digits (the format nan(3) expects) and
 * make a bit array (using the local endianness). We stop when we
 * encounter an invalid character, NUL, etc.  If we overflow, we do
 * the same as gcc's __builtin_nan(), namely, discard the high order bits.
 *
 * The format this routine accepts needs to be compatible with what is used
 * in contrib/gdtoa/hexnan.c (for strtod/scanf) and what is used in
 * __builtin_nan(). In fact, we're only 100% compatible for strings we
 * consider valid, so we might be violating the C standard. But it's
 * impossible to use nan(3) portably anyway, so this seems good enough.
 */
void
_scan_nan(uint32_t *words, int num_words, const char *s)
{
	int si;		/* index into s */
	int bitpos;	/* index into words (in bits) */

	bzero(words, num_words * sizeof(uint32_t));

	/* Allow a leading '0x'. (It's expected, but redundant.) */
	if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
		s += 2;

	/* Scan forwards in the string, looking for the end of the sequence. */
	for (si = 0; isxdigit(s[si]); si++)
		;

	/* Scan backwards, filling in the bits in words[] as we go. */
	for (bitpos = 0; bitpos < 32 * num_words; bitpos += 4) {
		if (--si < 0)
			break;
#if _BYTE_ORDER == _LITTLE_ENDIAN
		words[bitpos / 32] |= digittoint(s[si]) << (bitpos % 32);
#else
		words[num_words - 1 - bitpos / 32] |=
		    digittoint(s[si]) << (bitpos % 32);
#endif
	}
}

double
nan(const char *s)
{
	union {
		double d;
		uint32_t bits[2];
	} u;

	_scan_nan(u.bits, 2, s);
#if _BYTE_ORDER == _LITTLE_ENDIAN
	u.bits[1] |= 0x7ff80000;
#else
	u.bits[0] |= 0x7ff80000;
#endif
	return (u.d);
}

float
nanf(const char *s)
{
	union {
		float f;
		uint32_t bits[1];
	} u;

	_scan_nan(u.bits, 1, s);
	u.bits[0] |= 0x7fc00000;
	return (u.f);
}

#if (LDBL_MANT_DIG == 53)
__weak_reference(nan, nanl);
#endif
