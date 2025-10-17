/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 19, 2022.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/string/strlen.c,v 1.7 2009/01/26 07:31:28 delphij Exp $");

#include <limits.h>
#include <sys/types.h>
#include <platform/string.h>

#if !_PLATFORM_OPTIMIZED_STRLEN

/*
 * Portable strlen() for 32-bit and 64-bit systems.
 *
 * Rationale: it is generally much more efficient to do word length
 * operations and avoid branches on modern computer systems, as
 * compared to byte-length operations with a lot of branches.
 *
 * The expression:
 *
 *	((x - 0x01....01) & ~x & 0x80....80)
 *
 * would evaluate to a non-zero value iff any of the bytes in the
 * original word is zero.  However, we can further reduce ~1/3 of
 * time if we consider that strlen() usually operate on 7-bit ASCII
 * by employing the following expression, which allows false positive
 * when high bit of 1 and use the tail case to catch these case:
 *
 *	((x - 0x01....01) & 0x80....80)
 *
 * This is more than 5.2 times as fast as the raw implementation on
 * Intel T7300 under long mode for strings longer than word length.
 */

/* Magic numbers for the algorithm */
#if LONG_BIT == 32
static const unsigned long mask01 = 0x01010101;
static const unsigned long mask80 = 0x80808080;
#elif LONG_BIT == 64
static const unsigned long mask01 = 0x0101010101010101;
static const unsigned long mask80 = 0x8080808080808080;
#else
#error Unsupported word size
#endif

#define	LONGPTR_MASK (sizeof(long) - 1)

/*
 * Helper macro to return string length if we caught the zero
 * byte.
 */
#define testbyte(x)				\
	do {					\
		if (p[x] == '\0')		\
		    return (p - str + x);	\
	} while (0)

size_t
_platform_strlen(const char *str)
{
	const char *p;
	const unsigned long *lp;

	/* Skip the first few bytes until we have an aligned p */
	for (p = str; (uintptr_t)p & LONGPTR_MASK; p++)
	    if (*p == '\0')
		return (p - str);

	/* Scan the rest of the string using word sized operation */
	for (lp = (const unsigned long *)p; ; lp++)
	    if ((*lp - mask01) & mask80) {
		p = (const char *)(lp);
		testbyte(0);
		testbyte(1);
		testbyte(2);
		testbyte(3);
#if (LONG_BIT >= 64)
		testbyte(4);
		testbyte(5);
		testbyte(6);
		testbyte(7);
#endif
	    }

	/* NOTREACHED */
	return (0);
}

#if VARIANT_STATIC
size_t
strlen(const char *str)
{
	return _platform_strlen(str);
}
#endif

#endif

