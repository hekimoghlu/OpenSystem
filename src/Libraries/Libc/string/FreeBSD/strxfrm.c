/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 16, 2023.
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
__FBSDID("$FreeBSD: src/lib/libc/string/strxfrm.c,v 1.17 2008/10/19 09:10:44 delphij Exp $");

#include "xlocale_private.h"

#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <errno.h>
#include "collate.h"

/*
 * In the non-POSIX case, we transform each character into a string of
 * characters representing the character's priority.  Since char is usually
 * signed, we are limited by 7 bits per byte.  To avoid zero, we need to add
 * XFRM_OFFSET, so we can't use a full 7 bits.  For simplicity, we choose 6
 * bits per byte.  We choose 4 bytes per character as a good compromise
 * between maximum coverage and minimum size.  This gives 24 bits, or 16M
 * priorities.  So we choose COLLATE_MAX_PRIORITY to be (2^24 - 1).  This
 * this can be increased if more is needed.
 */

#define	XFRM_BYTES	4
#define	XFRM_OFFSET	('0')	/* make all printable characters */
#define	XFRM_SHIFT	6
#define	XFRM_MASK	((1 << XFRM_SHIFT) - 1)

static void
xfrm(unsigned char *p, int pri)
{

	p[3] = (pri & XFRM_MASK) + XFRM_OFFSET;
	pri >>= XFRM_SHIFT;
	p[2] = (pri & XFRM_MASK) + XFRM_OFFSET;
	pri >>= XFRM_SHIFT;
	p[1] = (pri & XFRM_MASK) + XFRM_OFFSET;
	pri >>= XFRM_SHIFT;
	p[0] = (pri & XFRM_MASK) + XFRM_OFFSET;
}

size_t
strxfrm_l(char * __restrict dest, const char * __restrict src, size_t len,
    locale_t loc)
{
	size_t slen;
	wchar_t *wcs, *xf[COLL_WEIGHTS_MAX];
	int sverrno;

	if (!*src && dest) {
		if (len > 0)
			*dest = '\0';
		return 0;
	}

	NORMALIZE_LOCALE(loc);
	if (XLOCALE_COLLATE(loc)->__collate_load_error ||
	    (wcs = __collate_mbstowcs(src, loc)) == NULL)
		return strlcpy(dest, src, len);

	__collate_xfrm(wcs, xf, loc);

	/*
	 * XXX This and wcsxfrm both need fixed to work in our new localedata
	 * world.
	 */
	slen = wcslen(xf[0]) * XFRM_BYTES;
	if (xf[1])
		slen += (wcslen(xf[1]) + 1) * XFRM_BYTES;
	if (len > 0) {
		wchar_t *w = xf[0];
		int b = 0;
		unsigned char buf[XFRM_BYTES];
		unsigned char *bp;
		while (len > 1) {
			if (!b) {
				if (!*w)
					break;
				xfrm(bp = buf, *w++);
				b = XFRM_BYTES;
			}
			*dest++ = *(char *)bp++;
			b--;
			len--;
		}
		if ((w = xf[1]) != NULL) {
			xfrm(bp = buf, 0);
			b = XFRM_BYTES;
			while (len > 1) {
				if (!b)
					break;
				*dest++ = *(char *)bp++;
				b--;
				len--;
			}
			b = 0;
			while (len > 1) {
				if (!b) {
					if (!*w)
						break;
					xfrm(bp = buf, *w++);
					b = XFRM_BYTES;
				}
				*dest++ = *(char *)bp++;
				b--;
				len--;
			}
		}
		*dest = 0;
 	}
	sverrno = errno;
	free(wcs);
	free(xf[0]);
	free(xf[1]);
	errno = sverrno;

	return slen;
}

size_t
strxfrm(char * __restrict dest, const char * __restrict src, size_t len)
{
	return strxfrm_l(dest, src, len, __current_locale());
}
