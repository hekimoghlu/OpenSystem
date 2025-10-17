/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#pragma prototyped

/*
 * return the length of the current record at b, size n, according to f
 * -1 returned on error
 *  0 returned if more data is required
 */

#include <recfmt.h>
#include <ctype.h>

ssize_t
reclen(Recfmt_t f, const void* b, size_t n)
{
	register unsigned char*	s = (unsigned char*)b;
	register unsigned char*	e;
	size_t			h;
	size_t			z;

	switch (RECTYPE(f))
	{
	case REC_delimited:
		if (e = (unsigned char*)memchr(s, REC_D_DELIMITER(f), n))
			return e - s + 1;
		return 0;
	case REC_fixed:
		return REC_F_SIZE(f);
	case REC_variable:
		h = REC_V_HEADER(f);
		if (n < h)
			return 0;
		z = 0;
		s += REC_V_OFFSET(f);
		e = s + REC_V_LENGTH(f);
		if (REC_V_LITTLE(f))
			while (e > s)
				z = (z<<8)|*--e;
		else
			while (s < e)
				z = (z<<8)|*s++;
		if (!REC_V_INCLUSIVE(f))
			z += h;
		else if (z < h)
			z = h;
		return z;
	case REC_method:
		return -1;
	}
	return -1;
}
