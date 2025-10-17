/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
#if defined(LIBC_SCCS) && !defined(lint)
static char *rcsid = "$OpenBSD: a64l.c,v 1.3 1997/08/17 22:58:34 millert Exp $";
#endif /* LIBC_SCCS and not lint */

#include <errno.h>
#include <stdlib.h>

long
a64l(const char *s)
{
	int value, digit, shift;
	int i;

	if (s == NULL) {
		errno = EINVAL;
		return(-1L);
	}

	value = 0;
	shift = 0;
	for (i = 0; *s && i < 6; i++, s++) {
		if (*s >= '.' && *s <= '/')
			digit = *s - '.';
		else if (*s >= '0' && *s <= '9')
			digit = *s - '0' + 2;
		else if (*s >= 'A' && *s <= 'Z')
			digit = *s - 'A' + 12;
		else if (*s >= 'a' && *s <= 'z')
			digit = *s - 'a' + 38;
		else {
			errno = EINVAL;
			return(-1L);
		}

		value |= digit << shift;
		shift += 6;
	}

	return(value);
}

