/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
/*LINTLIBRARY*/

#include "private.h"

const char *
scheck(const char *const string, const char *const format)
{
	register char *		fbuf;
	register const char *	fp;
	register char *		tp;
	register int		c;
	register const char *	result;
	char			dummy;

	result = "";
	if (string == NULL || format == NULL)
		return result;
	fbuf = malloc(2 * strlen(format) + 4);
	if (fbuf == NULL)
		return result;
	fp = format;
	tp = fbuf;

	/*
	** Copy directives, suppressing each conversion that is not
	** already suppressed.  Scansets containing '%' are not
	** supported; e.g., the conversion specification "%[%]" is not
	** supported.  Also, multibyte characters containing a
	** non-leading '%' byte are not supported.
	*/
	while ((*tp++ = c = *fp++) != '\0') {
		if (c != '%')
			continue;
		if (is_digit(*fp)) {
			char const *f = fp;
			char *t = tp;
			do {
				*t++ = c = *f++;
			} while (is_digit(c));
			if (c == '$') {
				fp = f;
				tp = t;
			}
		}
		*tp++ = '*';
		if (*fp == '*')
			++fp;
		if ((*tp++ = *fp++) == '\0')
			break;
	}

	*(tp - 1) = '%';
	*tp++ = 'c';
	*tp = '\0';
	if (sscanf(string, fbuf, &dummy) != 1)
		result = format;
	free(fbuf);
	return result;
}
