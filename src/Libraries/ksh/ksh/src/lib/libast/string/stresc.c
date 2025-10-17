/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 8, 2024.
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
 * Glenn Fowler
 * AT&T Research
 *
 * convert \X character constants in s in place
 * the length of the converted s is returned (may have embedded \0's)
 * wide chars absent locale guidance default to UTF-8
 * strexp() FMT_EXP_* flags passed to chrexp() for selective conversion
 */

#include <ast.h>

int
strexp(register char* s, int flags)
{
	register char*		t;
	register unsigned int	c;
	char*			b;
	char*			e;
	int			w;

	b = t = s;
	while (c = *s++)
	{
		if (c == '\\')
		{
			c = chrexp(s - 1, &e, &w, flags);
			s = e;
			if (w)
			{
				t += mbwide() ? mbconv(t, c) : wc2utf8(t, c);
				continue;
			}
		}
		*t++ = c;
	}
	*t = 0;
	return t - b;
}

int
stresc(register char* s)
{
	return strexp(s, FMT_EXP_CHAR|FMT_EXP_LINE|FMT_EXP_WIDE);
}
