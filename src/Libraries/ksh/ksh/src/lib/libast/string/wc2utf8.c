/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
 * convert wide character to utf8 in s
 * s must have room for at least 6 bytes
 * return value is the number of chars placed in s
 * thanks Tom Duff
 */

#include <ast.h>

typedef struct Utf8_s
{
	uint32_t	range;
	unsigned short	prefix;
	unsigned short	shift;
} Utf8_t;

static const Utf8_t	ops[] =
{
	{ 0x00000080, 0x00,  0 },
	{ 0x00000800, 0xc0,  6 },
	{ 0x00010000, 0xe0, 12 },
	{ 0x00200000, 0xf0, 18 },
	{ 0x04000000, 0xf8, 24 },
	{ 0x80000000, 0xfc, 30 }
};

int
wc2utf8(register char* s, register uint32_t w)
{
	register int	i;
	char*		b;

	for (i = 0; i < elementsof(ops); i++)
		if (w < ops[i].range)
		{
			b = s;
			*s++ = ops[i].prefix | (w >> ops[i].shift);
			switch (ops[i].shift)
			{
			case 30:	*s++ = 0x80 | ((w >> 24) & 0x3f);
			case 24:	*s++ = 0x80 | ((w >> 18) & 0x3f);
			case 18:	*s++ = 0x80 | ((w >> 12) & 0x3f);
			case 12:	*s++ = 0x80 | ((w >>  6) & 0x3f);
			case  6:	*s++ = 0x80 | (w & 0x3f);
			}
			return s - b;
		}
	return 0;
}
