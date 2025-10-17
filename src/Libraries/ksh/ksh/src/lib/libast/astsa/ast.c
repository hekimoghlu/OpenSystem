/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
 * standalone mini ast+sfio implementation
 */

#include <ast.h>

#define CHUNK		1024

_Ast_info_t		ast;

int
astwinsize(int fd, int* lines, int* columns)
{
	if (lines)
		*lines = 24;
	if (columns)
		*columns = 80;
	return 0;
}

char*
sfgetr(Sfio_t* sp, int c, int z)
{
	register char*		s;
	register char*		e;

	static char*		buf;
	static unsigned long	siz;

	if (!buf)
	{
		siz = CHUNK;
		if (!(buf = newof(0, char, siz, 0)))
			return 0;
	}
	if (z < 0)
		return *buf ? buf : (char*)0;
	s = buf;
	e = s + siz;
	for (;;)
	{
		if (s >= e)
		{
			siz += CHUNK;
			if (!(buf = newof(buf, char, siz, 0)))
				return 0;
			s = buf + (siz - CHUNK);
			e = s + siz;
		}
		if ((c = sfgetc(sp)) == EOF)
		{
			*s = 0;
			return 0;
		}
		if (c == '\n')
		{
			*s = z ? 0 : c;
			break;
		}
		*s++ = c;
	}
	return buf;
}
