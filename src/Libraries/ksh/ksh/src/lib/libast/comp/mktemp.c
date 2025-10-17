/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
 * mktemp,mkstemp implementation
 */

#define mktemp		______mktemp
#define mkstemp		______mkstemp

#include <ast.h>
#include <stdio.h>

#undef	mktemp
#undef	mkstemp

#undef	_def_map_ast
#include <ast_map.h>

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

static char*
temp(char* buf, int* fdp)
{
	char*	s;
	char*	d;
	int	n;
	size_t	len;

	len = strlen(buf);
	if (s = strrchr(buf, '/'))
	{
		*s++ = 0;
		d = buf;
	}
	else
	{
		s = buf;
		d = "";
	}
	if ((n = strlen(s)) < 6 || strcmp(s + n - 6, "XXXXXX"))
		*buf = 0;
	else
	{
		*(s + n - 6) = 0;
		if (!pathtemp(buf, len, d, s, fdp))
			*buf = 0;
	}
	return buf;
}

extern char*
mktemp(char* buf)
{
	return temp(buf, NiL);
}

extern int
mkstemp(char* buf)
{
	int	fd;

	return *temp(buf, &fd) ? fd : -1;
}
