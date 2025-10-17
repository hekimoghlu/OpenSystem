/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
 * strlcat implementation
 */

#define strlcat		______strlcat

#include <ast.h>

#undef	strlcat

#undef	_def_map_ast
#include <ast_map.h>

#if _lib_strlcat

NoN(strlcat)

#else

/*
 * append t onto s limiting total size of s to n
 * s 0 terminated if n>0
 * min(n,strlen(s))+strlen(t) returned
 */

#if defined(__EXPORT__)
#define extern	__EXPORT__
#endif

extern size_t
strlcat(register char* s, register const char* t, register size_t n)
{
	register size_t	m;
	const char*	o = t;

	if (m = n)
	{
		while (n && *s)
		{
			n--;
			s++;
		}
		m -= n;
		if (n)
			do
			{
				if (!--n)
				{
					*s = 0;
					break;
				}
			} while (*s++ = *t++);
		else
			*s = 0;
	}
	if (!n)
		while (*t++);
	return (t - o) + m - 1;
}

#endif
