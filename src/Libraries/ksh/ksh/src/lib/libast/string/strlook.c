/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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
 */

#include <ast.h>

/*
 * return pointer to name in tab with element size siz
 * where the first member of each element is a char*
 *
 * the last name in tab must be 0
 *
 * 0 returned if name not found
 */

void*
strlook(const void* tab, size_t siz, register const char* name)
{
	register char*	t = (char*)tab;
	register char*	s;
	register int	c = *name;

	for (; s = *((char**)t); t += siz)
		if (*s == c && !strcmp(s, name))
			return (void*)t;
	return 0;
}
