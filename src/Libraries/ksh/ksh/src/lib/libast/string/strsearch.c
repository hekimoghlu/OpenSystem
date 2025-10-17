/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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
 * return a pointer to the element matching
 * name in the (*comparf*)() sorted tab of num elements of
 * size siz where the first member of each
 * element is a char*
 *
 * 0 returned if name not found
 */

void*
strsearch(const void* tab, size_t num, size_t siz, Strcmp_f comparf, const char* name, void* context)
{
	register char*		lo = (char*)tab;
	register char*		hi = lo + (num - 1) * siz;
	register char*		mid;
	register int		v;

	while (lo <= hi)
	{
		mid = lo + (((hi - lo) / siz) / 2) * siz;
		if (!(v = context ? (*(Strcmp_context_f)comparf)(name, *((char**)mid), context) : (*comparf)(name, *((char**)mid))))
			return (void*)mid;
		else if (v > 0)
			lo = mid + siz;
		else hi = mid - siz;
	}
	return 0;
}
