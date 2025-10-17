/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
 * AT&T Bell Laboratories
 *
 * time conversion support
 */

#include <ast.h>
#include <tm.h>
#include <ctype.h>

/*
 * return the tab table index that matches s ignoring case and .'s
 * tm_data.format checked if tminfo.format!=tm_data.format
 *
 * ntab and nsuf are the number of elements in tab and suf,
 * -1 for 0 sentinel
 *
 * all isalpha() chars in str must match
 * suf is a table of nsuf valid str suffixes 
 * if e is non-null then it will point to first unmatched char in str
 * which will always be non-isalpha()
 */

int
tmlex(register const char* s, char** e, char** tab, int ntab, char** suf, int nsuf)
{
	register char**	p;
	register char*	x;
	register int	n;

	for (p = tab, n = ntab; n-- && (x = *p); p++)
		if (*x && *x != '%' && tmword(s, e, x, suf, nsuf))
			return p - tab;
	if (tm_info.format != tm_data.format && tab >= tm_info.format && tab < tm_info.format + TM_NFORM)
	{
		tab = tm_data.format + (tab - tm_info.format);
		if (suf && tab >= tm_info.format && tab < tm_info.format + TM_NFORM)
			suf = tm_data.format + (suf - tm_info.format);
		for (p = tab, n = ntab; n-- && (x = *p); p++)
			if (*x && *x != '%' && tmword(s, e, x, suf, nsuf))
				return p - tab;
	}
	return -1;
}
