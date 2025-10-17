/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
 * time conversion support
 */

#include <ast.h>
#include <tm.h>

/*
 * return timezone pointer given name and type
 *
 * if type==0 then all time zone types match
 * otherwise type must be one of tm_info.zone[].type
 *
 * if end is non-null then it will point to the next
 * unmatched char in name
 *
 * if dst!=0 then it will point to 0 for standard zones
 * and the offset for daylight zones
 *
 * 0 returned for no match
 */

Tm_zone_t*
tmzone(register const char* name, char** end, const char* type, int* dst)
{
	register Tm_zone_t*	zp;
	register char*		prev;
	char*			e;

	static Tm_zone_t	fixed;
	static char		off[16];

	tmset(tm_info.zone);
	if ((*name == '+' || *name == '-') && (fixed.west = tmgoff(name, &e, TM_LOCALZONE)) != TM_LOCALZONE && !*e)
	{
		strlcpy(fixed.standard = fixed.daylight = off, name, sizeof(off));
		if (end)
			*end = e;
		if (dst)
			*dst = 0;
		return &fixed;
	}
	zp = tm_info.local;
	prev = 0;
	do
	{
		if (zp->type)
			prev = zp->type;
		if (!type || type == prev || !prev)
		{
			if (tmword(name, end, zp->standard, NiL, 0))
			{
				if (dst)
					*dst = 0;
				return zp;
			}
			if (zp->dst && zp->daylight && tmword(name, end, zp->daylight, NiL, 0))
			{
				if (dst)
					*dst = zp->dst;
				return zp;
			}
		}
		if (zp == tm_info.local)
			zp = tm_data.zone;
		else
			zp++;
	} while (zp->standard);
	return 0;
}
