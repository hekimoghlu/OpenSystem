/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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

#include <tmx.h>
#include <ctype.h>

/*
 * parse duration expression in s and return Time_t value
 * if non-null, e points to the first unused char in s
 * returns 0 with *e==s on error
 */

Time_t
tmxduration(const char* s, char** e)
{
	Time_t		ns;
	Time_t		ts;
	Time_t		now;
	char*		last;
	char*		t;
	char*		x;
	Sfio_t*		f;
	int		i;

	now = TMX_NOW;
	while (isspace(*s))
		s++;
	if (*s == 'P' || *s == 'p')
		ns = tmxdate(s, &last, now) - now;
	else
	{
		ns = strtod(s, &last) * TMX_RESOLUTION;
		if (*last && (f = sfstropen()))
		{
			sfprintf(f, "exact %s", s);
			t = sfstruse(f);
			ts = tmxdate(t, &x, now);
			if ((i = x - t - 6) > (last - s))
			{
				last = (char*)s + i;
				ns = ts - now;
			}
			else
			{
				sfprintf(f, "p%s", s);
				t = sfstruse(f);
				ts = tmxdate(t, &x, now);
				if ((i = x - t - 1) > (last - s))
				{
					last = (char*)s + i;
					ns = ts - now;
				}
			}
			sfstrclose(f);
		}
	}
	if (e)
		*e = last;
	return ns;
}
