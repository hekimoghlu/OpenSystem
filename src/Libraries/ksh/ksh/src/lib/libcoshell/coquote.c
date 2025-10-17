/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
 * single quote s into sp
 * if type!=0 then /<getenv(<CO_ENV_TYPE>)/ translated to /$<CO_ENV_TYPE>/
 */

#include "colib.h"

void
coquote(register Sfio_t* sp, register const char* s, int type)
{
	register int	c;

	if (type && (!state.type || !*state.type))
		type = 0;
	while (c = *s++)
	{
		sfputc(sp, c);
		if (c == '\'')
		{
			sfputc(sp, '\\');
			sfputc(sp, '\'');
			sfputc(sp, '\'');
		}
		else if (type && c == '/' && *s == *state.type)
		{
			register const char*	x = s;
			register char*		t = state.type;

			while (*t && *t++ == *x) x++;
			if (!*t && *x == '/')
			{
				s = x;
				sfprintf(sp, "'$%s'", CO_ENV_TYPE);
			}
		}
	}
}
