/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
#include	"sfhdr.h"

/*	Read a long value coded in a portable format.
**
**	Written by Kiem-Phong Vo
*/

#if __STD_C
Sflong_t sfgetl(Sfio_t* f)
#else
Sflong_t sfgetl(f)
Sfio_t*	f;
#endif
{
	Sflong_t	v;
	uchar		*s, *ends, c;
	int		p;
	SFMTXDECL(f); /* declare a local stream variable for multithreading */

	SFMTXENTER(f,(Sflong_t)(-1));

	if(f->mode != SF_READ && _sfmode(f,SF_READ,0) < 0)
		SFMTXRETURN(f, (Sflong_t)(-1));
	SFLOCK(f,0);

	for(v = 0;;)
	{	if(SFRPEEK(f,s,p) <= 0)
		{	f->flags |= SF_ERROR;
			v = (Sflong_t)(-1);
			goto done;
		}
		for(ends = s+p; s < ends;)
		{	c = *s++;
			if(c&SF_MORE)
				v = ((Sfulong_t)v << SF_UBITS) | SFUVALUE(c);
			else
			{	/* special translation for this byte */
				v = ((Sfulong_t)v << SF_SBITS) | SFSVALUE(c);
				f->next = s;
				v = (c&SF_SIGN) ? -v-1 : v;
				goto done;
			}
		}
		f->next = s;
	}
done :
	SFOPEN(f,0);
	SFMTXRETURN(f, v);
}
