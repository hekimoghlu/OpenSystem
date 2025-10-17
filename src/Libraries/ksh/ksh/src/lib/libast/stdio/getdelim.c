/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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

#include "stdhdr.h"

ssize_t
getdelim(char** sp, size_t* np, int delim, Sfio_t* f)
{
	ssize_t		m;
	ssize_t		n;
	ssize_t		k;
	ssize_t		p;
	uchar*		s;
	uchar*		ps;
	SFMTXDECL(f);

	STDIO_INT(f, "getdelim", ssize_t, (char**, size_t*, int, Sfio_t*), (sp, np, delim, f))

	SFMTXENTER(f, -1);

	if(delim < 0 || delim > 255 || !sp || !np) /* bad parameters */
		SFMTXRETURN(f, -1);

	if(f->mode != SF_READ && _sfmode(f,SF_READ,0) < 0)
		SFMTXRETURN(f, -1);

	SFLOCK(f,0);

	if(!(s = (uchar*)(*sp)) || (n = *np) < 0)
		{ s = NIL(uchar*); n = 0; }
	for(m = 0;; )
	{	/* read new data */
		if((p = f->endb - (ps = f->next)) <= 0 )
		{	f->getr = delim;
			f->mode |= SF_RC;
			if(SFRPEEK(f,ps,p) <= 0)
			{	m = -1;
				break;
			}
		}

		for(k = 0; k < p; ++k) /* find the delimiter */
		{	if(ps[k] == delim)
			{	k += 1; /* include delim in copying */
				break;
			}
		}

		if((m+k+1) >= n ) /* make sure there is space */
		{	n = ((m+k+15)/8)*8;
			if(!(s = (uchar*)realloc(s, n)) )
			{	*sp = 0; *np = 0;
				m = -1;
				break;
			}
			*sp = (char*)s; *np = n;
		}

		memcpy(s+m, ps, k); m += k;
		f->next = ps+k; /* skip copied data in buffer */

		if(s[m-1] == delim)
		{	s[m] = 0; /* 0-terminated */
			break;
		}
	}

	SFOPEN(f,0);
	SFMTXRETURN(f,m);
}

ssize_t
__getdelim(char** sp, size_t* np, int delim, Sfio_t* f)
{
	return getdelim(sp, np, delim, f);
}
