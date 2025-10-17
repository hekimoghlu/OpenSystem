/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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

/*	Write out a character n times
**
**	Written by Kiem-Phong Vo.
*/

#if __STD_C
ssize_t sfnputc(Sfio_t* f, int c, size_t n)
#else
ssize_t sfnputc(f,c,n)
Sfio_t*		f;	/* file to write */
int		c;	/* char to be written */
size_t		n;	/* number of time to repeat */
#endif
{
	reg uchar*	ps;
	reg ssize_t	p, w;
	uchar		buf[128];
	reg int		local;
	SFMTXDECL(f); /* declare a local stream variable for multithreading */

	SFMTXENTER(f,-1);

	GETLOCAL(f,local);
	if(SFMODE(f,local) != SF_WRITE && _sfmode(f,SF_WRITE,local) < 0)
		SFMTXRETURN(f, -1);

	SFLOCK(f,local);

	/* write into a suitable buffer */
	if((size_t)(p = (f->endb-(ps = f->next))) < n)
		{ ps = buf; p = sizeof(buf); }
	if((size_t)p > n)
		p = n;
	MEMSET(ps,c,p);
	ps -= p;

	w = n;
	if(ps == f->next)
	{	/* simple sfwrite */
		f->next += p;
		if(c == '\n')
			(void)SFFLSBUF(f,-1);
		goto done;
	}

	for(;;)
	{	/* hard write of data */
		if((p = SFWRITE(f,(Void_t*)ps,p)) <= 0 || (n -= p) <= 0)
		{	w -= n;
			goto done;
		}
		if((size_t)p > n)
			p = n;
	}
done :
	SFOPEN(f,local);
	SFMTXRETURN(f, w);
}
