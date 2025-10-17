/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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

extern char*
_stdgets(Sfio_t* f, char* us, int n, int isgets)
{
	int		p;
	unsigned char*	is;
	unsigned char*	ps;

	if(n <= 0 || !us || (f->mode != SF_READ && _sfmode(f,SF_READ,0) < 0))
		return NIL(char*);

	SFLOCK(f,0);

	n -= 1;
	is = (uchar*)us;
	
	while(n)
	{	/* peek the read buffer for data */
		if((p = f->endb - (ps = f->next)) <= 0 )
		{	f->getr = '\n';
			f->mode |= SF_RC;
			if(SFRPEEK(f,ps,p) <= 0)
				break;
		}

		if(p > n)
			p = n;

#if _lib_memccpy
		if((ps = (uchar*)memccpy((char*)is,(char*)ps,'\n',p)) != NIL(uchar*))
			p = ps-is;
		is += p;
		ps  = f->next+p;
#else
		if(!(f->flags&(SF_BOTH|SF_MALLOC)))
		{	while(p-- && (*is++ = *ps++) != '\n')
				;
			p = ps-f->next;
		}
		else
		{	reg int	c = ps[p-1];
			if(c != '\n')
				ps[p-1] = '\n';
			while((*is++ = *ps++) != '\n')
				;
			if(c != '\n')
			{	f->next[p-1] = c;
				if((ps-f->next) >= p)
					is[-1] = c;
			}
		}
#endif

		/* gobble up read data and continue */
		f->next = ps;
		if(is[-1] == '\n')
			break;
		else if(n > 0)
			n -= p;
	}

	if((_Sfi = is - ((uchar*)us)) <= 0)
		us = NIL(char*);
	else if(isgets && is[-1] == '\n')
	{	is[-1] = '\0';
		_Sfi -= 1;
	}
	else	*is = '\0';

	SFOPEN(f,0);
	return us;
}

char*
fgets(char* s, int n, Sfio_t* f)
{
	STDIO_PTR(f, "fgets", char*, (char*, int, Sfio_t*), (s, n, f))

	return _stdgets(f, s, n, 0);
}

char*
gets(char* s)
{
	return _stdgets(sfstdin, s, BUFSIZ, 1);
}
