/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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

/*	Read a record delineated by a character.
**	The record length can be accessed via sfvalue(f).
**
**	Written by Kiem-Phong Vo
*/

#if __STD_C
char* sfgetr(Sfio_t *f, int rc, int type)
#else
char* sfgetr(f,rc,type)
Sfio_t*		f;	/* stream to read from	*/
int		rc;	/* record separator	*/
int		type;
#endif
{
	ssize_t		n, un;
	uchar		*s, *ends, *us;
	int		found;
	Sfrsrv_t*	rsrv;
	SFMTXDECL(f); /* declare a local stream variable for multithreading */

	SFMTXENTER(f, NIL(char*));

	if(rc < 0 || (f->mode != SF_READ && _sfmode(f,SF_READ,0) < 0) )
		SFMTXRETURN(f, NIL(char*));
	SFLOCK(f,0);

	/* buffer to be returned */
	rsrv = NIL(Sfrsrv_t*);
	us = NIL(uchar*);
	un = 0;
	found = 0;

	/* compatibility mode */
	type = type < 0 ? SF_LASTR : type == 1 ? SF_STRING : type;

	if(type&SF_LASTR) /* return the broken record */
	{	if((f->flags&SF_STRING) && (un = f->endb - f->next))
		{	us = f->next;
			f->next = f->endb;
			found = 1;
		}
		else if((rsrv = f->rsrv) && (un = -rsrv->slen) > 0)
		{	us = rsrv->data;
			found = 1;
		}
		goto done;
	}

	while(!found)
	{	/* fill buffer if necessary */
		if((n = (ends = f->endb) - (s = f->next)) <= 0)
		{	/* for unseekable devices, peek-read 1 record */
			f->getr = rc;
			f->mode |= SF_RC;

			/* fill buffer the conventional way */
			if(SFRPEEK(f,s,n) <= 0)
			{	us = NIL(uchar*);
				goto done;
			}
			else
			{	ends = s+n;
				if(f->mode&SF_RC)
				{	s = ends[-1] == rc ? ends-1 : ends;
					goto do_copy;
				}
			}
		}

#if _lib_memchr
		if(!(s = (uchar*)memchr((char*)s,rc,n)))
			s = ends;
#else
		while(*s != rc)
			if((s += 1) == ends)
				break;
#endif
	do_copy:
		if(s < ends) /* found separator */
		{	s += 1;		/* include the separator */
			found = 1;

			if(!us &&
			   (!(type&SF_STRING) || !(f->flags&SF_STRING) ||
			    ((f->flags&SF_STRING) && (f->bits&SF_BOTH) ) ) )
			{	/* returning data in buffer */
				us = f->next;
				un = s - f->next;
				f->next = s;
				goto done;
			}
		}

		/* amount to be read */
		n = s - f->next;

		if(!found && (_Sfmaxr > 0 && un+n+1 >= _Sfmaxr || (f->flags&SF_STRING))) /* already exceed limit */
		{	us = NIL(uchar*);
			goto done;
		}

		/* get internal buffer */
		if(!rsrv || rsrv->size < un+n+1)
		{	if(rsrv)
				rsrv->slen = un;
			if((rsrv = _sfrsrv(f,un+n+1)) != NIL(Sfrsrv_t*))
				us = rsrv->data;
			else
			{	us = NIL(uchar*);
				goto done;
			}
		}

		/* now copy data */
		s = us+un;
		un += n;
		ends = f->next;
		f->next += n;
		MEMCPY(s,ends,n);
	}

done:
	_Sfi = f->val = un;
	f->getr = 0;
	if(found && rc != 0 && (type&SF_STRING) )
	{	us[un-1] = '\0';
		if(us >= f->data && us < f->endb)
		{	f->getr = rc;
			f->mode |= SF_GETR;
		}
	}

	/* prepare for a call to get the broken record */
	if(rsrv)
		rsrv->slen = found ? 0 : -un;

	SFOPEN(f,0);

	if(us && (type&SF_LOCKR) )
	{	f->mode |= SF_PEEK|SF_GETR;
		f->endr = f->data;
	}

	SFMTXRETURN(f, (char*)us);
}
