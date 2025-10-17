/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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

/*	Print data with a given format
**
**	Written by Kiem-Phong Vo.
*/

#if __STD_C
int sfprintf(Sfio_t* f, const char* form, ...)
#else
int sfprintf(va_alist)
va_dcl
#endif
{
	va_list	args;
	reg int	rv;

#if __STD_C
	va_start(args,form);
#else
	reg Sfio_t*	f;
	reg char*	form;
	va_start(args);
	f = va_arg(args,Sfio_t*);
	form = va_arg(args,char*);
#endif
	rv = sfvprintf(f,form,args);

	va_end(args);
	return rv;
}

#if __STD_C
ssize_t sfvsprintf(char* s, size_t n, const char* form, va_list args)
#else
ssize_t sfvsprintf(s, n, form, args)
char*	s;
size_t	n;
char*	form;
va_list	args;
#endif
{
	Sfio_t		*f;
	ssize_t		rv;
	Sfnotify_f	notify = _Sfnotify;

	/* make a temp stream */
	_Sfnotify = 0;
	f = sfnew(NIL(Sfio_t*),NIL(char*),(size_t)SF_UNBOUND, -1,SF_WRITE|SF_STRING);
	_Sfnotify = notify;
	if(!f)
		return -1;

	if((rv = sfvprintf(f,form,args)) < 0 )
		return -1;
	if(s && n > 0)
	{	if((rv+1) >= n)
			n--;
		else
			n = rv;
		memcpy(s, f->data, n);
		s[n] = 0;
	}

	sfclose(f);

	_Sfi = rv;

	return rv;
}

#if __STD_C
ssize_t sfsprintf(char* s, size_t n, const char* form, ...)
#else
ssize_t sfsprintf(va_alist)
va_dcl
#endif
{
	va_list	args;
	ssize_t	rv;

#if __STD_C
	va_start(args,form);
#else
	reg char*	s;
	reg size_t	n;
	reg char*	form;
	va_start(args);
	s = va_arg(args,char*);
	n = va_arg(args,size_t);
	form = va_arg(args,char*);
#endif

	rv = sfvsprintf(s,n,form,args);
	va_end(args);

	return rv;
}
