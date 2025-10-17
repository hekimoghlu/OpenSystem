/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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

/*	Read formated data from a stream
**
**	Written by Kiem-Phong Vo.
*/

#if __STD_C
int sfscanf(Sfio_t* f, const char* form, ...)
#else
int sfscanf(va_alist)
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

	rv = (f && form) ? sfvscanf(f,form,args) : -1;
	va_end(args);
	return rv;
}

#if __STD_C
int sfvsscanf(const char* s, const char* form, va_list args)
#else
int sfvsscanf(s, form, args)
char*	s;
char*	form;
va_list	args;
#endif
{
	Sfio_t	f;

	if(!s || !form)
		return -1;

	/* make a fake stream */
	SFCLEAR(&f,NIL(Vtmutex_t*));
	f.flags = SF_STRING|SF_READ;
	f.bits = SF_PRIVATE;
	f.mode = SF_READ;
	f.size = strlen((char*)s);
	f.data = f.next = f.endw = (uchar*)s;
	f.endb = f.endr = f.data+f.size;

	return sfvscanf(&f,form,args);
}

#if __STD_C
int sfsscanf(const char* s, const char* form,...)
#else
int sfsscanf(va_alist)
va_dcl
#endif
{
	va_list		args;
	reg int		rv;
#if __STD_C
	va_start(args,form);
#else
	reg char*	s;
	reg char*	form;
	va_start(args);
	s = va_arg(args,char*);
	form = va_arg(args,char*);
#endif

	rv = (s && form) ? sfvsscanf(s,form,args) : -1;
	va_end(args);
	return rv;
}
