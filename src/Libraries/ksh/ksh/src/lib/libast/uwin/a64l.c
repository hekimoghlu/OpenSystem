/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
#include "FEATURE/uwin"

#if !_UWIN || _lib_a64l

void _STUB_a64l(){}

#else

#define a64l	______a64l
#define l64a	______l64a

#include	<stdlib.h>
#include	<string.h>

#undef	a64l
#undef	l64a

#if defined(__EXPORT__)
#define extern		__EXPORT__
#endif

static char letter[65] = "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

extern long a64l(const char *str)
{
	register unsigned long ul = 0;
	register int n = 6;
	register int c;
	register char *cp;
	for(n=0; n <6; n++)
	{
		if((c= *str++)==0)
			break;
		if(!(cp=strchr(letter,c)))
			break;
		ul |= (cp-letter)<< (6*n);
	}
	return((long)ul);
}

extern char *l64a(long l)
{
	static char buff[7];
	unsigned ul = ((unsigned long)l & 0xffffffff);
	register char *cp = buff;
	while(ul>0)
	{
		*cp++ = letter[ul&077];
		ul >>= 6;
	}
	*cp = 0;
	return(buff);
}

#endif
