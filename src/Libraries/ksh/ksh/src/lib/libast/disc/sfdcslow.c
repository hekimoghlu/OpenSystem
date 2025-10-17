/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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
#include "sfdchdr.h"

/*	Make a stream op return immediately on interrupts.
**	This is useful on slow streams (hence the name).
**
**	Written by Glenn Fowler (03/18/1998).
*/

#if __STD_C
static int slowexcept(Sfio_t* f, int type, Void_t* v, Sfdisc_t* disc)
#else
static int slowexcept(f, type, v, disc)
Sfio_t*		f;
int		type;
Void_t*		v;
Sfdisc_t*	disc;
#endif
{
	NOTUSED(f);
	NOTUSED(v);
	NOTUSED(disc);

	switch (type)
	{
	case SF_FINAL:
	case SF_DPOP:
		free(disc);
		break;
	case SF_READ:
	case SF_WRITE:
		if (errno == EINTR)
			return(-1);
		break;
	}

	return(0);
}

#if __STD_C
int sfdcslow(Sfio_t* f)
#else
int sfdcslow(f)
Sfio_t*	f;
#endif
{
	Sfdisc_t*	disc;

	if(!(disc = (Sfdisc_t*)malloc(sizeof(Sfdisc_t))) )
		return(-1);

	disc->readf = NIL(Sfread_f);
	disc->writef = NIL(Sfwrite_f);
	disc->seekf = NIL(Sfseek_f);
	disc->exceptf = slowexcept;

	if(sfdisc(f,disc) != disc)
	{	free(disc);
		return(-1);
	}
	sfset(f,SF_IOINTR,1);

	return(0);
}
