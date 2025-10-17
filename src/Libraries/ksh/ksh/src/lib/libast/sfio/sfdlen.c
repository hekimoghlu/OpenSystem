/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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

/*	Return the length of a double value if coded in a portable format
**
**	Written by Kiem-Phong Vo
*/

#if __STD_C
int _sfdlen(Sfdouble_t v)
#else
int _sfdlen(v)
Sfdouble_t	v;
#endif
{
#define N_ARRAY		(16*sizeof(Sfdouble_t))
	reg int		n, w;
	Sfdouble_t	x;
	int		exp;

	if(v < 0)
		v = -v;

	/* make the magnitude of v < 1 */
	if(v != 0.)
		v = frexpl(v,&exp);
	else	exp = 0;

	for(w = 1; w <= N_ARRAY; ++w)
	{	/* get 2^SF_PRECIS precision at a time */
		n = (int)(x = ldexpl(v,SF_PRECIS));
		v = x-n;
		if(v <= 0.)
			break;
	}

	return 1 + sfulen(exp) + w;
}
